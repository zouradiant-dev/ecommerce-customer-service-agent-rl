"""Dev-only learned-model evaluation for the opt-in v18 protocol.

The v18 route has a stricter tool state machine, structured terminal contract,
and reward implementation than the historical evaluator.  This module keeps
those semantics together and deliberately refuses test rows: model and
protocol selection must happen on v18 dev before a future test-v2 is opened.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable

from .experiment_prompt_v18 import V18_PROMPT_VERSION
from .models import TaskSpec
from .simulator import Agent
from .tasks import load_task_paths
from .tools import WRITE_TOOLS
from .tools_v18 import CustomerServiceToolsV18, TOOLS_V18_VERSION
from .verifier_v18 import V18_VERIFIER_VERSION, score_v18, verify_task_v18


V18_EVALUATION_VERSION = "ecommerce_cs_v18_learned_eval_v1"
_SAFE_STAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CRITICAL_WRITES = {"cancel_order", "update_shipping_address", "create_refund"}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_stage(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("stage must use NAME=MODEL")
    name, model = (item.strip() for item in value.split("=", 1))
    if not _SAFE_STAGE.fullmatch(name):
        raise ValueError(f"invalid stage name:{name!r}")
    if not model:
        raise ValueError("stage model must be non-empty")
    return name, model


def select_v18_dev_tasks(
    manifest: Path,
    *,
    max_tasks_per_scenario: int = 0,
    scenarios: Iterable[str] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    """Validate a frozen v18 train/dev manifest and select only dev tasks."""

    manifest = manifest.resolve()
    if max_tasks_per_scenario < 0:
        raise ValueError("max_tasks_per_scenario must be non-negative")
    scenario_filter = {str(value) for value in scenarios or () if str(value)}
    raw_rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    forbidden = [str(row.get("task_id") or "<missing>") for row in raw_rows if row.get("split") == "test"]
    if forbidden:
        raise ValueError("test rows are forbidden in v18 dev evaluation:" + ",".join(forbidden[:5]))

    selected: list[Path] = []
    scenario_counts: Counter[str] = Counter()
    all_dev = load_task_paths(manifest, split="dev")
    dev_rows = [row for row in raw_rows if row.get("split") == "dev"]
    seen: set[str] = set()
    if not all_dev:
        raise ValueError("v18 manifest contains no dev tasks")
    available_scenarios = {str(row.get("scenario") or "") for row in dev_rows}
    unknown_scenarios = sorted(scenario_filter - available_scenarios)
    if unknown_scenarios:
        raise ValueError("unknown v18 dev scenarios:" + ",".join(unknown_scenarios))
    for path, row in zip(all_dev, dev_rows):
        if not path.is_file():
            raise FileNotFoundError(f"missing v18 dev task file; upload the complete evaluator data package:{path}")
        task = TaskSpec.from_file(path)
        if task.task_id != row.get("task_id"):
            raise ValueError(f"manifest_task_id_mismatch:{path}")
        if task.task_id in seen:
            raise ValueError(f"duplicate v18 dev task:{task.task_id}")
        seen.add(task.task_id)
        if task.split != "dev":
            raise ValueError(f"non-dev task selected:{task.task_id}:{task.split}")
        if not task.task_id.startswith("v18_"):
            raise ValueError(f"non-v18 task id:{task.task_id}")
        if not isinstance(task.metadata.get("terminal_v18"), dict):
            raise ValueError(f"missing terminal_v18 metadata:{task.task_id}")
        if not isinstance(task.metadata.get("required_tool_sequence_v18"), list):
            raise ValueError(f"missing required_tool_sequence_v18:{task.task_id}")
        if not task.database_path.is_file():
            raise FileNotFoundError(f"missing v18 dev SQLite; upload the complete evaluator data package:{task.database_path}")
        if row.get("database") and task.database_path != (manifest.parent / row["database"]).resolve():
            raise ValueError(f"manifest_database_path_mismatch:{task.task_id}")
        for payload_path, hash_key in (
            (path, "task_file_sha256"),
            (task.database_path, "database_sha256"),
        ):
            if row.get(hash_key) and sha256_file(payload_path) != row[hash_key]:
                raise ValueError(f"frozen_dev_payload_hash_mismatch:{task.task_id}:{hash_key}")
        if scenario_filter and task.scenario not in scenario_filter:
            continue
        if max_tasks_per_scenario and scenario_counts[task.scenario] >= max_tasks_per_scenario:
            continue
        selected.append(path)
        scenario_counts[task.scenario] += 1

    return selected, {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "all_dev_tasks": len(all_dev),
        "selected_tasks": len(selected),
        "max_tasks_per_scenario": max_tasks_per_scenario,
        "scenario_filter": sorted(scenario_filter),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "split": "dev",
        "test_rows_seen": 0,
    }


def _decode_arguments(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, dict):
        return dict(value), None
    if not isinstance(value, str):
        return None, "arguments_not_json_object"
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return None, "arguments_invalid_json"
    if not isinstance(parsed, dict):
        return None, "arguments_not_json_object"
    return parsed, None


def audit_tool_call_format(messages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Check raw model tool calls without repairing them."""

    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in CustomerServiceToolsV18.schemas()}
    failures: list[str] = []
    calls = 0
    for turn, message in enumerate(messages):
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            failures.append(f"turn_{turn}:tool_calls_not_list")
            continue
        for index, call in enumerate(raw_calls):
            calls += 1
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                failures.append(f"turn_{turn}_call_{index}:missing_function")
                continue
            name = str(function.get("name") or call.get("name") or "")
            if name not in schemas:
                failures.append(f"turn_{turn}_call_{index}:unknown_tool:{name}")
                continue
            arguments, error = _decode_arguments(function.get("arguments", call.get("arguments", "{}")))
            if error:
                failures.append(f"turn_{turn}_call_{index}:{error}")
                continue
            assert arguments is not None
            parameters = schemas[name]
            required = set(parameters.get("required") or [])
            properties = dict(parameters.get("properties") or {})
            missing = sorted(required - set(arguments))
            extra = sorted(set(arguments) - set(properties))
            if missing:
                failures.append(f"turn_{turn}_call_{index}:missing:{','.join(missing)}")
            if extra and parameters.get("additionalProperties") is False:
                failures.append(f"turn_{turn}_call_{index}:extra:{','.join(extra)}")
            for key, argument in arguments.items():
                expected = (properties.get(key) or {}).get("type")
                valid = (
                    expected == "string" and isinstance(argument, str)
                    or expected == "boolean" and type(argument) is bool
                    or expected == "integer" and isinstance(argument, int) and not isinstance(argument, bool)
                    or expected not in {"string", "boolean", "integer"}
                )
                if not valid:
                    failures.append(f"turn_{turn}_call_{index}:type:{key}:{expected}")
    return {"passed": not failures, "calls": calls, "failures": failures}


def _run_task_v18(
    task: TaskSpec,
    agent: Agent,
    *,
    trial: int,
    stage: str,
    model: str,
    prompt_version: str = V18_PROMPT_VERSION,
    required_split: str,
) -> dict[str, Any]:
    """Run one learned-model episode against an isolated v18 database."""

    if task.split != required_split:
        raise ValueError(
            f"v18 learned rollout requires {required_split}:{task.task_id}:{task.split}"
        )
    with tempfile.TemporaryDirectory(prefix="ecommerce-v18-eval-") as temporary:
        database = Path(temporary) / "state.sqlite"
        shutil.copy2(task.database_path, database)
        isolated = replace(task, database_path=database)
        tools = CustomerServiceToolsV18(database, isolated)
        tools.observe_user_message(isolated.user_prompt)
        messages: tuple[dict[str, Any], ...] = ()
        final_response = ""
        stopped_reason = "worker_error"
        error: str | None = None
        try:
            run = agent.run(isolated, tools)
            messages = run.messages
            final_response = run.final_response
            stopped_reason = run.stopped_reason
        except Exception as exc:  # A failed episode must be recorded, not disappear.
            error = f"{type(exc).__name__}: {exc}"

        calls = [item.to_dict() for item in tools.executions]
        verification = verify_task_v18(isolated, final_response, calls)
        if error:
            verification = replace(
                verification,
                passed=False,
                failures=tuple(dict.fromkeys((*verification.failures, "worker_error"))),
            )
        reward = score_v18(
            isolated,
            verification,
            calls,
            stopped_reason="error" if error else stopped_reason,
        )
        format_audit = audit_tool_call_format(messages)
        return {
            "schema_version": "ecommerce_cs_v18_learned_rollout_v1",
            "evaluation_version": V18_EVALUATION_VERSION,
            "rollout_id": f"{stage}:{task.task_id}:trial-{trial}",
            "stage": stage,
            "model": model,
            "task_id": task.task_id,
            "split": task.split,
            "scenario": task.scenario,
            "trial": trial,
            "reward": reward.strict_binary_v18,
            "messages": list(messages),
            "tool_executions": calls,
            "tool_format": format_audit,
            "final_response": final_response,
            "verification": verification.to_dict(),
            "verifier_version": V18_VERIFIER_VERSION,
            "reward_components": reward.to_dict(),
            "stopped_reason": stopped_reason,
            "error": error,
            "prompt_version": prompt_version,
            "tools_version": TOOLS_V18_VERSION,
            "learned_model_result": True,
        }


def run_task_v18(
    task: TaskSpec,
    agent: Agent,
    *,
    trial: int,
    stage: str,
    model: str,
    prompt_version: str = V18_PROMPT_VERSION,
) -> dict[str, Any]:
    """Run one frozen-dev episode. Train/test rows remain forbidden here."""

    return _run_task_v18(
        task,
        agent,
        trial=trial,
        stage=stage,
        model=model,
        prompt_version=prompt_version,
        required_split="dev",
    )


def run_train_task_v18(
    task: TaskSpec,
    agent: Agent,
    *,
    trial: int,
    stage: str,
    model: str,
    prompt_version: str = V18_PROMPT_VERSION,
) -> dict[str, Any]:
    """Run one train-only calibration episode without opening dev or test."""

    return _run_task_v18(
        task,
        agent,
        trial=trial,
        stage=stage,
        model=model,
        prompt_version=prompt_version,
        required_split="train",
    )


def _safe_rate(numerator: int | float, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict = [float(row["reward"]) for row in rows]
    shaped = [float(row["reward_components"]["train_shaped_v18"]) for row in rows]
    task_rewards: dict[str, list[float]] = defaultdict(list)
    successful_writes: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    policy_violations: Counter[str] = Counter()
    for row in rows:
        task_rewards[str(row["task_id"])].append(float(row["reward"]))
        failure_counts.update(str(item) for item in row["verification"].get("failures") or [])
        for call in row.get("tool_executions") or []:
            if call.get("ok") and call.get("name") in WRITE_TOOLS:
                successful_writes[str(call["name"])] += 1
            if call.get("policy_violation"):
                policy_violations[str(call["policy_violation"])] += 1
    mixed = sum(len(values) >= 2 and len(set(values)) >= 2 for values in task_rewards.values())
    pass_at_k = sum(any(value >= 1.0 for value in values) for values in task_rewards.values())
    return {
        "tasks": len(task_rewards),
        "rollouts": len(rows),
        "strict_passes": sum(value >= 1.0 for value in strict),
        "strict_pass_rate": _safe_rate(sum(value >= 1.0 for value in strict), len(rows)),
        "pass_at_k": _safe_rate(pass_at_k, len(task_rewards)),
        "mean_train_shaped_v18": sum(shaped) / len(shaped) if shaped else None,
        "tool_format_stable_rate": _safe_rate(
            sum(bool(row["tool_format"]["passed"]) for row in rows), len(rows)
        ),
        "structured_terminal_rate": _safe_rate(
            sum(bool(row["verification"]["terminal_passed"]) for row in rows), len(rows)
        ),
        "safety_gate_pass_rate": _safe_rate(
            sum(bool(row["verification"]["safety_gate_passed"]) for row in rows), len(rows)
        ),
        "hard_safety_failure_rollouts": sum(
            bool(row["verification"].get("hard_safety_failures")) for row in rows
        ),
        "worker_errors": sum(bool(row.get("error")) for row in rows),
        "max_step_exhaustions": sum(row.get("stopped_reason") == "max_steps" for row in rows),
        "mixed_group_rate": _safe_rate(mixed, len(task_rewards)),
        "successful_write_counts": dict(sorted(successful_writes.items())),
        "missing_critical_write_coverage": sorted(_CRITICAL_WRITES - set(successful_writes)),
        "failure_counts": dict(failure_counts.most_common()),
        "policy_violation_counts": dict(policy_violations.most_common()),
    }


def summarize_v18_rollouts(
    rows: Iterable[dict[str, Any]],
    *,
    expected_tasks: int | None = None,
    trials_per_task: int | None = None,
) -> dict[str, Any]:
    materialized = list(rows)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_scenario[str(row["scenario"])].append(row)
    overall = _aggregate(materialized)
    expected_rollouts = (
        expected_tasks * trials_per_task
        if expected_tasks is not None and trials_per_task is not None
        else None
    )
    return {
        "schema_version": "ecommerce_cs_v18_evaluation_summary_v1",
        "evaluation_version": V18_EVALUATION_VERSION,
        "complete": expected_rollouts is None or len(materialized) == expected_rollouts,
        "expected_tasks": expected_tasks,
        "trials_per_task": trials_per_task,
        "expected_rollouts": expected_rollouts,
        "overall": overall,
        "by_scenario": {
            scenario: _aggregate(scenario_rows)
            for scenario, scenario_rows in sorted(by_scenario.items())
        },
    }


def compare_v18_stages(
    baseline: Iterable[dict[str, Any]],
    candidate: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    left = {(str(row["task_id"]), int(row["trial"])): float(row["reward"]) for row in baseline}
    right = {(str(row["task_id"]), int(row["trial"])): float(row["reward"]) for row in candidate}
    if set(left) != set(right):
        raise ValueError("paired stage comparison requires identical task/trial keys")
    wins = losses = ties = 0
    for key in sorted(left):
        delta = right[key] - left[key]
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
    count = len(left)
    left_rate = _safe_rate(sum(value >= 1.0 for value in left.values()), count)
    right_rate = _safe_rate(sum(value >= 1.0 for value in right.values()), count)
    return {
        "paired_rollouts": count,
        "baseline_pass_rate": left_rate,
        "candidate_pass_rate": right_rate,
        "pass_rate_delta": right_rate - left_rate,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }
