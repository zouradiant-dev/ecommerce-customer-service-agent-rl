"""Fail-closed v18 GRPO selection, calibration and qualification guards."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from .experiment_prompt_v18 import V18_TRAINED_PROMPT_VERSION, V18_TRAINED_SYSTEM_PROMPT
from .grpo_memory_v13 import MEMORY_POLICY, summarize_memory
from .local_hf import sha256_file
from .models import TaskSpec
from .online_env import _final_response
from .online_env_v18 import EcommerceCustomerServiceEnvV18
from .v18_readiness import assess_v18_reward_calibration
from .verifier_v18 import TRAIN_REWARD_V18, V18_VERIFIER_VERSION


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
EXPECTED_MANIFEST_SHA256 = "b2564f4ed0d28dab405e6a3cfa19cff85ec29f3451b4f204926f0ee5c2c17c30"
EXPECTED_SFT_ADAPTER_SHA256 = "f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c"

CALIBRATION_BOUND_SOURCES = (
    "src/ecommerce_customer_service/communication.py",
    "src/ecommerce_customer_service/communication_facts.py",
    "src/ecommerce_customer_service/communication_facts_v13.py",
    "src/ecommerce_customer_service/communication_v17.py",
    "src/ecommerce_customer_service/policies.py",
    "src/ecommerce_customer_service/tools.py",
    "src/ecommerce_customer_service/tools_v18.py",
    "src/ecommerce_customer_service/terminal_v18.py",
    "src/ecommerce_customer_service/verifier.py",
    "src/ecommerce_customer_service/verifier_v17.py",
    "src/ecommerce_customer_service/verifier_v18.py",
    "src/ecommerce_customer_service/v18_readiness.py",
    "configs/v18-reward.json",
)

CALIBRATION_RUNTIME_SOURCES = (
    "src/ecommerce_customer_service/grpo_v18.py",
    "scripts/sample_grpo_v18_calibration.py",
    "scripts/evaluate_sft_v18.py",
    "src/ecommerce_customer_service/evaluation_v18.py",
    "src/ecommerce_customer_service/experiment_prompt_v18.py",
    "src/ecommerce_customer_service/local_hf.py",
    "src/ecommerce_customer_service/communication.py",
    "src/ecommerce_customer_service/communication_facts.py",
    "src/ecommerce_customer_service/communication_facts_v13.py",
    "src/ecommerce_customer_service/communication_v17.py",
    "src/ecommerce_customer_service/policies.py",
    "src/ecommerce_customer_service/simulator.py",
    "src/ecommerce_customer_service/tools.py",
    "src/ecommerce_customer_service/tools_v18.py",
    "src/ecommerce_customer_service/terminal_v18.py",
    "src/ecommerce_customer_service/verifier.py",
    "src/ecommerce_customer_service/verifier_v17.py",
    "src/ecommerce_customer_service/verifier_v18.py",
    "src/ecommerce_customer_service/v18_readiness.py",
)

TRAINING_RUNTIME_SOURCES = (
    "scripts/train_grpo_v18.py",
    "scripts/train_grpo_smoke_v12_r1.py",
    "scripts/audit_grpo_smoke_result.py",
    "src/ecommerce_customer_service/communication.py",
    "src/ecommerce_customer_service/communication_facts.py",
    "src/ecommerce_customer_service/communication_facts_v13.py",
    "src/ecommerce_customer_service/communication_v17.py",
    "src/ecommerce_customer_service/database.py",
    "src/ecommerce_customer_service/grpo_v18.py",
    "src/ecommerce_customer_service/grpo_memory_v13.py",
    "src/ecommerce_customer_service/local_hf.py",
    "src/ecommerce_customer_service/online_env.py",
    "src/ecommerce_customer_service/online_env_v18.py",
    "src/ecommerce_customer_service/experiment_prompt_v18.py",
    "src/ecommerce_customer_service/policies.py",
    "src/ecommerce_customer_service/rewards.py",
    "src/ecommerce_customer_service/simulator.py",
    "src/ecommerce_customer_service/training.py",
    "src/ecommerce_customer_service/tools.py",
    "src/ecommerce_customer_service/tools_v18.py",
    "src/ecommerce_customer_service/terminal_v18.py",
    "src/ecommerce_customer_service/verifier.py",
    "src/ecommerce_customer_service/verifier_v17.py",
    "src/ecommerce_customer_service/verifier_v18.py",
    "src/ecommerce_customer_service/v18_readiness.py",
    "src/ecommerce_customer_service/models.py",
    "configs/v18-reward.json",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required:{path}")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required:{path}:line-{number}")
        rows.append(value)
    return rows


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return sha256(payload.encode()).hexdigest()


def source_hashes(root: Path, relatives: Iterable[str]) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in relatives}


def calibration_sources(root: Path, reviewed_rollouts: Path, protocol: Path) -> dict[str, str]:
    sources = source_hashes(root, CALIBRATION_BOUND_SOURCES)
    sources[reviewed_rollouts.resolve().as_posix()] = sha256_file(reviewed_rollouts.resolve())
    sources[protocol.resolve().as_posix()] = sha256_file(protocol.resolve())
    return sources


def build_calibration_receipt(
    root: Path,
    reviewed_rollouts: Path,
    protocol: Path,
) -> dict[str, Any]:
    rows = read_rows(reviewed_rollouts)
    return assess_v18_reward_calibration(
        rows,
        source_sha256=calibration_sources(root, reviewed_rollouts, protocol),
    )


def select_v18_train_tasks(
    manifest: Path,
    *,
    tasks_per_scenario: int,
    scenarios: Iterable[str] | None = None,
    selection_seed: int | None = None,
    excluded_task_ids: Iterable[str] = (),
) -> tuple[list[tuple[Path, TaskSpec]], dict[str, Any]]:
    """Select deterministic, balanced train rows from the frozen v18 manifest."""

    if tasks_per_scenario < 1 or tasks_per_scenario > 100:
        raise ValueError("tasks_per_scenario must be in [1, 100]")
    if selection_seed is not None and (type(selection_seed) is not int or not 0 <= selection_seed < 2**32):
        raise ValueError("invalid_selection_seed")
    excluded = set(excluded_task_ids)
    if any(not isinstance(value,str) or not value.startswith('v18_train_') for value in excluded):
        raise ValueError('excluded_ids_must_be_v18_train_tasks')
    manifest = manifest.resolve()
    manifest_hash = sha256_file(manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("unexpected_v18_manifest_sha256")
    raw_rows = read_rows(manifest)
    forbidden = [str(row.get("task_id") or "<missing>") for row in raw_rows if row.get("split") == "test"]
    if forbidden:
        raise ValueError("test rows are forbidden in v18 GRPO:" + ",".join(forbidden[:5]))
    train_rows = [row for row in raw_rows if row.get("split") == "train"]
    if excluded - {str(row.get('task_id')) for row in train_rows}:
        raise ValueError('excluded_task_not_in_frozen_train_manifest')
    available = {str(row.get("scenario") or "") for row in train_rows}
    requested = {str(value) for value in scenarios or () if str(value)} or available
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError("unknown v18 train scenarios:" + ",".join(unknown))

    groups: dict[str, list[tuple[Path, TaskSpec]]] = defaultdict(list)
    seen: set[str] = set()
    for row in train_rows:
        scenario = str(row.get("scenario") or "")
        if scenario not in requested:
            continue
        task_path = (manifest.parent / str(row["task_file"])).resolve()
        if not task_path.is_file():
            raise FileNotFoundError(f"missing v18 train task:{task_path}")
        task = TaskSpec.from_file(task_path)
        if task.task_id != row.get("task_id") or task.split != "train" or task.scenario != scenario:
            raise ValueError(f"v18 train manifest/task mismatch:{task_path}")
        if task.task_id in seen:
            raise ValueError(f"duplicate v18 train task:{task.task_id}")
        seen.add(task.task_id)
        if not task.task_id.startswith("v18_train_"):
            raise ValueError(f"non-v18 train task id:{task.task_id}")
        if task.metadata.get("scripted_user_replies") or not task.metadata.get("inline_dialogue_v18"):
            raise ValueError(f"non-inline v18 GRPO task:{task.task_id}")
        if not isinstance(task.metadata.get("terminal_v18"), dict):
            raise ValueError(f"missing terminal_v18 metadata:{task.task_id}")
        if not isinstance(task.metadata.get("required_tool_sequence_v18"), list):
            raise ValueError(f"missing v18 tool sequence:{task.task_id}")
        if not task.database_path.is_file():
            raise FileNotFoundError(f"missing v18 train SQLite:{task.database_path}")
        if row.get("task_file_sha256") and sha256_file(task_path) != row["task_file_sha256"]:
            raise ValueError(f"v18 train task hash mismatch:{task.task_id}")
        if row.get("database_sha256") and sha256_file(task.database_path) != row["database_sha256"]:
            raise ValueError(f"v18 train database hash mismatch:{task.task_id}")
        groups[scenario].append((task_path, task))

    selected: list[tuple[Path, TaskSpec]] = []
    counts: Counter[str] = Counter()
    for scenario in sorted(requested):
        values = [item for item in groups.get(scenario, ()) if item[1].task_id not in excluded]
        values = sorted(values, key=lambda item: item[1].task_id if selection_seed is None else
                        sha256(f'{selection_seed}/{item[1].task_id}'.encode()).hexdigest())
        if len(values) < tasks_per_scenario:
            raise ValueError(f"not enough v18 train tasks:{scenario}:{len(values)}<{tasks_per_scenario}")
        chosen = values[:tasks_per_scenario]
        selected.extend(chosen)
        counts[scenario] = len(chosen)
    if not selected:
        raise ValueError("no v18 train tasks selected")
    return selected, {
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "split": "train",
        "test_rows_seen": 0,
        "tasks_per_scenario": tasks_per_scenario,
        "scenarios": sorted(requested),
        "scenario_counts": dict(sorted(counts.items())),
        "selected_tasks": len(selected),
        "task_ids": [task.task_id for _, task in selected],
        "database_sha256": {
            task.task_id: sha256_file(task.database_path) for _, task in selected
        },
        **({'selection_seed':selection_seed,'excluded_task_ids':sorted(excluded)}
           if selection_seed is not None or excluded else {}),
    }


def replay_v18_row(row: dict[str, Any], task_path: Path) -> None:
    """Replay one recorded rollout against a fresh DB and recompute v18 reward."""

    env = EcommerceCustomerServiceEnvV18(task_path=task_path, reward_mode=TRAIN_REWARD_V18)
    try:
        env.reset()
        for call in row.get("tool_executions") or ():
            actual = env._tools.execute(call["name"], call["arguments"]).to_dict()
            if actual != call:
                raise ValueError(f"v18 tool replay mismatch:{row.get('task_id')}:{call.get('name')}")
        final = row.get("final_response") if "final_response" in row else _final_response(row.get("completion"))
        reward = env._evaluate(str(final or ""))
        verification = json.loads(json.dumps(env._last_v18_verification.to_dict()))
        breakdown = json.loads(json.dumps(env._last_v18_breakdown.to_dict()))
        expected_reward = (row.get("reward_components") or {}).get(TRAIN_REWARD_V18, row.get("reward"))
        if reward != expected_reward or verification != row.get("verification") or breakdown != row.get("reward_components"):
            raise ValueError(f"v18 reward/state replay mismatch:{row.get('task_id')}")
    finally:
        env._cleanup()


def validate_calibration_evidence(
    *,
    root: Path,
    receipt_path: Path,
    reviewed_rollouts: Path,
    protocol_path: Path,
    manifest: Path,
    adapter_sha256: str,
) -> dict[str, Any]:
    """Validate reviewed train-only sampling and recompute its calibration receipt."""

    protocol = read_json(protocol_path)
    if (
        protocol.get("schema_version") != "ecommerce_cs_v18_grpo_calibration_protocol_v1"
        or protocol.get("split") != "train"
        or protocol.get("stage") != "sft"
        or protocol.get("policy_controller") is not None
        or protocol.get("adapter_sha256") != adapter_sha256
        or protocol.get("manifest_sha256") != sha256_file(manifest)
        or protocol.get("prompt_version") != V18_TRAINED_PROMPT_VERSION
        or protocol.get("prompt_sha256") != sha256(V18_TRAINED_SYSTEM_PROMPT.encode()).hexdigest()
        or protocol.get("tool_surface") != "native_dynamic"
        or int(protocol.get("trials_per_task", 0)) < 2
        or float(protocol.get("temperature", 0)) <= 0
    ):
        raise ValueError("invalid_or_mismatched_v18_calibration_protocol")
    if protocol.get("source_sha256") != source_hashes(root, CALIBRATION_RUNTIME_SOURCES):
        raise ValueError("v18 calibration runtime sources changed")

    selected, plan = select_v18_train_tasks(
        manifest,
        tasks_per_scenario=int(protocol["tasks_per_scenario"]),
        scenarios=protocol["scenarios"],
        selection_seed=protocol.get('selection_seed'),
        excluded_task_ids=protocol.get('excluded_task_ids',()),
    )
    if (
        protocol.get("task_ids") != plan["task_ids"]
        or protocol.get("database_sha256") != plan["database_sha256"]
    ):
        raise ValueError("v18 calibration task plan changed")
    paths = {task.task_id: path for path, task in selected}
    rows = read_rows(reviewed_rollouts)
    if len(rows) != len(selected) * int(protocol["trials_per_task"]):
        raise ValueError("v18 calibration rollout count mismatch")
    expected_keys = {
        (task.task_id, trial)
        for _, task in selected
        for trial in range(int(protocol["trials_per_task"]))
    }
    actual_keys = {(str(row.get("task_id") or ""), int(row.get("trial", -1))) for row in rows}
    if len(actual_keys) != len(rows) or actual_keys != expected_keys:
        raise ValueError("v18 calibration task/trial coverage mismatch")
    for row in rows:
        if (
            row.get("split") != "train"
            or row.get("stage") != "sft"
            or row.get("prompt_version") != V18_TRAINED_PROMPT_VERSION
            or row.get("verifier_version") != V18_VERIFIER_VERSION
            or row.get("error")
        ):
            raise ValueError(f"invalid v18 calibration rollout:{row.get('rollout_id')}")
        review = row.get("human_review")
        if (
            not isinstance(review, dict)
            or type(review.get("safe")) is not bool
            or type(review.get("communication_passed")) is not bool
            or not str(review.get("evidence") or "").strip()
        ):
            raise ValueError(f"human review missing for v18 calibration rollout:{row.get('rollout_id')}")
        replay_v18_row(row, paths[str(row["task_id"])])
    receipt = read_json(receipt_path)
    recomputed = build_calibration_receipt(root, reviewed_rollouts, protocol_path)
    if receipt != recomputed or receipt.get("reward_calibrated") is not True:
        raise ValueError("v18 reward calibration receipt failed or does not match reviewed evidence")
    return {
        "receipt_sha256": sha256_file(receipt_path),
        "reviewed_rollouts_sha256": sha256_file(reviewed_rollouts),
        "protocol_sha256": sha256_file(protocol_path),
        "metrics": receipt["metrics"],
        "reward_calibrated": True,
        "rows_replayed": len(rows),
        "native_to_static_runtime_gap": True,
    }


def validate_qualification(
    directory: Path,
    *,
    context: dict[str, Any],
    paths: dict[str, Path],
    root: Path,
) -> dict[str, Any]:
    """Recompute every material qualification condition before 100 steps."""

    directory = directory.resolve()
    preflight = read_json(directory / "preflight.json")
    result = read_json(directory / "complete.json")
    expected = {**context, "phase": "qualification", "max_steps": 10, "save_steps": 10}
    if any(preflight.get(key) != value for key, value in expected.items()):
        raise ValueError("v18 qualification context mismatch")
    if (
        result.get("complete") is not True
        or result.get("phase") != "qualification"
        or result.get("optimizer_steps") != 10
        or result.get("nonzero_gradient_steps", 0) < 1
        or result.get("mixed_group_rate", 0) < 0.20
        or any(
            result.get(key) is not True
            for key in (
                "actual_policy_update_verified",
                "adapter_save_reload_verified",
                "sft_reference_unchanged",
                "source_databases_unchanged",
            )
        )
        or result.get("memory_headroom_gib", 0) < 1.0
        or result.get("memory_policy") != MEMORY_POLICY
    ):
        raise ValueError("v18 qualification did not prove signal, update and memory safety")
    required_artifacts = {
        "preflight.json",
        "initialization.json",
        "online_rewards.jsonl",
        "gradient_audit.jsonl",
        "adapter_model.safetensors",
        "ref/adapter_model.safetensors",
        "memory_audit.jsonl",
        "memory_report.json",
    }
    if set(result.get("artifacts_sha256") or {}) != required_artifacts:
        raise ValueError("v18 qualification artifact set mismatch")
    for name, expected_hash in result["artifacts_sha256"].items():
        if sha256_file(directory / name) != expected_hash:
            raise ValueError("v18 qualification artifact changed:" + name)
    memory = summarize_memory(read_rows(directory / "memory_audit.jsonl"), 10)
    if (
        read_json(directory / "memory_report.json") != memory
        or not memory["memory_safety_passed"]
        or result["memory_headroom_gib"] != memory["memory_headroom_gib"]
    ):
        raise ValueError("v18 qualification memory evidence mismatch")
    gradients = read_rows(directory / "gradient_audit.jsonl")
    if len(gradients) != 10 or [row.get("next_optimizer_step") for row in gradients] != list(range(1, 11)):
        raise ValueError("v18 qualification gradient evidence incomplete")
    rows = read_rows(directory / "online_rewards.jsonl")
    if len(rows) != 80:
        raise ValueError("v18 qualification must contain exactly 80 online rollouts")
    mixed = 0
    for batch in range(1, 11):
        batch_rows = [row for row in rows if row.get("batch") == batch]
        if len(batch_rows) != 8:
            raise ValueError("v18 qualification reward batch incomplete")
        for start in (0, 4):
            group = batch_rows[start : start + 4]
            if len({row.get("task_id") for row in group}) != 1:
                raise ValueError("v18 qualification reward group mixes tasks")
            mixed += len({row.get("reward") for row in group}) > 1
    if mixed / 20 != result["mixed_group_rate"] or mixed / 20 < 0.20:
        raise ValueError("v18 qualification mixed-group evidence mismatch")
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if (
            task_id not in paths
            or row.get("split") != "train"
            or row.get("verifier_version") != V18_VERIFIER_VERSION
            or row.get("reward_mode") != TRAIN_REWARD_V18
        ):
            raise ValueError("v18 qualification online row mismatch")
        replay_v18_row(row, paths[task_id])
    if preflight.get("source_sha256") != source_hashes(root, TRAINING_RUNTIME_SOURCES):
        raise ValueError("v18 qualification training sources changed")
    return {
        "directory": str(directory),
        "complete_sha256": sha256_file(directory / "complete.json"),
        "online_rollouts_replayed": len(rows),
        "mixed_group_rate": mixed / 20,
        "formal_rl_gate_passed": False,
    }
