from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .diagnostics import classify_bad_case
from .tools import CustomerServiceTools


VALUABLE_FAILURES = {
    "wrong_tool_selection",
    "argument_binding_error",
    "premature_write",
    "premature_escalation",
    "unsupported_success_claim",
    "missing_confirmation",
    "unknown_tool",
    "repeated_tool_loop",
    "tool_error_recovery_failure",
    "policy_violation",
    "state_not_updated",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{number}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def merge_successful_retries(
    base_path: Path,
    retry_paths: Iterable[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Replace failed base rows with successful retry rows while preserving lineage."""

    base_path = base_path.resolve()
    retry_paths = [path.resolve() for path in retry_paths]
    output_path = output_path.resolve()
    base_rows = load_jsonl(base_path)
    base_ids = [str(row.get("task_id") or "") for row in base_rows]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("Base rollout file contains duplicate task IDs")

    selected = {task_id: dict(row) for task_id, row in zip(base_ids, base_rows)}
    base_failed = {
        task_id
        for task_id, row in selected.items()
        if float(row.get("reward") or 0.0) < 1.0 - 1e-6
    }
    selected_source = {task_id: str(base_path) for task_id in base_ids}
    replacements_by_source: dict[str, int] = {}

    for retry_path in retry_paths:
        rows = load_jsonl(retry_path)
        retry_ids = [str(row.get("task_id") or "") for row in rows]
        if len(retry_ids) != len(set(retry_ids)):
            raise ValueError(f"Retry file contains duplicate task IDs: {retry_path}")
        unknown = set(retry_ids) - set(selected)
        if unknown:
            raise ValueError(f"Retry file contains unknown task IDs: {', '.join(sorted(unknown))}")
        replacements = 0
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if task_id not in base_failed or float(row.get("reward") or 0.0) < 1.0 - 1e-6:
                continue
            selected[task_id] = dict(row)
            selected_source[task_id] = str(retry_path)
            replacements += 1
        replacements_by_source[str(retry_path)] = replacements

    merged_rows: list[dict[str, Any]] = []
    for task_id in base_ids:
        row = dict(selected[task_id])
        row["selection_metadata"] = {
            "method": "successful_retry_replaces_failed_base",
            "base_rollouts": str(base_path),
            "selected_source": selected_source[task_id],
        }
        merged_rows.append(row)
    write_jsonl(output_path, merged_rows)
    still_failed = sum(float(row.get("reward") or 0.0) < 1.0 - 1e-6 for row in merged_rows)
    return {
        "base_rollouts": str(base_path),
        "retry_rollouts": [str(path) for path in retry_paths],
        "output": str(output_path),
        "rows": len(merged_rows),
        "base_failed": len(base_failed),
        "replacements": sum(source != str(base_path) for source in selected_source.values()),
        "replacements_by_source": replacements_by_source,
        "passed": len(merged_rows) - still_failed,
        "still_failed": still_failed,
    }


def combine_rollout_files(input_paths: Iterable[Path], output_path: Path) -> dict[str, Any]:
    """Combine disjoint rollout files and retain per-row source provenance."""

    input_paths = [path.resolve() for path in input_paths]
    output_path = output_path.resolve()
    if not input_paths:
        raise ValueError("At least one rollout input is required")
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    split_counts: dict[str, int] = defaultdict(int)
    for input_path in input_paths:
        for source in load_jsonl(input_path):
            task_id = str(source.get("task_id") or "")
            if not task_id:
                raise ValueError(f"Rollout without task_id in {input_path}")
            if task_id in seen:
                raise ValueError(f"Duplicate task ID across rollout inputs: {task_id}")
            seen.add(task_id)
            row = dict(source)
            row["combination_metadata"] = {
                "method": "combine_disjoint_rollout_files",
                "source": str(input_path),
            }
            combined.append(row)
            split_counts[str(row.get("split") or "unknown")] += 1
    combined.sort(key=lambda row: (str(row.get("split") or ""), str(row["task_id"])))
    write_jsonl(output_path, combined)
    passed = sum(float(row.get("reward") or 0.0) >= 1.0 - 1e-6 for row in combined)
    return {
        "inputs": [str(path) for path in input_paths],
        "output": str(output_path),
        "rows": len(combined),
        "passed": passed,
        "failed": len(combined) - passed,
        "split_counts": dict(sorted(split_counts.items())),
    }


def tool_sequence_signature(row: dict[str, Any]) -> str:
    sequence = [
        {"name": item.get("name"), "arguments": item.get("arguments") or {}}
        for item in row.get("tool_executions") or []
    ]
    encoded = json.dumps(sequence, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deduplicate_by_tool_sequence(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("task_id")), tool_sequence_signature(row))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def curate_trajectories(
    rows: Iterable[dict[str, Any]],
    *,
    keep_successes: bool = True,
    keep_valuable_failures: bool = True,
    max_per_scenario: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = defaultdict(int)
    for row in deduplicate_by_tool_sequence(rows):
        success = float(row.get("reward") or 0.0) >= 1.0 - 1e-6
        categories = set(classify_bad_case(row))
        valuable = bool(categories & VALUABLE_FAILURES)
        reason = None
        if success and not keep_successes:
            reason = "successes_disabled"
        elif not success and not (keep_valuable_failures and valuable):
            reason = "low_value_failure"
        scenario = str(row.get("scenario") or "unknown")
        if reason is None and max_per_scenario is not None and scenario_counts[scenario] >= max_per_scenario:
            reason = "scenario_cap"
        if reason:
            rejected.append({"task_id": row.get("task_id"), "reason": reason, "bad_case_categories": sorted(categories)})
            continue
        enriched = dict(row)
        enriched["bad_case_categories"] = sorted(categories)
        kept.append(enriched)
        scenario_counts[scenario] += 1
    return kept, rejected


def build_sft_dataset(
    rollout_path: Path,
    output_dir: Path,
    *,
    tool_scope: str = "all",
) -> dict[str, Any]:
    if tool_scope not in {"all", "trajectory", "stateful_turn"}:
        raise ValueError(f"unsupported tool scope: {tool_scope}")
    rows = deduplicate_by_tool_sequence(load_jsonl(rollout_path))
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    skipped = 0
    for row in rows:
        if float(row.get("reward") or 0.0) < 1.0 - 1e-6 or row.get("split") == "test":
            skipped += 1
            continue
        split = "dev" if row.get("split") == "dev" else "train"
        messages = list(row.get("messages") or [])
        if tool_scope == "stateful_turn":
            for turn, index in enumerate(
                position for position, message in enumerate(messages) if message.get("role") == "assistant"
            ):
                prefix = messages[:index]
                training_messages = messages[: index + 1]
                if messages[index].get("tool_calls"):
                    cursor = index + 1
                    while cursor < len(messages) and messages[cursor].get("role") == "tool":
                        training_messages.append(messages[cursor])
                        cursor += 1
                tools = stateful_tool_schemas(prefix)
                visible = {item["function"]["name"] for item in tools}
                requested = {
                    str((call.get("function") or call).get("name") or "")
                    for call in messages[index].get("tool_calls") or []
                }
                if not requested.issubset(visible):
                    missing = ",".join(sorted(requested - visible))
                    raise ValueError(f"target_tool_not_visible:{row['task_id']}:{turn}:{missing}")
                loss_mask = [False] * len(training_messages)
                loss_mask[index] = True
                by_split[split].append(
                    {
                        "id": f"sft_{row['task_id']}_trial{row.get('trial', 0)}_turn{turn}",
                        "task_id": row["task_id"],
                        "scenario": row.get("scenario"),
                        "format_version": "ecommerce_cs_agent_sft_v1",
                        "messages": training_messages,
                        "tools": tools,
                        "loss_mask": loss_mask,
                        "metadata": {
                            "source": "verified_teacher_trajectory",
                            "split": split,
                            "reward": row.get("reward"),
                            "tool_scope": tool_scope,
                            "assistant_turn": turn,
                        },
                    }
                )
            continue
        tools = CustomerServiceTools.schemas()
        if tool_scope == "trajectory":
            used_names = {str(item.get("name") or "") for item in row.get("tool_executions") or []}
            tools = [item for item in tools if item["function"]["name"] in used_names]
        by_split[split].append(
            {
                "id": f"sft_{row['task_id']}_trial{row.get('trial', 0)}",
                "task_id": row["task_id"],
                "scenario": row.get("scenario"),
                "format_version": "ecommerce_cs_agent_sft_v1",
                "messages": messages,
                "tools": tools,
                "loss_mask": [message.get("role") == "assistant" for message in messages],
                "metadata": {
                    "source": "verified_teacher_trajectory",
                    "split": split,
                    "reward": row.get("reward"),
                    "tool_scope": tool_scope,
                },
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", by_split["train"])
    write_jsonl(output_dir / "dev.jsonl", by_split["dev"])
    return {
        "train": len(by_split["train"]),
        "dev": len(by_split["dev"]),
        "skipped": skipped,
        "tool_scope": tool_scope,
    }


def stateful_tool_schemas(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct the deployment tool surface immediately after a message prefix."""

    successful: set[str] = set()
    failed_counts: dict[str, int] = defaultdict(int)
    observed_user_messages: list[str] = []
    for message in messages:
        role = str(message.get("role") or "")
        if role == "user":
            observed_user_messages.append(str(message.get("content") or ""))
            continue
        if role != "tool":
            continue
        try:
            content = json.loads(str(message.get("content") or "{}"))
        except json.JSONDecodeError:
            continue
        if isinstance(content, dict):
            name = str(message.get("name") or "")
            if "error" in content:
                failed_counts[name] += 1
            else:
                successful.add(name)
    return CustomerServiceTools.schemas_for_state(
        identity_verified="verify_customer_identity" in successful,
        successful_tools=successful,
        observed_user_messages=observed_user_messages,
        failed_tool_counts=dict(failed_counts),
    )


def build_offline_group_dataset(rollout_path: Path, output_path: Path) -> dict[str, int]:
    """Build mixed-outcome static groups for offline AWR-style baselines."""

    rows = load_jsonl(rollout_path)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "test":
            continue
        by_task[str(row.get("task_id"))].append(row)
    selected: list[dict[str, Any]] = []
    skipped_groups = 0
    for group in by_task.values():
        outcomes = {float(item.get("reward") or 0.0) >= 1.0 - 1e-6 for item in group}
        if len(group) < 2 or len(outcomes) < 2:
            skipped_groups += 1
            continue
        selected.extend(group)
    write_jsonl(output_path, selected)
    return {
        "tasks": len({str(item["task_id"]) for item in selected}),
        "rollouts": len(selected),
        "skipped_zero_variance_groups": skipped_groups,
    }


def build_grpo_dataset(rollout_path: Path, output_path: Path) -> dict[str, int]:
    """Compatibility alias for the former name; this does not perform online GRPO."""

    return build_offline_group_dataset(rollout_path, output_path)


def assert_no_template_leakage(task_manifest: Path) -> dict[str, list[str]]:
    rows = load_jsonl(task_manifest)
    splits_by_template: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        template = str((row.get("metadata") or {}).get("template_id") or row.get("scenario"))
        splits_by_template[template].add(str(row.get("split")))
    leaks = {template: sorted(splits) for template, splits in splits_by_template.items() if len(splits) > 1}
    if leaks:
        raise ValueError(f"Task template leakage detected: {leaks}")
    return {template: sorted(splits) for template, splits in sorted(splits_by_template.items())}
