"""CPU-only train selection and engineering checks, not the formal RL gate."""
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from .local_hf import sha256_file
from .models import TaskSpec


DEFAULT_TEMPLATES = (
    "cancel_unshipped", "update_address", "damaged_item_refund",
    "partial_quantity_refund", "shipment_status", "identity_tool_transient_error",
    "refund_tool_transient_error", "persistent_tool_error_escalation",
)
CRITICAL_WRITES = {"cancel_order", "update_shipping_address", "create_refund"}


def select_train_tasks(manifest: Path, *, templates: tuple[str, ...] = DEFAULT_TEMPLATES,
                       per_template: int = 1) -> list[tuple[Path, TaskSpec]]:
    if per_template < 1 or not templates or len(set(templates)) != len(templates):
        raise ValueError("positive count and distinct nonempty templates required")
    manifest = manifest.resolve()
    groups: dict[str, list[tuple[Path, TaskSpec]]] = defaultdict(list)
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        path = (manifest.parent / row["task_file"]).resolve()
        task = TaskSpec.from_file(path)
        if task.split != "train" or task.task_id != row["task_id"]:
            raise ValueError(f"manifest_task_mismatch:{path}")
        if task.task_id in seen:
            raise ValueError(f"duplicate_task_id:{task.task_id}")
        seen.add(task.task_id)
        if not task.database_path.is_file():
            raise FileNotFoundError(task.database_path)
        # The current TRL environment cannot manufacture further user turns.
        if task.metadata.get("scripted_user_replies"):
            continue
        template = str(task.metadata.get("template_id") or task.scenario)
        groups[template].append((path, task))
    selected = []
    for template in templates:
        values = sorted(groups.get(template, []), key=lambda item: item[1].task_id)
        if len(values) < per_template:
            raise ValueError(f"not_enough_inline_train_tasks:{template}:{len(values)}<{per_template}")
        selected.extend(values[:per_template])
    return selected


def sampling_checks(rows: list[dict[str, Any]], *, trials_per_task: int,
                    expected_task_ids: list[str]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate = False
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("readiness requires actual train rollouts")
        key = (row["task_id"], row["trial"])
        duplicate |= key in seen
        seen.add(key)
        if row.get("reward") not in (0.0, 1.0):
            raise ValueError("strict binary reward required")
        if bool(row["verifier"]["passed"]) != bool(row["reward"]):
            raise ValueError("reward_verifier_mismatch")
        by_task[row["task_id"]].append(row)
    mixed = [task_id for task_id, values in by_task.items()
             if {row["reward"] for row in values} == {0.0, 1.0}]
    writes = {name: {"attempts": 0, "successful_calls": 0, "passing_rollouts": 0}
              for name in sorted(CRITICAL_WRITES)}
    for row in rows:
        for name, counters in writes.items():
            calls = [call for call in row["tool_executions"] if call["name"] == name]
            counters["attempts"] += len(calls)
            counters["successful_calls"] += sum(bool(call["ok"]) for call in calls)
            counters["passing_rollouts"] += int(row["reward"] == 1.0 and any(call["ok"] for call in calls))
    malformed = sum(row["stopped_reason"] == "invalid_tool_call" for row in rows)
    truncated = sum(any(message.get("local_finish_reason") == "length" for message in row["messages"])
                    for row in rows)
    checks = {
        "complete_train_groups": not duplicate and set(by_task) == set(expected_task_ids)
            and all({row["trial"] for row in values} == set(range(trials_per_task))
                    for values in by_task.values()),
        "multiple_trials": trials_per_task >= 2,
        "no_worker_errors": all(not row.get("error") and row["stopped_reason"] != "worker_error" for row in rows),
        "no_invalid_calls_or_truncation": malformed == 0 and truncated == 0,
        "no_max_step_exhaustion": all(row["stopped_reason"] != "max_steps" for row in rows),
        "positive_write_coverage": all(counters["passing_rollouts"] > 0 for counters in writes.values()),
        "mixed_groups_at_least_20pct": bool(by_task) and len(mixed) / len(by_task) >= 0.20,
    }
    return {
        "engineering_sampling_checks_passed": all(checks.values()),
        "checks": checks, "blocking_reasons": [name for name, ok in checks.items() if not ok],
        "tasks": len(by_task), "rollouts": len(rows), "mixed_task_ids": sorted(mixed),
        "mixed_group_rate": len(mixed) / max(1, len(by_task)), "writes": writes,
        "per_task_successes": {key: sum(row["reward"] == 1.0 for row in value) for key, value in sorted(by_task.items())},
        "reward_calibrated": False, "formal_rl_gate_passed": False,
        "limitations": ["Train diagnostic, not held-out performance.",
                       "Native dynamic-tool sampler is not the TRL static-tool training loop.",
                       "Human review of positive AND negative rewards is still required.",
                       "GPU training/runtime/backward/save-load have not been validated by these checks."],
    }


def validate_smoke_evidence(directory: Path, *, adapter_sha256: str, manifest_sha256: str) -> dict[str, Any]:
    """A calibrated train pilot permits only a bounded engineering smoke."""
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    readiness = json.loads((directory / "readiness.json").read_text(encoding="utf-8"))
    review = json.loads((directory / "reward_review.json").read_text(encoding="utf-8"))
    rollout_path = directory / "sft" / "rollouts.jsonl"
    if not summary.get("complete") or protocol.get("split") != "train":
        raise ValueError("complete_train_pilot_required")
    if protocol.get("adapter_sha256") != adapter_sha256 or protocol.get("manifest_sha256") != manifest_sha256:
        raise ValueError("pilot_weights_or_manifest_mismatch")
    if review.get("rollouts_sha256") != sha256_file(rollout_path):
        raise ValueError("review_rollout_hash_mismatch")
    rows = [json.loads(line) for line in rollout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fresh = sampling_checks(rows, trials_per_task=protocol["trials_per_task"], expected_task_ids=protocol["task_ids"])
    if not fresh["engineering_sampling_checks_passed"] or fresh != readiness:
        raise ValueError("engineering_sampling_checks_failed_or_stale")
    verdicts = review.get("rows") or []
    keys = {(row["task_id"], row["trial"]): row["reward"] for row in rows}
    reviewed = {(row.get("task_id"), row.get("trial")): row for row in verdicts}
    if len(verdicts) != len(keys) or set(reviewed) != set(keys):
        raise ValueError("all_positive_and_negative_rollouts_require_review")
    if not review.get("reviewer") or not review.get("reviewed_at_utc"):
        raise ValueError("reward_review_pending")
    for key, reward in keys.items():
        verdict = reviewed[key]
        if (verdict.get("status") != "confirmed" or verdict.get("expected_reward") != reward
                or verdict.get("recorded_reward") != reward or not verdict.get("evidence")):
            raise ValueError(f"reward_review_pending_or_disputed:{key}")
    return {"pilot": str(directory.resolve()), "review_sha256": sha256_file(directory / "reward_review.json"),
            "readiness_sha256": sha256_file(directory / "readiness.json"), "formal_rl_gate_passed": False}
