from __future__ import annotations

from collections import Counter, defaultdict
import json
from statistics import mean
from typing import Any, Iterable


def classify_bad_case(row: dict[str, Any]) -> list[str]:
    if float(row.get("reward") or 0.0) >= 1.0 - 1e-6:
        return []
    categories: set[str] = set()
    verifier = row.get("verifier") or {}
    failures = [str(item) for item in verifier.get("failures") or []]
    executions = list(row.get("tool_executions") or [])
    if any(item.startswith("required_tool_missing_or_failed") for item in failures):
        categories.add("wrong_tool_selection")
    if any("premature_write" in item for item in failures):
        categories.add("premature_write")
    if any("premature_escalation" in item for item in failures):
        categories.add("premature_escalation")
    if any("unsupported_success_claim" in item for item in failures):
        categories.add("unsupported_success_claim")
    if any(
        marker in item
        for item in failures
        for marker in ("customer_confirmation_required", "customer_confirmation_not_observed")
    ):
        categories.add("missing_confirmation")
    if any((item.get("output") or {}).get("error") == "unknown_tool" for item in executions):
        categories.add("unknown_tool")
    argument_markers = ("unexpected keyword", "required positional", "invalid_return_quantity", "exceeds_purchased")
    if any(
        any(marker in str((item.get("output") or {}).get("error") or "") for marker in argument_markers)
        for item in executions
    ):
        categories.add("argument_binding_error")
    signatures = [
        json.dumps({"name": item.get("name"), "arguments": item.get("arguments") or {}}, sort_keys=True, ensure_ascii=False)
        for item in executions
    ]
    if len(signatures) - len(set(signatures)) >= 2:
        categories.add("repeated_tool_loop")
    if any(item.startswith("unrecovered_tool_error") for item in failures):
        categories.add("tool_error_recovery_failure")
    if any(item.startswith(("policy_violation", "unexpected_write")) for item in failures):
        categories.add("policy_violation")
    if any(item.startswith("state_") for item in failures):
        categories.add("state_not_updated")
    if not verifier.get("communication_passed", False):
        categories.add("communication_failure")
    if row.get("stopped_reason") == "max_steps":
        categories.add("repeated_tool_loop")
    if row.get("stopped_reason") == "worker_error":
        categories.add("environment_or_worker_error")
    return sorted(categories or {"unclassified_failure"})


def analyze_bad_cases(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for row in materialized:
        for category in classify_bad_case(row):
            counts[category] += 1
            if len(examples[category]) < 5:
                examples[category].append(str(row.get("task_id")))
    return {
        "rollouts": len(materialized),
        "failed_rollouts": sum(float(row.get("reward") or 0.0) < 1.0 - 1e-6 for row in materialized),
        "counts": dict(sorted(counts.items())),
        "example_task_ids": dict(sorted(examples.items())),
    }


def assess_rl_gate(
    rows: Iterable[dict[str, Any]],
    checkpoint_pass_rates: list[float],
    *,
    critical_tools: set[str],
    plateau_window: int = 3,
    plateau_delta: float = 0.02,
    min_mixed_group_rate: float = 0.20,
    min_strategic_failure_share: float = 0.60,
) -> dict[str, Any]:
    materialized = list(rows)
    calls = [item for row in materialized for item in (row.get("tool_executions") or [])]
    unknown_calls = sum((item.get("output") or {}).get("error") == "unknown_tool" for item in calls)
    tool_format_stability = 1.0 - unknown_calls / max(1, len(calls))
    successful_tools = {str(item.get("name")) for item in calls if item.get("ok")}
    missing_coverage = sorted(critical_tools - successful_tools)

    plateau = False
    if len(checkpoint_pass_rates) >= plateau_window:
        tail = checkpoint_pass_rates[-plateau_window:]
        plateau = max(tail) - min(tail) <= plateau_delta

    by_task: dict[str, list[float]] = defaultdict(list)
    for row in materialized:
        by_task[str(row.get("task_id"))].append(1.0 if float(row.get("reward") or 0.0) >= 1.0 - 1e-6 else 0.0)
    mixed_groups = sum(len(values) >= 2 and 0.0 < mean(values) < 1.0 for values in by_task.values())
    mixed_group_rate = mixed_groups / max(1, len(by_task))

    strategic = {
        "wrong_tool_selection",
        "premature_write",
        "premature_escalation",
        "unsupported_success_claim",
        "tool_error_recovery_failure",
        "policy_violation",
        "state_not_updated",
        "communication_failure",
    }
    failed = [row for row in materialized if float(row.get("reward") or 0.0) < 1.0 - 1e-6]
    strategic_failures = sum(bool(set(classify_bad_case(row)) & strategic) for row in failed)
    strategic_failure_share = strategic_failures / max(1, len(failed))

    checks = {
        "tool_format_stable": tool_format_stability >= 0.98,
        "critical_tool_coverage": not missing_coverage,
        "sft_plateau": plateau,
        "failures_are_strategic": strategic_failure_share >= min_strategic_failure_share,
        "mixed_outcome_groups": mixed_group_rate >= min_mixed_group_rate,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "tool_format_stability": tool_format_stability,
            "missing_critical_tools": missing_coverage,
            "checkpoint_pass_rates": checkpoint_pass_rates,
            "mixed_group_rate": mixed_group_rate,
            "strategic_failure_share": strategic_failure_share,
            "tasks": len(by_task),
            "rollouts": len(materialized),
        },
        "blocking_reasons": reasons,
    }
