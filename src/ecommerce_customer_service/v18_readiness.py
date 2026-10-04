"""CPU-only calibration gate for v18 rewards.

This gate consumes reviewed rollout rows and derives ``reward_calibrated``.
Training code must consume the generated receipt instead of a hand-authored
boolean.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re
from statistics import mean
from typing import Any, Iterable


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CRITICAL_WRITES = {"cancel_order", "update_shipping_address", "create_refund"}


def assess_v18_reward_calibration(
    rows: Iterable[dict[str, Any]],
    *,
    source_sha256: dict[str, str],
    min_mixed_group_rate: float = 0.20,
    min_communication_precision: float = 0.95,
    min_communication_recall: float = 0.95,
) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("v18 calibration requires reviewed rows")
    if not 0 <= min_mixed_group_rate <= 1:
        raise ValueError("invalid mixed-group threshold")
    if not 0 <= min_communication_precision <= 1 or not 0 <= min_communication_recall <= 1:
        raise ValueError("invalid communication thresholds")

    malformed: list[str] = []
    unsafe_false_negatives: list[str] = []
    rewards: list[float] = []
    by_task: dict[str, list[float]] = defaultdict(list)
    scenario_outcomes: dict[str, set[str]] = defaultdict(set)
    successful_writes: set[str] = set()
    reviewed_safe: set[bool] = set()
    reviewed_reward_signs: set[str] = set()
    tp = fp = fn = 0

    for index, row in enumerate(materialized):
        row_id = str(row.get("rollout_id") or f"row-{index}")
        review = row.get("human_review")
        verification = row.get("verification")
        components = row.get("reward_components")
        if not isinstance(review, dict) or not isinstance(verification, dict) or not isinstance(components, dict):
            malformed.append(row_id + ":missing_review_or_verification")
            continue
        reward = components.get("train_shaped_v18")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(reward):
            malformed.append(row_id + ":invalid_reward")
            continue
        reward = float(reward)
        if reward < -1 or reward > 1:
            malformed.append(row_id + ":reward_out_of_range")
            continue
        rewards.append(reward)
        task_id = str(row.get("task_id") or "")
        scenario = str(row.get("scenario") or "")
        if not task_id or not scenario:
            malformed.append(row_id + ":missing_task_or_scenario")
            continue
        by_task[task_id].append(reward)
        scenario_outcomes[scenario].add("high" if reward >= 0.75 else "low")
        reviewed_reward_signs.add("positive" if reward > 0 else "non_positive")

        human_safe = review.get("safe")
        predicted_safe = verification.get("safety_gate_passed")
        human_communication = review.get("communication_passed")
        predicted_communication = verification.get("communication_passed")
        if type(human_safe) is not bool or type(predicted_safe) is not bool:
            malformed.append(row_id + ":invalid_safety_review")
        else:
            reviewed_safe.add(human_safe)
            if not human_safe and predicted_safe:
                unsafe_false_negatives.append(row_id)
        if type(human_communication) is not bool or type(predicted_communication) is not bool:
            malformed.append(row_id + ":invalid_communication_review")
        elif predicted_communication and human_communication:
            tp += 1
        elif predicted_communication and not human_communication:
            fp += 1
        elif not predicted_communication and human_communication:
            fn += 1

        if row.get("error") or row.get("stopped_reason") in {"max_steps", "max_tool_iterations", "error"}:
            malformed.append(row_id + ":incomplete_trajectory")
        for call in row.get("tool_executions") or []:
            if call.get("ok") and call.get("name") in CRITICAL_WRITES:
                successful_writes.add(str(call["name"]))

    mixed_groups = sum(len(values) >= 2 and len(set(values)) >= 2 for values in by_task.values())
    mixed_group_rate = mixed_groups / max(1, len(by_task))
    communication_precision = tp / max(1, tp + fp)
    communication_recall = tp / max(1, tp + fn)
    scenarios_without_contrast = sorted(
        scenario for scenario, outcomes in scenario_outcomes.items() if outcomes != {"high", "low"}
    )
    missing_writes = sorted(CRITICAL_WRITES - successful_writes)
    bad_hashes = sorted(
        name for name, digest in source_sha256.items()
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)
    )
    if not source_sha256:
        bad_hashes.append("<missing>")

    checks = {
        "rows_well_formed": not malformed,
        "unsafe_false_negatives_zero": not unsafe_false_negatives,
        "positive_and_negative_safety_reviewed": reviewed_safe == {False, True},
        "positive_and_non_positive_rewards_reviewed": reviewed_reward_signs == {"positive", "non_positive"},
        "communication_precision": communication_precision >= min_communication_precision,
        "communication_recall": communication_recall >= min_communication_recall,
        "mixed_group_rate": mixed_group_rate >= min_mixed_group_rate,
        "scenario_reward_contrast": not scenarios_without_contrast,
        "critical_write_coverage": not missing_writes,
        "source_hashes_bound": not bad_hashes,
    }
    passed = all(checks.values())
    return {
        "schema_version": "ecommerce_cs_v18_reward_calibration_receipt_v1",
        "reward_calibrated": passed,
        "checks": checks,
        "metrics": {
            "rows": len(materialized),
            "tasks": len(by_task),
            "scenarios": len(scenario_outcomes),
            "mean_train_shaped_v18": mean(rewards) if rewards else None,
            "mixed_group_rate": mixed_group_rate,
            "communication_precision": communication_precision,
            "communication_recall": communication_recall,
            "unsafe_false_negatives": len(unsafe_false_negatives),
        },
        "blocking": {
            "malformed_rows": malformed,
            "unsafe_false_negative_ids": unsafe_false_negatives,
            "scenarios_without_high_and_low": scenarios_without_contrast,
            "missing_successful_writes": missing_writes,
            "invalid_source_hashes": bad_hashes,
        },
        "source_sha256": dict(sorted(source_sha256.items())),
    }
