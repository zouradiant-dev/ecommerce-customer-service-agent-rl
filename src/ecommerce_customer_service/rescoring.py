from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .data import load_jsonl, write_jsonl
from .models import TaskSpec, VerificationResult
from .rewards import score_verification
from .tasks import load_task_paths
from .verifier import (
    VERIFIER_VERSION,
    SUPPORTED_VERIFIER_VERSIONS,
    communication_failures,
    grounded_fact_failures,
    policy_failures,
    required_tool_failures,
)


RECOMPUTED_FAILURE_PREFIXES = (
    "missing_value:",
    "missing_communication:",
    "missing_intent:",
    "required_tool_missing_or_failed:",
    "policy_violation:",
    "unexpected_write:",
    "premature_write:",
    "premature_escalation:",
    "unrecovered_tool_error:",
    "unsupported_success_claim:",
    "fact_mismatch:",
)


def rescore_rollouts(
    rollouts: Path, manifest: Path, output: Path, *, verifier_version: str = VERIFIER_VERSION,
) -> dict[str, Any]:
    """Recompute trajectory-derived checks without changing raw rollouts.

    State results are retained from the original isolated execution because the
    post-rollout SQLite snapshot is intentionally ephemeral. Communication, policy,
    recovery, and required-tool checks are reproducible from the saved trajectory.
    """

    if verifier_version not in SUPPORTED_VERIFIER_VERSIONS:
        raise ValueError(f"unsupported_verifier_version:{verifier_version}")
    tasks = {
        task.task_id: task
        for path in load_task_paths(manifest)
        for task in (TaskSpec.from_file(path),)
    }
    source_rows = load_jsonl(rollouts)
    rescored_rows: list[dict[str, Any]] = []
    changed = 0
    passed_before = 0
    passed_after = 0

    for source in source_rows:
        row = deepcopy(source)
        task_id = str(row.get("task_id") or "")
        if task_id not in tasks:
            raise ValueError(f"Task {task_id!r} is not present in manifest {manifest}")
        task = tasks[task_id]
        old_verifier = dict(row.get("verifier") or {})
        old_passed = bool(old_verifier.get("passed"))
        passed_before += int(old_passed)

        final_response = str(row.get("final_response") or "")
        communication_failure_rows = communication_failures(task, final_response, verifier_version=verifier_version)
        communication_failure_rows += grounded_fact_failures(
            task, final_response, row.get("tool_executions") or [], verifier_version=verifier_version,
        )
        required_failure_rows = required_tool_failures(task, row.get("tool_executions") or [])
        policy_failure_rows = policy_failures(
            task,
            str(row.get("final_response") or ""),
            row.get("tool_executions") or [],
            verifier_version=verifier_version,
        )
        retained_failures = [
            str(item)
            for item in old_verifier.get("failures") or []
            if not str(item).startswith(RECOMPUTED_FAILURE_PREFIXES)
        ]
        failures = (
            tuple(retained_failures)
            + communication_failure_rows
            + required_failure_rows
            + policy_failure_rows
        )
        verification = VerificationResult(
            passed=(
                bool(old_verifier.get("state_passed"))
                and not communication_failure_rows
                and not policy_failure_rows
                and not required_failure_rows
                and not retained_failures
            ),
            state_passed=bool(old_verifier.get("state_passed")),
            communication_passed=not communication_failure_rows,
            policy_passed=not policy_failure_rows,
            required_tools_passed=not required_failure_rows,
            failures=failures,
        )
        breakdown = score_verification(verification)
        row["schema_version"] = "ecommerce_cs_rollout_v2"
        row["verifier"] = verification.to_dict()
        row["verifier_version"] = verifier_version
        row["reward"] = breakdown.strict_binary
        row["reward_components"] = breakdown.to_dict()
        row["rescore_metadata"] = {
            "method": "retain_state_recompute_communication_policy_required_tools",
            "source_verifier_version": source.get("verifier_version") or "unversioned",
            "target_verifier_version": verifier_version,
            "source_reward": source.get("reward"),
        }
        passed_after += int(verification.passed)
        changed += int(old_passed != verification.passed or list(old_verifier.get("failures") or []) != list(failures))
        rescored_rows.append(row)

    write_jsonl(output, rescored_rows)
    return {
        "rollouts": len(rescored_rows),
        "passed_before": passed_before,
        "passed_after": passed_after,
        "changed_rows": changed,
        "verifier_version": verifier_version,
        "output": str(output.resolve()),
        "method": "retain_state_recompute_communication_policy_required_tools",
    }
