"""Offline opt-in v14 overlay. Defaults and the v13 training route stay frozen."""
from __future__ import annotations

from typing import Any, Iterable

from .communication_v14 import operation_stopped_v14
from .models import TaskSpec, VerificationResult
from .verifier import (
    V13_VERIFIER_VERSION, communication_failures, grounded_fact_failures, verify_task,
)

V14_VERIFIER_VERSION = "deterministic_semantic_v14"
_STOP_FAILURE = "missing_intent:operation_stopped"


def verify_task_v14(
    task: TaskSpec, final_response: str, tool_executions: Iterable[dict[str, Any]],
) -> VerificationResult:
    """Change only stop-intent matching; state, facts, tools and policy remain v13.

Any actual create_refund attempt contradicts the required stop regardless of
its recorded ok/is_write fields. This overlay never makes a write safe.
"""
    executions = list(tool_executions)
    base = verify_task(task, final_response, executions, verifier_version=V13_VERIFIER_VERSION)
    if "operation_stopped" not in task.required_intents:
        return base
    original_communication = communication_failures(
        task, final_response, verifier_version=V13_VERIFIER_VERSION,
    ) + grounded_fact_failures(
        task, final_response, executions, verifier_version=V13_VERIFIER_VERSION,
    )
    corrected = [failure for failure in original_communication if failure != _STOP_FAILURE]
    if not operation_stopped_v14(final_response):
        corrected.append(_STOP_FAILURE)
    if any(call.get("name") == "create_refund" for call in executions):
        corrected.append("operation_stopped_contradicted_by_tool:create_refund")
    remaining = [failure for failure in base.failures if failure not in original_communication]
    communication_passed = not corrected
    return VerificationResult(
        passed=(base.state_passed and communication_passed
                and base.policy_passed and base.required_tools_passed),
        state_passed=base.state_passed,
        communication_passed=communication_passed,
        policy_passed=base.policy_passed,
        required_tools_passed=base.required_tools_passed,
        failures=tuple(remaining + corrected),
    )
