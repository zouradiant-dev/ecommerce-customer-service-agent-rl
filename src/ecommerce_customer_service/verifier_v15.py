"""Offline v15 opt-in overlay: replace only eleven missing-intent decisions.

The v14 state, policy, required-tools and grounded fact checks are retained.
Defaults, historical reward files and training routes are never modified.
"""
from __future__ import annotations

from typing import Any, Iterable

from .communication_v15 import SPECS, recognize_required
from .models import TaskSpec, VerificationResult
from .verifier_v14 import verify_task_v14

V15_VERIFIER_VERSION = "deterministic_semantic_v15"
PROHIBITED_ATTEMPTS = {
    "operation_stopped": {"create_refund"},
    "order_id_confirmation_needed": {"cancel_order", "create_refund"},
    "shipped_address_change_blocked": {"update_shipping_address"},
    "return_before_delivery_blocked": {"create_refund"},
    "no_duplicate_action": {"cancel_order", "create_refund"},
}


def verify_task_v15_with_evidence(task: TaskSpec, final_response: str, tool_executions: Iterable[dict[str, Any]]):
    calls = list(tool_executions)
    base = verify_task_v14(task, final_response, calls)
    decisions = recognize_required(task.required_intents, final_response)
    replaced = {"missing_intent:" + intent for intent in task.required_intents if intent in SPECS}
    remaining = [failure for failure in base.failures if failure not in replaced]
    new = []
    for decision in decisions:
        for call in calls:
            if call.get("name") in PROHIBITED_ATTEMPTS.get(decision["intent"], set()):
                decision["passed"] = False
                decision["status"] = "contradicted"
                decision["contradictions"].append({"tool": call["name"], "reason": "prohibited_action_attempt"})
        if not decision["passed"]:
            new.append("missing_intent:" + decision["intent"])
    failures = tuple(remaining + new)
    communication_passed = not new and not any(
        failure.startswith(("missing_intent:", "missing_value:", "missing_communication:",
                            "fact_mismatch:", "operation_stopped_contradicted_by_tool:"))
        for failure in remaining)
    result = VerificationResult(
        passed=(not failures and base.state_passed and communication_passed and base.policy_passed and base.required_tools_passed),
        state_passed=base.state_passed, communication_passed=communication_passed,
        policy_passed=base.policy_passed, required_tools_passed=base.required_tools_passed, failures=failures)
    return result, decisions


def verify_task_v15(task: TaskSpec, final_response: str, tool_executions: Iterable[dict[str, Any]]):
    return verify_task_v15_with_evidence(task, final_response, tool_executions)[0]
