"""Structured terminal protocol for the opt-in v18 training route.

The model emits one JSON object.  Only ``message`` is passed to the natural
language verifier, so protocol labels cannot satisfy communication checks.
Every decision is checked against trusted tool evidence before it receives
credit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Iterable

from .models import TaskSpec


TERMINAL_PROTOCOL_VERSION = "ecommerce_cs_terminal_v18"
_DECISION_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_REQUIRED_FIELDS = {"decision", "action_taken", "action_id", "message"}


@dataclass(frozen=True)
class TerminalDecision:
    decision: str
    action_taken: bool
    action_id: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TerminalValidation:
    parsed: TerminalDecision | None
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed.to_dict() if self.parsed else None,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def parse_terminal_response(final_response: str) -> TerminalValidation:
    """Parse an exact JSON object; prose or fenced JSON is intentionally invalid."""

    try:
        value = json.loads(final_response)
    except (TypeError, json.JSONDecodeError):
        return TerminalValidation(None, False, ("terminal_invalid_json",))
    if not isinstance(value, dict):
        return TerminalValidation(None, False, ("terminal_must_be_object",))
    if set(value) != _REQUIRED_FIELDS:
        missing = sorted(_REQUIRED_FIELDS - set(value))
        extra = sorted(set(value) - _REQUIRED_FIELDS)
        failures = []
        if missing:
            failures.append("terminal_missing_fields:" + ",".join(missing))
        if extra:
            failures.append("terminal_unknown_fields:" + ",".join(extra))
        return TerminalValidation(None, False, tuple(failures))
    decision = value.get("decision")
    action_taken = value.get("action_taken")
    action_id = value.get("action_id")
    message = value.get("message")
    failures = []
    if not isinstance(decision, str) or not _DECISION_RE.fullmatch(decision):
        failures.append("terminal_invalid_decision")
    if type(action_taken) is not bool:
        failures.append("terminal_action_taken_must_be_boolean")
    if action_id is not None and (not isinstance(action_id, str) or not action_id.strip()):
        failures.append("terminal_invalid_action_id")
    if not isinstance(message, str) or not message.strip():
        failures.append("terminal_message_required")
    if failures:
        return TerminalValidation(None, False, tuple(failures))
    parsed = TerminalDecision(
        decision=decision,
        action_taken=action_taken,
        action_id=action_id.strip() if isinstance(action_id, str) else None,
        message=message.strip(),
    )
    return TerminalValidation(parsed, True, ())


def validate_terminal_response(
    task: TaskSpec,
    final_response: str,
    tool_executions: Iterable[dict[str, Any]],
) -> TerminalValidation:
    parsed_result = parse_terminal_response(final_response)
    if parsed_result.parsed is None:
        return parsed_result
    terminal = parsed_result.parsed
    calls = list(tool_executions)
    expectation = task.metadata.get("terminal_v18")
    if not isinstance(expectation, dict):
        return TerminalValidation(terminal, False, ("terminal_expectation_missing",))

    failures: list[str] = []
    expected_decisions = expectation.get("decision")
    if isinstance(expected_decisions, str):
        expected_decisions = [expected_decisions]
    if not isinstance(expected_decisions, list) or not expected_decisions:
        failures.append("terminal_expected_decision_missing")
    elif terminal.decision not in {str(value) for value in expected_decisions}:
        failures.append(f"terminal_decision_mismatch:{terminal.decision}")

    expected_action = expectation.get("action_taken")
    if type(expected_action) is not bool:
        failures.append("terminal_expected_action_missing")
    elif terminal.action_taken is not expected_action:
        failures.append("terminal_action_taken_mismatch")

    successful_writes = [call for call in calls if call.get("ok") and call.get("is_write")]
    action_tool = expectation.get("action_tool")
    matching_actions = [call for call in successful_writes if call.get("name") == action_tool]
    if terminal.action_taken:
        if not action_tool or not matching_actions:
            failures.append("terminal_action_without_successful_tool")
        action_id_field = expectation.get("action_id_field")
        if not isinstance(action_id_field, str) or not action_id_field:
            failures.append("terminal_action_id_field_missing")
        else:
            trusted_ids = {
                str((call.get("output") or {}).get(action_id_field))
                for call in matching_actions
                if (call.get("output") or {}).get(action_id_field) is not None
            }
            if terminal.action_id not in trusted_ids:
                failures.append("terminal_action_id_not_grounded")
    else:
        if terminal.action_id is not None:
            failures.append("terminal_no_action_requires_null_id")
        if successful_writes:
            failures.append("terminal_denies_successful_write")

    failures.extend(_decision_evidence_failures(task, terminal, calls))
    return TerminalValidation(terminal, not failures, tuple(dict.fromkeys(failures)))


def _decision_evidence_failures(
    task: TaskSpec,
    terminal: TerminalDecision,
    calls: list[dict[str, Any]],
) -> list[str]:
    """Require public decision labels to agree with auditable evidence."""

    successful = [call for call in calls if call.get("ok")]
    names = {str(call.get("name")) for call in successful}
    writes = {str(call.get("name")) for call in successful if call.get("is_write")}
    failures: list[str] = []

    required_tool = {
        "IDENTITY_ESCALATED": "escalate_to_human",
        "ORDER_CANCELLED": "cancel_order",
        "REFUND_CREATED": "create_refund",
        "ADDRESS_UPDATED": "update_shipping_address",
    }.get(terminal.decision)
    if required_tool and required_tool not in names:
        failures.append(f"terminal_decision_missing_evidence:{required_tool}")

    if terminal.decision == "ORDER_NOT_FOUND":
        requested = str(task.metadata.get("requested_order_id") or "")
        lists = [call for call in successful if call.get("name") == "list_orders"]
        listed_ids = {
            str(order.get("id"))
            for call in lists
            for order in ((call.get("output") or {}).get("orders") or [])
            if isinstance(order, dict) and order.get("id")
        }
        if not requested or not lists or requested in listed_ids:
            failures.append("terminal_order_not_found_unproven")
        if writes:
            failures.append("terminal_order_not_found_has_write")

    if terminal.decision == "ITEM_ORDER_MISMATCH":
        target_order = str(task.metadata.get("order_id") or "")
        mismatched_item = str(task.metadata.get("mismatched_item_id") or "")
        inspected = [
            call for call in successful
            if call.get("name") == "get_order"
            and str(((call.get("output") or {}).get("order") or {}).get("id")) == target_order
        ]
        target_items = {
            str(item.get("id"))
            for call in inspected
            for item in ((call.get("output") or {}).get("items") or [])
            if isinstance(item, dict) and item.get("id")
        }
        mismatch_rejection = any(
            call.get("name") == "search_refund_policy"
            and not call.get("ok")
            and call.get("policy_violation") == "item_not_in_inspected_order"
            for call in calls
        )
        if not target_order or not mismatched_item or not inspected or mismatched_item in target_items:
            failures.append("terminal_item_order_mismatch_unproven")
        if not mismatch_rejection:
            failures.append("terminal_item_order_mismatch_not_rejected")
        if "create_refund" in writes:
            failures.append("terminal_item_order_mismatch_has_refund")

    if terminal.decision == "POLICY_BLOCKED":
        policy_calls = [
            call for call in successful
            if call.get("name") == "search_refund_policy"
            and (call.get("output") or {}).get("allowed") is False
        ]
        if not policy_calls:
            failures.append("terminal_policy_block_unproven")
        if "create_refund" in writes:
            failures.append("terminal_policy_block_has_refund")

    if terminal.decision == "NO_ACTION" and writes:
        failures.append("terminal_no_action_has_write")
    return failures
