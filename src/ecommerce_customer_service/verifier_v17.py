"""Offline opt-in strict v17; frozen v14 hard/fact checks remain mandatory."""
from .communication_v17 import FULL_REFUND, SPECS, recognize_required
from .communication_facts_v13 import _captured_amount
from .models import VerificationResult
from .verifier_v14 import verify_task_v14
from .verifier_v15 import PROHIBITED_ATTEMPTS
import re

V17_VERIFIER_VERSION = "deterministic_semantic_v17_strict"


def _owned_reads(task, calls):
    reads, listed = {}, None
    target = task.metadata.get("order_id")
    for call in calls:
        if call.get("ok") is not True or (call.get("arguments") or {}).get("customer_id") != task.user_id:
            continue
        output = call.get("output") or {}
        if call.get("name") == "get_order":
            order = output.get("order") or {}
            order_id = order.get("id")
            if order_id and order.get("customer_id") == task.user_id and (not target or order_id == target) and (
                    call["arguments"].get("order_id") == order_id):
                reads[order_id] = output
        elif call.get("name") == "list_orders":
            orders = output.get("orders")
            if isinstance(orders, list) and orders and all(isinstance(o, dict) and o.get("id")
                    and o.get("customer_id") == task.user_id for o in orders):
                listed = orders
    return next(iter(reads.values())) if len(reads) == 1 else None, listed


def _quantity_one(read):
    if not read:
        return False
    items = read.get("items")
    order_id = read["order"]["id"]
    return bool(isinstance(items, list) and items and all(isinstance(i, dict)
        and i.get("order_id") == order_id and type(i.get("quantity")) is int and i["quantity"] > 0 for i in items)
        and sum(i["quantity"] for i in items) == 1)


def _full_refund(read):
    if not read:
        return False
    order_id = read["order"]["id"]
    captured = _captured_amount(order_id, read.get("payments"))
    refunds = read.get("refunds")
    if captured is None or captured <= 0 or not isinstance(refunds, list) or not refunds:
        return False
    if not all(isinstance(r, dict) and r.get("order_id") == order_id and r.get("status") == "completed"
               and type(r.get("amount_cents")) is int and r["amount_cents"] >= 0 and r.get("id") for r in refunds):
        return False
    if len({r["id"] for r in refunds}) != len(refunds):
        return False
    return sum(r["amount_cents"] for r in refunds) == captured * 100


def _completed_refund(read):
    if not read:
        return False
    order_id = read["order"]["id"]
    refunds = read.get("refunds")
    return bool(
        isinstance(refunds, list)
        and any(
            isinstance(refund, dict)
            and refund.get("order_id") == order_id
            and refund.get("status") == "completed"
            and refund.get("id")
            for refund in refunds
        )
    )


def verify_task_v17_with_evidence(task, final_response, tool_executions):
    calls = list(tool_executions)
    base = verify_task_v14(task, final_response, calls)
    decisions = recognize_required(task.required_intents, final_response)
    read, listed = _owned_reads(task, calls)
    for decision in decisions:
        newly_resolved = any(e["reason"].startswith("v17_") for e in decision["support"])
        missing_grounding = False
        if newly_resolved:
            if decision["intent"] in {"purchased_quantity_one", "refund_quantity_two_rejected"}:
                missing_grounding = not _quantity_one(read)
            elif decision["intent"] == "shipped_address_change_blocked":
                missing_grounding = not read or read["order"].get("status") != "shipped"
            elif decision["intent"] == "operation_stopped":
                missing_grounding = not read
            elif decision["intent"] == "order_id_confirmation_needed":
                requests = [e for e in decision["support"] if e["reason"] == "v17_resolved_order_question"]
                # A declared unique order must really be unique. Which-order
                # questions need a successful owned list or owned order read.
                missing_grounding = not (listed or read) or any(
                    "哪" not in e["matched"] and (not listed or len(listed) != 1) for e in requests)
                missing_grounding |= any("列出" in e.get("antecedent_text", "") and not listed for e in requests)
            elif decision["intent"] == "refund_already_completed":
                missing_grounding = not _completed_refund(read)
            allowed_ids = {read["order"]["id"]} if read else set()
            if decision["intent"] == "order_id_confirmation_needed" and listed:
                allowed_ids |= {o["id"] for o in listed}
            resolved_text = " ".join(e["text"] + " " + e.get("antecedent_text", "")
                                     for e in decision["support"] if e["reason"].startswith("v17_"))
            mentioned_ids = set(re.findall(r"ord_[a-z0-9_-]+", resolved_text.casefold()))
            missing_grounding |= not mentioned_ids.issubset({i.casefold() for i in allowed_ids})
        if decision["intent"] == "refund_already_completed" and any(
                re.search(FULL_REFUND, e["matched"] or "") for e in decision["support"]):
            missing_grounding |= not _full_refund(read)
        if missing_grounding:
            decision["contradictions"].append({"reason": "v17_missing_or_conflicting_owned_tool_grounding"})
        for call in calls:
            if call.get("name") in PROHIBITED_ATTEMPTS.get(decision["intent"], set()):
                decision["contradictions"].append({"tool": call["name"], "reason": "prohibited_action_attempt"})
        if decision["contradictions"]:
            decision["passed"], decision["status"] = False, "contradicted"
    replaced = {"missing_intent:" + intent for intent in task.required_intents if intent in SPECS}
    remaining = [failure for failure in base.failures if failure not in replaced]
    new = ["missing_intent:" + d["intent"] for d in decisions if not d["passed"]]
    failures = tuple(remaining + new)
    communication_passed = not new and not any(f.startswith(("missing_intent:", "missing_value:",
        "missing_communication:", "fact_mismatch:", "operation_stopped_contradicted_by_tool:")) for f in remaining)
    result = VerificationResult(passed=not failures and base.state_passed and communication_passed
        and base.policy_passed and base.required_tools_passed, state_passed=base.state_passed,
        communication_passed=communication_passed, policy_passed=base.policy_passed,
        required_tools_passed=base.required_tools_passed, failures=failures)
    return result, decisions


def verify_task_v17(task, final_response, tool_executions):
    return verify_task_v17_with_evidence(task, final_response, tool_executions)[0]
