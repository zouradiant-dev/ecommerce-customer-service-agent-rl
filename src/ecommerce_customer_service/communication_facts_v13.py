"""Explicit v13 additions; frozen v12 facts and intent rules remain unchanged.

Only an evidenced absent cancellation refund ID is exempt from the old ID
token check. Labelled order totals and actual captured payment amounts use
separate tool fields. This is a scoped deterministic check, not a general judge.
"""
from __future__ import annotations

from decimal import Decimal
import re
from typing import Any, Iterable

from .communication_facts import _text, grounded_communication_failures
from .models import TaskSpec


def _cents(value: Any) -> Decimal | None:
    # Database currency fields are nonnegative integer cents, never booleans,
    # floating point approximations, strings, NaN or unknown/missing values.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return Decimal(value) / 100


def _owned_target(task: TaskSpec, call: dict[str, Any]) -> bool:
    arguments = call.get("arguments") or {}
    target = task.metadata.get("order_id")
    return bool(target and call.get("ok") is True
                and arguments.get("customer_id") == task.user_id
                and arguments.get("order_id") == target)


def _captured_amount(order_id: str, payments: Any) -> Decimal | None:
    if not isinstance(payments, list):
        return None
    total = Decimal(0)
    for payment in payments:
        if not isinstance(payment, dict) or payment.get("order_id") != order_id:
            return None  # mixed ownership or incomplete evidence is not zero
        if payment.get("status") not in {"captured", "refunded", "authorized", "failed"}:
            return None
        if payment["status"] in {"captured", "refunded"}:
            amount = _cents(payment.get("amount_cents"))
            if amount is None:
                return None
            # Refunding a payment does not mean it was never captured.
            total += amount
    return total


def _facts(task: TaskSpec, calls: list[dict[str, Any]]) -> tuple[bool, Decimal | None, Decimal | None]:
    target = task.metadata.get("order_id")
    order_total = captured = None
    supported_null_cancel = False
    complete_unpaid_read = False
    has_refund = False
    for call in calls:
        if not _owned_target(task, call):
            continue
        output = call.get("output") or {}
        if call.get("name") == "get_order":
            order = output.get("order") or {}
            if order.get("id") != target or order.get("customer_id") != task.user_id:
                continue
            order_total = _cents(order.get("total_paid_cents"))
            captured = _captured_amount(target, output.get("payments"))
            refunds = output.get("refunds")
            if isinstance(refunds, list):
                has_refund |= any(isinstance(item, dict) and item.get("order_id") == target
                                  and item.get("id") for item in refunds)
            complete_unpaid_read = captured == Decimal(0) and refunds == []
        elif call.get("name") in {"cancel_order", "create_refund"}:
            has_refund |= bool(output.get("refund_id"))
            if call["name"] == "cancel_order":
                # Missing field, false result or failed execution never proves
                # absence. Keep the latest acknowledged cancellation result.
                supported_null_cancel = (output.get("cancelled") is True
                                         and "refund_id" in output and output["refund_id"] is None
                                         and complete_unpaid_read)
    return supported_null_cancel and complete_unpaid_read and not has_refund, order_total, captured


MONEY = r"(-?\d+(?:\.\d+)?)\s*(元|分|cny|rmb|cents?|yuan)(?![a-z])"


def _amount_failures(text: str, label: str, expected: Decimal | None, field: str) -> list[str]:
    if expected is None:
        return []
    failures = []
    pattern = rf"{label}\s*(?::|=|为|是)?\s*[¥￥]?\s*{MONEY}"
    for match in re.finditer(pattern, text):
        # Do not mistake an explicit conditional example for an actual result.
        # Unknown/unlabelled prose remains outside this deliberately small rule.
        clause_start = max((text.rfind(separator, 0, match.start())
                            for separator in "。！？!?；;\n"), default=-1) + 1
        if re.search(r"(?:如果|假如|假设|若)", text[clause_start:match.start()]):
            continue
        claims = [(match.group(1), match.group(2))]
        alternate = re.match(rf"\s*\((?:即|合计|约|相当于)?\s*{MONEY}\s*\)", text[match.end():])
        if alternate:
            claims.append((alternate.group(1), alternate.group(2)))
        for value, unit in claims:
            amount = Decimal(value) / (100 if unit in {"分", "cent", "cents"} else 1)
            if amount != expected:
                failures.append(f"fact_mismatch:{field}:{value}{unit}")
    return failures


def grounded_communication_failures_v13(
    task: TaskSpec, final_response: str, tool_executions: Iterable[dict[str, Any]],
) -> tuple[str, ...]:
    calls = list(tool_executions)
    failures = list(grounded_communication_failures(task, final_response, calls))
    evidenced_null, order_total, captured = _facts(task, calls)
    if evidenced_null:
        failures = [value for value in failures if value != "fact_mismatch:refund_id:null"]
    text = _text(final_response)
    target = task.metadata.get("order_id")
    # Both labels draw on actual owned reads, not task-specific expected answers.
    # No cancellation status update changes the monetary order total.
    order_label = r"订单\s*(?:总金额|合计金额|金额|总价)"
    if target:
        order_label = rf"(?:{order_label}|订单\s*{re.escape(str(target).casefold())}\s*(?:的\s*)?(?:总金额|合计金额|金额|总价))"
    failures += _amount_failures(text, order_label, order_total, "order_total")
    failures += _amount_failures(text, r"(?:实际\s*扣款(?:\s*金额)?|实扣(?:\s*金额)?)", captured, "captured_amount")
    return tuple(dict.fromkeys(failures))
