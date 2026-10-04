"""Conservative checks of explicit business facts against recorded tool evidence.

This is not a general semantic judge. It checks typed IDs, reported refund
amounts/destinations and labelled result statuses; unrecognised prose is not
silently treated as independently verified. Task-specific answers are not used.
"""
from __future__ import annotations

from decimal import Decimal
import re
from typing import Any, Iterable
import unicodedata

from .models import TaskSpec


ID_LABELS = {
    "refund_id": r"(?:退款(?:流水)?\s*(?:id|编号|号)|refund_id)",
    "return_id": r"(?:退货(?:项|记录|申请|单)?\s*(?:id|编号|号)|return_id)",
    "ticket_id": r"(?:工单\s*(?:id|编号|号)|ticket_id)",
}
ID_TOKEN = r"[a-z][a-z0-9_-]{2,}"
PAYMENT_NAMES = {"alipay": ("支付宝", "alipay"), "wechat": ("微信", "wechat")}
STATUS_NAMES = {
    "completed": ("已完成", "完成", "completed", "已退款", "refunded", "成功"),
    "cancelled": ("已取消", "取消", "cancelled", "canceled"),
    "pending": ("待处理", "pending", "未发货"),
    "paid": ("已付款", "paid", "未发货"),
    "shipped": ("已发货", "shipped"),
    "delivered": ("已签收", "已送达", "delivered"),
    "open": ("已创建", "已升级", "open"),
}
KNOWN_STATUSES = {name for values in STATUS_NAMES.values() for name in values} | {
    "未退款", "未完成", "失败", "failed", "processing", "处理中", "未取消", "未创建",
}


def _text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().translate(
        str.maketrans("", "", "*`\"'"))


def grounded_communication_failures(
    task: TaskSpec, final_response: str, tool_executions: Iterable[dict[str, Any]],
) -> tuple[str, ...]:
    """Check claims using successful, owned, target-order tool results only."""
    ids: dict[str, set[str]] = {field: set() for field in ID_LABELS}
    refund_amounts: set[Decimal] = set()
    payment_methods: set[str] = set()
    captured_amount: Decimal | None = None
    statuses: dict[str, set[str]] = {"退款": set(), "订单": set(), "工单": set()}
    order_id = task.metadata.get("order_id")
    for call in tool_executions:
        if not call.get("ok"):
            continue
        arguments, output = call.get("arguments") or {}, call.get("output") or {}
        name = call.get("name")
        if name == "escalate_to_human":
            ticket = str(output.get("ticket_id") or "")
            # The business tool routes tickets to the verified/task customer.
            if output.get("escalated") and ticket.startswith(f"ticket_{task.user_id}_"):
                ids["ticket_id"].add(ticket.casefold())
                statuses["工单"] = {"open"}
            continue
        if arguments.get("customer_id") != task.user_id:
            continue
        if order_id and arguments.get("order_id") != order_id:
            continue
        if name == "get_order":
            order = output.get("order") or {}
            if order.get("customer_id") != task.user_id or (order_id and order.get("id") != order_id):
                continue
            statuses["订单"] = {str(order.get("status"))}
            for payment in output.get("payments") or []:
                if payment.get("order_id") == order.get("id"):
                    payment_methods.add(str(payment.get("method")))
            captured = sorted((payment for payment in output.get("payments") or []
                               if payment.get("order_id") == order.get("id")
                               and payment.get("status") == "captured"), key=lambda payment: payment["id"])
            captured_amount = Decimal(captured[0]["amount_cents"]) / 100 if captured else None
            for record in output.get("returns") or []:
                if record.get("order_id") == order.get("id"):
                    ids["return_id"].add(str(record["id"]).casefold())
            for record in output.get("refunds") or []:
                if record.get("order_id") == order.get("id"):
                    ids["refund_id"].add(str(record["id"]).casefold())
                    refund_amounts.add(Decimal(record["amount_cents"]) / 100)
                    statuses["退款"].add(str(record.get("status")))
        elif name in {"create_refund", "cancel_order"}:
            for field in ("refund_id", "return_id"):
                if output.get(field):
                    ids[field].add(str(output[field]).casefold())
            if output.get("refund_id"):
                statuses["退款"] = {"completed"}
            if name == "create_refund" and output.get("refunded"):
                refund_amounts.add(Decimal(output["amount_cents"]) / 100)
            if name == "cancel_order" and output.get("cancelled"):
                statuses["订单"] = {"cancelled"}
                if output.get("refund_id") and captured_amount is not None:
                    refund_amounts.add(captured_amount)

    text = _text(final_response)
    failures: list[str] = []
    for field, label in ID_LABELS.items():
        pattern = rf"{label}\s*(?::|=|为|是)\s*({ID_TOKEN})"
        for match in re.finditer(pattern, text):
            value = match.group(1)
            if value not in ids[field]:
                failures.append(f"fact_mismatch:{field}:{value}")

    # Do not confuse a pre-operation quote with a completed refund. Where a
    # completed amount is recorded, check both yuan and parenthesised cents.
    if refund_amounts:
        money = r"(\d+(?:\.\d+)?)\s*(元|分|cny|rmb|cents?|yuan)"
        pattern = rf"(?:退款(?:金额)?|退还金额)\s*(?::|=|为|是)?\s*[¥￥]?\s*{money}"
        for match in re.finditer(pattern, text):
            claims = [(match.group(1), match.group(2))]
            parenthesis = re.match(rf"\s*\((?:即|合计|约|相当于)?\s*{money}\s*\)", text[match.end():])
            if parenthesis:
                claims.append((parenthesis.group(1), parenthesis.group(2)))
            for value, unit in claims:
                amount = Decimal(value) / (100 if unit in {"分", "cent", "cents"} else 1)
                if amount not in refund_amounts:
                    failures.append(f"fact_mismatch:refund_amount:{value}{unit}")

        # A matching "original payment" phrase must not conceal a different
        # explicitly named destination. Only supported payment methods are
        # checked here; no guess is made for an unknown channel.
        for method, aliases in PAYMENT_NAMES.items():
            pattern = rf"(?:退款(?:方式|渠道)|原路(?:退回|返回)|原支付(?:路径|渠道)).{{0,16}}(?:{'|'.join(aliases)})"
            if re.search(pattern, text) and payment_methods and method not in payment_methods:
                failures.append(f"fact_mismatch:refund_destination:{method}")

    for entity, expected in statuses.items():
        if not expected:
            continue
        allowed = {alias for state in expected for alias in STATUS_NAMES.get(state, (state,))}
        for match in re.finditer(rf"{entity}状态\s*(?::|=|为|是)\s*([^。\n;，,()]+)", text):
            claim = match.group(1).strip()
            if claim in KNOWN_STATUSES and claim not in allowed:
                failures.append(f"fact_mismatch:{entity}_status:{claim}")
    return tuple(dict.fromkeys(failures))
