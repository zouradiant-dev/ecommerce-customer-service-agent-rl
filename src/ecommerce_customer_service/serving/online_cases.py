"""Synthetic development regressions; fixtures are for engineering smoke only."""
from __future__ import annotations
from typing import Any


def tool(name: str, **arguments: Any) -> dict[str, Any]:
    import json
    return {"tool_calls": [{"id": "fixture", "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}


def say(text: str) -> dict[str, Any]:
    return {"content": text}


def prepare(order: str, *, listing: bool = True) -> list[dict[str, Any]]:
    return ([tool("list_orders")] if listing else []) + [tool("get_order", order_id=order),
        tool("preview_cancel", order_id=order), say("请确认取消申请。")]


def defaults() -> list[dict[str, Any]]:
    cases = []
    def add(name, category, steps, fixtures, *, changes=None, refunds=0, cancellations=0, **expected):
        state = {"order_A": "paid", "order_B": "paid", "order_C": "shipped", "order_D": "paid"}
        state.update(changes or {})
        cases.append({"schema_version": "online_agent_case_v1", "case_id": name, "split": "dev",
            "category": category, "customer_id": "customer_alice", "steps": steps,
            "expected": {"order_statuses": state, "refund_count": refunds, "cancellation_count": cancellations,
                         "forbidden_reply_phrases": ["退款已到账", "退款已经到账"],
                         "conversation_status": "active", "pending_count": 0, **expected},
            "smoke_responses": fixtures})
    def message(text, save_as="latest"):
        return {"kind": "message", "text": text, "save_as": save_as}
    def confirm(ref="latest", error=None):
        return {"kind": "confirm", "ref": ref, **({"expected_error": error} if error else {})}
    def event(name, **values):
        return {"kind": "event", "name": name, **values}

    add("cancel_earphone", "basic", [message("请帮我取消无线耳机订单。"), confirm()], prepare("order_A"),
        changes={"order_A": "cancelled"}, refunds=1, cancellations=1, operation_status="succeeded", refund_status="pending")
    add("disambiguate_order", "memory", [message("我想取消一笔订单。"), message("取消键盘那一笔。"), confirm()],
        [tool("list_orders"), say("请问取消耳机还是键盘订单？")] + prepare("order_B", listing=False),
        changes={"order_B": "cancelled"}, refunds=1, cancellations=1)
    add("change_target", "memory", [message("取消耳机订单。", "old"), message("改成取消键盘，耳机先保留。", "new"),
        confirm("old", "confirmation_invalidated"), confirm("new")], prepare("order_A") + prepare("order_B", listing=False),
        changes={"order_B": "cancelled"}, refunds=1, cancellations=1)
    add("withdraw_confirmation", "memory", [message("取消耳机订单。", "old"), message("先不要取消了。"),
        confirm("old", "confirmation_invalidated")], prepare("order_A") + [say("好的，暂不取消。")], pending_count=0)
    add("shipped_order", "policy", [message("帮我取消鼠标订单。")],
        [tool("list_orders"), tool("get_order", order_id="order_C"), say("鼠标订单已发货，不能直接取消。")], pending_count=0)
    add("other_customer_order", "policy", [message("请取消 order_D。")],
        [tool("get_order", order_id="order_D"), say("无法访问这笔订单，请核对您自己的订单。")], pending_count=0)
    add("state_changed", "race", [message("取消耳机。"), event("change_order", order_id="order_A", status="shipped"), confirm()],
        prepare("order_A"), changes={"order_A": "shipped"}, operation_status="failed")
    add("confirmation_expired", "policy", [message("取消耳机。"), event("expire_confirmation"), confirm(error="confirmation_expired")],
        prepare("order_A"))
    add("duplicate_confirmation", "idempotency", [message("取消耳机。"), confirm(), confirm()], prepare("order_A"),
        changes={"order_A": "cancelled"}, refunds=1, cancellations=1, operation_status="succeeded")
    add("commit_then_timeout", "recovery", [message("取消耳机。"), event("lose_ack"), confirm(), {"kind": "query", "ref": "latest"}],
        prepare("order_A"), changes={"order_A": "cancelled"}, refunds=1, cancellations=1, operation_status="succeeded")
    add("unknown_result", "recovery", [message("取消耳机。"), event("backend_unavailable"), confirm(), confirm(),
        {"kind": "query", "ref": "latest"}], prepare("order_A"), operation_status="unknown")
    add("restart_then_confirm", "recovery", [message("取消耳机。"), event("restart"), confirm()], prepare("order_A"),
        changes={"order_A": "cancelled"}, refunds=1, cancellations=1)
    add("long_history_reference", "memory", [message("先看一下我的耳机订单。")]
        + [message(f"我再补充一句，这是第 {index} 条咨询信息，暂时不要操作。") for index in range(12)]
        + [message("现在取消刚才说的耳机订单。"), confirm()],
        [tool("list_orders"), tool("get_order", order_id="order_A"), say("耳机订单尚未发货。")]
        + [say("好的，暂不操作。") for _ in range(12)] + prepare("order_A", listing=False),
        changes={"order_A": "cancelled"}, refunds=1, cancellations=1, require_context_omission=True)
    add("human_handoff", "handoff", [message("请转人工处理。"),
        {"kind": "message", "text": "再帮我取消耳机。", "expected_error": "conversation_handed_off"}],
        [tool("escalate_to_human", reason="用户要求人工")], conversation_status="handed_off", pending_count=0)
    add("read_transient_error", "recovery", [event("read_timeout_once"), message("请取消耳机订单。"), confirm()],
        [tool("list_orders"), tool("get_order", order_id="order_A"), tool("get_order", order_id="order_A"),
         tool("preview_cancel", order_id="order_A"), say("请确认取消申请。")],
        changes={"order_A": "cancelled"}, refunds=1, cancellations=1)
    return cases
