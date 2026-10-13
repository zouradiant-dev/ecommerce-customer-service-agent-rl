"""Extra synthetic dev guards, authored before ablation scores; not frozen test."""
from copy import deepcopy
from .online_cases import prepare, say, tool


def cases() -> list[dict]:
    base = [
        {"id": "R-731", "customer_id": "customer_alice", "product": "电动牙刷", "status": "paid", "paid_cents": 12900, "version": 1},
        {"id": "R-864", "customer_id": "customer_alice", "product": "旅行箱", "status": "paid", "paid_cents": 45900, "version": 1},
        {"id": "R-905", "customer_id": "customer_alice", "product": "床头灯", "status": "shipped", "paid_cents": 15900, "version": 1},
        {"id": "R-228", "customer_id": "customer_bob", "product": "保温杯", "status": "paid", "paid_cents": 8900, "version": 1},
    ]
    result = []
    def message(text, save_as="latest"):
        return {"kind": "message", "text": text, "save_as": save_as}
    def confirm(ref="latest", error=None):
        return {"kind": "confirm", "ref": ref, **({"expected_error": error} if error else {})}
    def checkpoint(pending=0):
        return {"kind": "checkpoint", "assertions": {"pending_count": pending, "cancellation_count": 0}}
    def add(name, category, steps, fixtures, *, orders=None, cancel=None, pending=0):
        rows = deepcopy(orders if orders is not None else base)
        statuses = {row["id"]: row["status"] for row in rows}
        amount = next((row["paid_cents"] if row["status"] == "paid" else 0 for row in rows if row["id"] == cancel), 0)
        if cancel:
            statuses[cancel] = "cancelled"
        result.append({"schema_version": "online_agent_case_v2", "case_id": name, "split": "dev", "category": category,
            "customer_id": "customer_alice", "initial_orders": rows, "steps": steps,
            "expected": {"order_statuses": statuses, "refund_count": int(bool(cancel and amount)),
                "cancellation_count": int(bool(cancel)), "pending_count": pending, "conversation_status": "active",
                "forbidden_reply_phrases": ["退款已到账", "退款已经到账"],
                **({"refund_status": "pending"} if cancel and amount else {})}, "smoke_responses": fixtures})

    add("new_product_exact", "binding", [message("帮我取消电动牙刷这单。"), confirm()], prepare("R-731"), cancel="R-731")
    add("product_short_name", "binding", [message("牙刷不用了，帮我取消。"), confirm()], prepare("R-731"), cancel="R-731")
    add("product_paraphrase", "binding", [message("那只行李箱不想要了，取消订单。"), confirm()], prepare("R-864"), cancel="R-864")
    add("unseen_order_id", "binding", [message("取消 R-864。"), confirm()], prepare("R-864"), cancel="R-864")
    duplicated = deepcopy(base)
    duplicated[1]["product"] = "电动牙刷"
    add("same_product_clarification", "ambiguity", [message("取消牙刷。"), checkpoint(), message("129 元那笔。"), confirm()],
        [tool("list_orders"), say("两笔牙刷订单，您指哪一笔？")] + prepare("R-731", listing=False), orders=duplicated, cancel="R-731")
    add("no_matching_product", "policy", [message("取消洗衣机订单。"), checkpoint()],
        [tool("list_orders"), say("没有找到您的洗衣机订单。")])
    add("query_does_not_preview", "intent", [message("只查一下牙刷订单，暂时别取消。"), checkpoint()],
        [tool("list_orders"), tool("get_order", order_id="R-731"), say("牙刷订单已付款，未取消。")])
    add("different_shipped_product", "policy", [message("取消床头灯。"), checkpoint()],
        [tool("list_orders"), tool("get_order", order_id="R-905"), say("床头灯已发货，不能直接取消。")])
    add("withdraw_new_product", "intent", [message("取消牙刷。", "old"), checkpoint(1), message("先保留，不取消了。"),
        confirm("old", "confirmation_invalidated")], prepare("R-731") + [say("好的，保留订单。")])
    add("switch_new_products", "memory", [message("取消牙刷。", "old"), checkpoint(1), message("改为取消旅行箱，牙刷保留。", "new"),
        confirm("old", "confirmation_invalidated"), confirm("new")], prepare("R-731") + prepare("R-864", listing=False), cancel="R-864")
    unpaid = deepcopy(base)
    unpaid[0].update(status="pending", paid_cents=0)
    add("unpaid_no_refund", "policy", [message("取消牙刷订单。"), confirm()], prepare("R-731"), orders=unpaid, cancel="R-731")
    add("different_foreign_order", "policy", [message("帮我取消 R-228。"), checkpoint()],
        [tool("get_order", order_id="R-228"), say("无法访问这笔订单。")])
    return result
