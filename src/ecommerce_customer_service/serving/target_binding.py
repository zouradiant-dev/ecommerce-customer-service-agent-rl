"""Conservative server-side matching for supported cancellation expressions.

This is a bounded grammar, not a general natural-language intent classifier.
Unsupported expressions require clarification; model-selected IDs are not evidence.
"""
from __future__ import annotations

from decimal import Decimal
import re
from typing import Any

from .storage import ServiceError

ALIASES = {
    "无线耳机": ("耳机",), "机械键盘": ("键盘",), "电动牙刷": ("牙刷",),
    "咖啡豆": ("咖啡",), "瑜伽垫": ("垫子",), "蓝牙音箱": ("音箱",),
    "运动鞋": ("鞋子",), "电热水壶": ("水壶",), "电风扇": ("风扇",),
    "电饭煲": ("饭煲",), "双肩背包": ("背包",), "电磁炉": ("炉子",),
    "书桌": ("桌子",), "空气净化器": ("净化器",), "行车记录仪": ("记录仪",),
    "电烤箱": ("烤箱",), "旅行箱": ("行李箱",),
}
PATTERNS = (
    r"^(?:请帮我|请|帮我|麻烦|我想|现在|改为|改成)?(?:取消|撤销)([^，,。！？?!]+)(?:[，,。！？?!].*)?$",
    r"^请处理(.+?)的取消申请[。！？?!]?$",
    r"^(.+?)(?:这单|这一单|订单)可以帮我取消吗[？?]?$",
    r"^我不想要(.+?)了[，,]请取消这单[。！!]?$",
    r"^(.+?)不用了[，,](?:请帮我|帮我|请)取消[。！!]?$",
    r"^那只(.+?)不想要了[，,](?:请)?取消订单[。！!]?$",
)
SUFFIX = re.compile(r"(?:的那一笔|那一笔|那笔|这一笔|这笔|这一单|这单|订单|。|\.)$")
AMOUNT = re.compile(r"(?:金额\s*)?(\d+(?:\.\d{1,2})?)\s*元(?:的)?(?:那一笔|那笔|那单|这一笔)?$")


def subject(text: str) -> str | None:
    for pattern in PATTERNS:
        matched = re.fullmatch(pattern, text.strip())
        if matched:
            value = matched.group(1).strip()
            if pattern == PATTERNS[0]:
                tail = re.split(r"[，,。！？?!]", text.strip(), maxsplit=1)
                if len(tail) == 2 and tail[1].strip("，,。！？?! "):
                    kept = re.fullmatch(r"([^，,。！？?!]+?)(?:先)?保留[。！？?!]?", tail[1].strip())
                    if not kept:
                        return None
                    variants = {value, SUFFIX.sub("", value).strip()}
                    for product, aliases in ALIASES.items():
                        if variants & {product, *aliases}:
                            variants.update({product, *aliases})
                    if kept.group(1) in variants:
                        return None
            while SUFFIX.search(value):
                value = SUFFIX.sub("", value).strip()
            return value.removeprefix("订单 ").removeprefix("刚才说的").strip() or None
    return None


def resolve(text: str, previous_users: list[str], orders: list[dict[str, Any]]) -> dict[str, Any]:
    target = subject(text)
    amount_text = target if target and AMOUNT.fullmatch(target) else text.strip(" 。")
    amount_match = AMOUNT.fullmatch(amount_text)
    followup = text.strip(" 。") in {"确认取消", "确定取消", "确认", "确定"} or amount_match is not None
    if followup:
        # Only the immediately preceding user request may supply the target.
        # An old preview or a model reply must never manufacture a cancellation intent.
        target = subject(previous_users[-1]) if previous_users else None
    if not target or target in {"一笔", "订单", "这", "那", "刚才那笔"}:
        raise ServiceError("target_needs_clarification", 409)
    candidates = [row for row in orders if target == row.get("id") or
                  target == row.get("product") or target in ALIASES.get(row.get("product"), ())]
    if amount_match:
        cents = int(Decimal(amount_match.group(1)) * 100)
        candidates = [row for row in candidates if row.get("paid_cents") == cents]
    if not candidates:
        raise ServiceError("target_not_found", 409)
    if len(candidates) != 1:
        raise ServiceError("target_ambiguous", 409)
    return {"order_id": candidates[0]["id"], "source": "user_request_and_fresh_owned_orders",
            "order_version": candidates[0]["version"], "matching_version": "bounded_target_v2", "candidate_count": 1}


def require_target(store, backend, customer: str, conversation: str, request: str, order: str) -> dict:
    with store.transaction() as db:
        store.require_owner(db, customer, conversation)
        turn = db.execute("SELECT text,status FROM turns WHERE conversation_id=? AND request_id=?",
                          (conversation, request)).fetchone()
        if not turn or turn["status"] != "running":
            raise ServiceError("turn_not_active", 409)
    messages = store.model_history(customer, conversation)
    users = [str(message.get("content", "")) for message in messages if message.get("role") == "user"]
    listing = backend.list_orders(customer)
    rows = listing.get("orders") if isinstance(listing, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) or row.get("customer_id") != customer
                                        or not isinstance(row.get("id"), str) or type(row.get("version")) is not int
                                        or row["version"] < 1 for row in rows):
        raise ServiceError("invalid_owned_order_evidence", 502)
    evidence = resolve(turn["text"], users[:-1], rows)
    if evidence["order_id"] != order:
        raise ServiceError("target_order_mismatch", 409)
    store.record(conversation, request, "target_binding", {"order_id": order}, evidence)
    return evidence


TARGET_MESSAGES = {
    "target_order_changed": "订单在目标核验后发生变化，未创建取消申请。请重新查询订单。",
    "target_not_found": "没有找到与您指定商品或金额匹配的本人订单，未创建取消申请。请核对商品或订单号。",
    "target_ambiguous": "有多笔匹配订单，未创建取消申请。请指定订单号或金额。",
    "target_needs_clarification": "请明确要取消的商品或订单号，本轮未创建取消申请。",
    "target_order_mismatch": "所选订单与您的需求不一致，未创建取消申请。请核对商品或订单号。",
}


def preview_claim(text: str) -> bool:
    return bool(re.search(r"(?:已经|已)(?:为您)?(?:生成|创建).{0,12}(?:预览|申请)|"
                          r"已有.{0,8}(?:预览|申请)|请.{0,12}点击.{0,18}确认", text))
