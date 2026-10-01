from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def can_cancel_order(status: str) -> PolicyDecision:
    if status in {"pending", "paid"}:
        return PolicyDecision(True, "cancel_allowed", "订单尚未履约，可以取消。")
    return PolicyDecision(False, "cancel_not_allowed", f"订单状态为 {status}，不能直接取消。")


def can_update_address(status: str) -> PolicyDecision:
    if status in {"pending", "paid"}:
        return PolicyDecision(True, "address_update_allowed", "订单尚未进入履约，可以修改地址。")
    return PolicyDecision(False, "address_update_not_allowed", f"订单状态为 {status}，不能修改地址。")


def can_return_item(
    *,
    delivered_at: str | None,
    reference_time: str,
    returnable: bool,
    return_window_days: int,
) -> PolicyDecision:
    if not returnable:
        return PolicyDecision(False, "product_not_returnable", "该商品属于不支持退货的品类。")
    if not delivered_at:
        return PolicyDecision(False, "not_delivered", "订单尚未签收，不能创建售后退货。")
    delivered = datetime.fromisoformat(delivered_at.replace("Z", "+00:00"))
    current = datetime.fromisoformat(reference_time.replace("Z", "+00:00"))
    elapsed = (current - delivered).days
    if elapsed < 0:
        return PolicyDecision(False, "invalid_delivery_time", "签收时间晚于当前时间。")
    if elapsed > return_window_days:
        return PolicyDecision(
            False,
            "return_window_expired",
            f"商品已超过{return_window_days}天退货期限。",
            {"elapsed_days": elapsed, "return_window_days": return_window_days},
        )
    return PolicyDecision(
        True,
        "return_allowed",
        "商品在退货期限内，可以创建退货。",
        {"elapsed_days": elapsed, "return_window_days": return_window_days},
    )

