"""Evidence-only safety controller for learned v18 agents.

The controller never reads ``TaskSpec.scenario``, ``required_*`` fields, or the
frozen ``terminal_v18`` answer.  It uses only user turns, the currently exposed
tool surface, and audited tool results.  This mirrors a production gateway:
the model proposes actions and wording, while deterministic code owns write
authorization and the structured terminal decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

from .simulator import AgentRun
from .terminal_v18 import parse_terminal_response
from .tools import ToolExecution


POLICY_CONTROLLER_V18_VERSION = "ecommerce_cs_policy_controller_v18_r6"
_CONTROLLED_WRITES = {"cancel_order", "update_shipping_address", "create_refund"}


@dataclass(frozen=True)
class BlockedProposal:
    name: str
    arguments: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "reason": self.reason}


@dataclass(frozen=True)
class PolicyResolution:
    decision: str
    action_taken: bool
    action_id: str | None
    message: str
    evidence: str

    def response(self) -> str:
        return json.dumps(
            {
                "decision": self.decision,
                "action_taken": self.action_taken,
                "action_id": self.action_id,
                "message": self.message,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _parse_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(arguments, str):
        value = json.loads(arguments or "{}")
        if not isinstance(value, dict):
            raise ValueError("tool_arguments_must_be_object")
        return value
    return dict(arguments or {})


def _successful(executions: Iterable[Any], name: str) -> list[Any]:
    return [item for item in executions if item.name == name and item.ok]


def _inspected_order(executions: Iterable[Any], order_id: str) -> dict[str, Any] | None:
    for execution in reversed(_successful(executions, "get_order")):
        output = dict(execution.output or {})
        order = dict(output.get("order") or {})
        if str(order.get("id") or "") == order_id:
            return output
    return None


def _precheck_write(tools: Any, name: str, arguments: dict[str, Any]) -> str | None:
    visible = {item["function"]["name"] for item in tools.available_schemas()}
    order_id = str(arguments.get("order_id") or "")
    inspected = _inspected_order(tools.executions, order_id)
    if inspected is None:
        return "write_tool_not_exposed" if name not in visible else "order_not_inspected"
    order = dict(inspected.get("order") or {})
    status = str(order.get("status") or "").lower()
    if name == "cancel_order" and status not in {"pending", "paid"}:
        return "order_already_cancelled" if status == "cancelled" else "order_not_cancellable"
    if name == "update_shipping_address" and status not in {"pending", "paid"}:
        return "order_not_address_changeable"
    if name != "create_refund":
        return None if name in visible else "write_tool_not_exposed"

    item_id = str(arguments.get("order_item_id") or "")
    item = next(
        (
            dict(value)
            for value in inspected.get("items") or []
            if isinstance(value, dict)
            and str(value.get("id") or "") == item_id
            and str(value.get("order_id") or "") == order_id
        ),
        None,
    )
    if item is None:
        return "item_not_in_inspected_order"
    policies = [
        execution
        for execution in _successful(tools.executions, "search_refund_policy")
        if str(execution.arguments.get("order_item_id") or "") == item_id
        and str(execution.output.get("order_id") or "") == order_id
        and execution.output.get("allowed") is True
    ]
    if not policies:
        return "refund_policy_not_allowed"
    try:
        requested = int(arguments.get("quantity"))
    except (TypeError, ValueError):
        return "invalid_refund_quantity"
    already = sum(
        int(value.get("quantity") or 0)
        for value in inspected.get("returns") or []
        if isinstance(value, dict) and str(value.get("order_item_id") or "") == item_id
    )
    remaining = int(item.get("quantity") or 0) - already
    if requested < 1 or requested > remaining:
        return "return_quantity_exceeds_purchased"
    return None if name in visible else "write_tool_not_exposed"


class PolicyGuardedTools:
    """Proxy that blocks unsafe write proposals before business-tool execution."""

    def __init__(self, tools: Any) -> None:
        self._tools = tools
        self.blocked_proposals: list[BlockedProposal] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    def available_schemas(self) -> list[dict[str, Any]]:
        return self._tools.available_schemas()

    def execute(self, name: str, arguments: str | dict[str, Any] | None) -> ToolExecution:
        if name not in _CONTROLLED_WRITES:
            return self._tools.execute(name, arguments)
        try:
            parsed = _parse_arguments(arguments)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return self._tools.execute(name, arguments)
        reason = _precheck_write(self._tools, name, parsed)
        if reason is None:
            return self._tools.execute(name, parsed)
        self.blocked_proposals.append(BlockedProposal(name, parsed, reason))
        return ToolExecution(
            name=name,
            arguments=parsed,
            ok=False,
            output={
                "error": "blocked_by_policy_controller",
                "reason": reason,
                "write_executed": False,
            },
            is_write=True,
            policy_violation="blocked_by_policy_controller",
        )


def _user_text(messages: Iterable[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    )


def _requested_ids(text: str, prefix: str) -> set[str]:
    """Extract public order/item identifiers without relying on word boundaries.

    ``\b`` is a poor fit here because generated identifiers contain underscores
    and are frequently adjacent to Chinese punctuation.  Match the two public ID
    families (for example ``ord_...``/``ordalt_...``) explicitly instead.
    """

    pattern = rf"(?<![A-Za-z0-9]){re.escape(prefix)}[a-z0-9]*_[a-z0-9]+(?![A-Za-z0-9_])"
    return {value.lower() for value in re.findall(pattern, text, flags=re.IGNORECASE)}


def _ambiguous_order_request(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    uncertainty = any(
        value in compact
        for value in (
            "目标订单不确定",
            "订单信息存在矛盾",
            "可能说错订单",
            "也可能不是这个订单",
            "说混了",
            "说乱了",
            "弄混了",
        )
    )
    wait_for_user = any(
        value in compact
        for value in (
            "让我确认",
            "要求我确认",
            "等我给出准确编号",
            "先不要操作",
            "不要执行写操作",
            "不要改任何",
            "先别操作",
        )
    )
    return uncertainty and wait_for_user


def _intent(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text)
    if any(value in compact for value in ("地址", "收货地")) and any(
        value in compact for value in ("改", "修改", "更换", "更新")
    ):
        return "address"
    if "取消" in compact:
        return "cancel"
    if any(value in compact for value in ("退款", "退货", "退回", "多退", "都退")) or (
        "退" in compact and "件" in compact
    ):
        return "refund"
    return None


def _chinese_integer(value: str) -> int | None:
    """Parse the small Chinese quantities used in customer requests."""

    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value.isdigit():
        return int(value)
    if value in digits:
        return digits[value]
    if "十" not in value or value.count("十") != 1:
        return None
    left, right = value.split("十")
    tens = 1 if not left else digits.get(left)
    ones = 0 if not right else digits.get(right)
    if tens is None or ones is None:
        return None
    return tens * 10 + ones


def _requested_item_quantities(text: str) -> list[int]:
    values: list[int] = []
    for raw in re.findall(r"(?<![A-Za-z0-9])(\d{1,3}|[零〇一二两三四五六七八九十]{1,3})\s*件", text):
        parsed = _chinese_integer(raw)
        if parsed is not None:
            values.append(parsed)
    return values


def _latest_order_output(executions: list[Any]) -> dict[str, Any] | None:
    values = _successful(executions, "get_order")
    return dict(values[-1].output or {}) if values else None


def _money(cents: Any) -> str | None:
    try:
        return f"{int(cents) / 100:.2f}"
    except (TypeError, ValueError):
        return None


def _successful_write_resolution(executions: list[Any], fallback_message: str | None) -> PolicyResolution | None:
    writes = [item for item in executions if item.ok and item.is_write]
    if not writes:
        return None
    execution = writes[-1]
    output = dict(execution.output or {})
    if execution.name == "cancel_order":
        inspected = _latest_order_output(executions) or {}
        payments = list(inspected.get("payments") or [])
        captured = next((item for item in payments if item.get("status") in {"captured", "refunded"}), None)
        amount = _money((captured or {}).get("amount_cents"))
        canonical = (
            f"订单已取消，{amount} 元将按原支付路径退款。"
            if amount and output.get("refund_id")
            else "订单已取消；未发生实际扣款，因此无需退款。"
        )
        cancellation_markers = ("订单已取消", "订单已经取消", "取消成功")
        message = fallback_message or canonical
        if fallback_message and not any(value in fallback_message for value in cancellation_markers):
            message = f"订单已取消；{fallback_message}"
        return PolicyResolution("ORDER_CANCELLED", True, str(output.get("order_id")), message, "cancel_order")
    if execution.name == "update_shipping_address":
        address = str(output.get("shipping_address") or "")
        return PolicyResolution(
            "ADDRESS_UPDATED", True, str(output.get("order_id")),
            fallback_message or f"收货地址已更新为{address}。", "update_shipping_address"
        )
    if execution.name == "create_refund":
        amount = _money(output.get("amount_cents"))
        remaining = output.get("remaining_quantity")
        message = fallback_message or f"已创建退货并按原支付路径退款{(' ' + amount + ' 元') if amount else ''}。"
        if fallback_message and amount and amount not in message:
            message += f"退款金额为 {amount} 元。"
        if isinstance(remaining, int) and remaining > 0 and not re.search(
            rf"(?:还剩|剩余)\s*{remaining}\s*件", message
        ):
            message += f"订单中还剩 {remaining} 件未退。"
        return PolicyResolution("REFUND_CREATED", True, str(output.get("refund_id")), message, "create_refund")
    identity_failed = any(
        item.name == "verify_customer_identity" and not item.ok for item in executions
    )
    decision = "IDENTITY_ESCALATED" if identity_failed else "HUMAN_ESCALATED"
    message = fallback_message or "已创建人工工单，后续将由人工客服处理。"
    return PolicyResolution(decision, True, str(output.get("ticket_id")), message, "escalate_to_human")


def resolve_from_evidence(
    messages: Iterable[dict[str, Any]],
    executions: Iterable[Any],
    blocked: Iterable[BlockedProposal] = (),
) -> PolicyResolution | None:
    """Return a decision only when public evidence determines it unambiguously."""

    messages = list(messages)
    executions = list(executions)
    terminal = next(
        (
            parse_terminal_response(str(message.get("content") or "")).parsed
            for message in reversed(messages)
            if message.get("role") == "assistant" and not message.get("tool_calls")
        ),
        None,
    )
    fallback = terminal.message if terminal else None
    successful = _successful_write_resolution(executions, fallback)
    if successful:
        return successful

    text = _user_text(messages)
    intent = _intent(text)
    blocked = list(blocked)
    reasons = {item.reason for item in blocked}
    inspected = _latest_order_output(executions)
    requested_items = _requested_ids(text, "item")
    inspected_items = {
        str(value.get("id") or "").lower()
        for value in (inspected or {}).get("items") or []
        if isinstance(value, dict) and value.get("id")
    }
    queried_items = {
        str(item.arguments.get("order_item_id") or "").lower()
        for item in _successful(executions, "search_refund_policy")
        if item.arguments.get("order_item_id")
    }
    requested_item_mismatch = bool(
        inspected
        and requested_items
        and (
            requested_items.isdisjoint(inspected_items)
            or (queried_items and requested_items.isdisjoint(queried_items))
        )
    )
    if requested_item_mismatch or "item_not_in_inspected_order" in reasons or any(
        item.name == "search_refund_policy"
        and not item.ok
        and item.policy_violation == "item_not_in_inspected_order"
        for item in executions
    ):
        return PolicyResolution(
            "ITEM_ORDER_MISMATCH", False, None,
            "商品项不属于指定订单，商品与订单不匹配；我没有执行退款。",
            "item_order_mismatch",
        )
    if "return_quantity_exceeds_purchased" in reasons:
        return PolicyResolution(
            "QUANTITY_REJECTED", False, None,
            "申请退款数量超过实际可退款数量，我没有创建超额退款。",
            "blocked_quantity",
        )

    if inspected:
        order = dict(inspected.get("order") or {})
        status = str(order.get("status") or "").lower()
        terminal_already_cancelled = bool(terminal and terminal.decision == "ALREADY_CANCELLED")
        if (intent == "cancel" or terminal_already_cancelled) and status == "cancelled":
            return PolicyResolution(
                "ALREADY_CANCELLED", False, None,
                "该订单已经取消并且退款已经完成；我没有重复取消或重复退款。",
                "order_status_cancelled",
            )
        if intent == "cancel" and status in {"shipped", "delivered", "completed"}:
            return PolicyResolution(
                "ORDER_CANCELLATION_BLOCKED", False, None,
                "订单已经发货，无法取消；我没有执行取消或创建退款。",
                "order_status_fulfilled",
            )
        if intent == "address" and status in {"shipped", "delivered", "completed"}:
            return PolicyResolution(
                "ADDRESS_CHANGE_BLOCKED", False, None,
                "订单已经发货，无法修改收货地址；我没有执行地址变更。",
                "order_status_fulfilled",
            )
        if intent == "refund":
            requested_quantities = _requested_item_quantities(text)
            if requested_quantities:
                remaining = []
                returned_by_item: dict[str, int] = {}
                for value in inspected.get("returns") or []:
                    if isinstance(value, dict):
                        item_id = str(value.get("order_item_id") or "")
                        returned_by_item[item_id] = returned_by_item.get(item_id, 0) + int(
                            value.get("quantity") or 0
                        )
                for value in inspected.get("items") or []:
                    if isinstance(value, dict):
                        item_id = str(value.get("id") or "")
                        remaining.append(int(value.get("quantity") or 0) - returned_by_item.get(item_id, 0))
                if remaining and max(requested_quantities) > max(remaining):
                    return PolicyResolution(
                        "QUANTITY_REJECTED", False, None,
                        f"订单记录显示实际购买数量为 {max(remaining)} 件；"
                        f"不能按 {max(requested_quantities)} 件退款，"
                        "我没有创建超额退款。",
                        "requested_quantity_exceeds_order_evidence",
                    )

    if _ambiguous_order_request(text) and _successful(executions, "list_orders"):
        return PolicyResolution(
            "ORDER_CONFIRMATION_REQUIRED", False, None,
            "您提到了多笔订单，请确认准确订单号或目标订单后我再操作；目前没有修改任何订单。",
            "ambiguous_multiple_orders",
        )

    if intent == "refund":
        denied = [
            item for item in _successful(executions, "search_refund_policy")
            if item.output.get("allowed") is False
        ]
        if denied:
            message = fallback or str(
                denied[-1].output.get("message") or "退款政策不允许当前操作。"
            )
            if not any(value in message for value in ("不能", "无法", "不允许", "不支持")):
                message += "当前不能退款。"
            if not any(value in message for value in ("没有执行退款", "没有创建退款", "未执行退款")):
                message += "我没有执行退款。"
            return PolicyResolution("POLICY_BLOCKED", False, None, message, "refund_policy_denied")
    return None


def _complete_required_read(
    messages: list[dict[str, Any]], guarded: PolicyGuardedTools
) -> ToolExecution | None:
    """Inspect one explicitly requested, owned order when the model stops early."""

    text = _user_text(messages)
    if _ambiguous_order_request(text):
        return None
    requested_orders = _requested_ids(text, "ord")
    if len(requested_orders) != 1:
        return None
    requested_order = next(iter(requested_orders))
    tools = guarded._tools
    if any(item.ok and item.is_write for item in tools.executions):
        return None
    if any(
        item.name == "get_order"
        and str(item.arguments.get("order_id") or "").lower() == requested_order
        for item in tools.executions
    ):
        return None
    listed = _successful(tools.executions, "list_orders")
    if not listed:
        return None
    latest = listed[-1]
    customer_id = str(latest.arguments.get("customer_id") or "")
    listed_ids = {
        str(order.get("id") or "").lower()
        for order in latest.output.get("orders") or []
        if isinstance(order, dict)
        and str(order.get("customer_id") or "") == customer_id
        and order.get("id")
    }
    if not customer_id or requested_order not in listed_ids:
        return None
    return guarded.execute(
        "get_order", {"customer_id": customer_id, "order_id": requested_order}
    )


def _complete_item_binding_check(
    messages: list[dict[str, Any]], guarded: PolicyGuardedTools
) -> ToolExecution | None:
    """Ask the read-only policy tool to explicitly reject a cross-order item.

    A terminal mismatch is only evidence-complete after the public tool layer has
    rejected the exact item supplied by the user.  The model may instead query an
    item that belongs to the inspected order, so the controller performs this one
    bounded read when the requested order/item pair is unambiguously inconsistent.
    """

    text = _user_text(messages)
    requested_orders = _requested_ids(text, "ord")
    requested_items = _requested_ids(text, "item")
    if len(requested_orders) != 1 or len(requested_items) != 1:
        return None
    requested_order = next(iter(requested_orders))
    requested_item = next(iter(requested_items))
    tools = guarded._tools
    inspected = next(
        (
            item
            for item in reversed(_successful(tools.executions, "get_order"))
            if str(((item.output or {}).get("order") or {}).get("id") or "").lower()
            == requested_order
        ),
        None,
    )
    if inspected is None:
        return None
    inspected_items = {
        str(value.get("id") or "").lower()
        for value in (inspected.output or {}).get("items") or []
        if isinstance(value, dict) and value.get("id")
    }
    if requested_item in inspected_items:
        return None
    if any(
        item.name == "search_refund_policy"
        and str(item.arguments.get("order_item_id") or "").lower() == requested_item
        and (
            item.ok
            or item.policy_violation == "item_not_in_inspected_order"
        )
        for item in tools.executions
    ):
        return None
    customer_id = str(inspected.arguments.get("customer_id") or "")
    if not customer_id:
        return None
    return guarded.execute(
        "search_refund_policy",
        {"customer_id": customer_id, "order_item_id": requested_item},
    )


def _insert_controller_execution(
    messages: list[dict[str, Any]], execution: ToolExecution, suffix: str
) -> None:
    call_id = f"call_policy_controller_v18_{suffix}"
    call_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": execution.name,
                    "arguments": json.dumps(execution.arguments, ensure_ascii=False),
                },
            }
        ],
        "policy_controller": {"version": POLICY_CONTROLLER_V18_VERSION},
    }
    tool_message = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": execution.name,
        "content": json.dumps(execution.output, ensure_ascii=False),
    }
    final_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "assistant"
            and not messages[index].get("tool_calls")
        ),
        len(messages),
    )
    messages[final_index:final_index] = [call_message, tool_message]


def _execute_confirmed_safe_action(
    messages: list[dict[str, Any]], guarded: PolicyGuardedTools
) -> ToolExecution | None:
    """Complete a cancellation only when consent and order evidence are unambiguous."""

    if _intent(_user_text(messages)) != "cancel":
        return None
    tools = guarded._tools
    if not tools.has_observed_confirmation("cancel_order"):
        return None
    if any(item.ok and item.is_write for item in tools.executions):
        return None
    inspected = _latest_order_output(tools.executions)
    if not inspected:
        return None
    order = dict(inspected.get("order") or {})
    if str(order.get("status") or "").lower() not in {"pending", "paid"}:
        return None
    order_id = str(order.get("id") or "")
    customer_id = str(order.get("customer_id") or "")
    if not order_id or not customer_id:
        return None
    return guarded.execute(
        "cancel_order",
        {"customer_id": customer_id, "order_id": order_id, "confirmed": True},
    )


class PolicyControlledAgent:
    """Wrap any learned agent with pre-write checks and evidence-grounded terminal output."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.last_audit: dict[str, Any] = {}

    def run(self, task: Any, tools: Any) -> AgentRun:
        guarded = PolicyGuardedTools(tools)
        run = self.agent.run(task, guarded)
        messages = list(run.messages)
        controller_read = _complete_required_read(messages, guarded)
        if controller_read and controller_read.ok:
            _insert_controller_execution(messages, controller_read, "read")
        controller_evidence = _complete_item_binding_check(messages, guarded)
        if controller_evidence:
            _insert_controller_execution(messages, controller_evidence, "evidence")
        controller_action = _execute_confirmed_safe_action(messages, guarded)
        if controller_action and controller_action.ok:
            _insert_controller_execution(messages, controller_action, "action")
        resolution = (
            _successful_write_resolution(list(tools.executions), None)
            if controller_action and controller_action.ok
            else resolve_from_evidence(messages, tools.executions, guarded.blocked_proposals)
        )
        final = resolution.response() if resolution else run.final_response
        if resolution:
            for index in range(len(messages) - 1, -1, -1):
                if messages[index].get("role") == "assistant" and not messages[index].get("tool_calls"):
                    messages[index] = {
                        **messages[index],
                        "content": final,
                        "policy_controller": {
                            "version": POLICY_CONTROLLER_V18_VERSION,
                            "evidence": resolution.evidence,
                        },
                    }
                    break
        self.last_audit = {
            "version": POLICY_CONTROLLER_V18_VERSION,
            "blocked_proposals": [item.to_dict() for item in guarded.blocked_proposals],
            "terminal_normalized": bool(resolution),
            "terminal_evidence": resolution.evidence if resolution else None,
            "controller_read": controller_read.to_dict() if controller_read else None,
            "controller_evidence": (
                controller_evidence.to_dict() if controller_evidence else None
            ),
            "controller_action": controller_action.to_dict() if controller_action else None,
        }
        return AgentRun(tuple(messages), final, run.stopped_reason)
