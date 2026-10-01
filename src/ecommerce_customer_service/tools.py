from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterable

from .database import connect
from .models import TaskSpec
from .policies import can_cancel_order, can_return_item, can_update_address


WRITE_TOOLS = {
    "cancel_order",
    "update_shipping_address",
    "create_refund",
    "escalate_to_human",
}


@dataclass(frozen=True)
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    ok: bool
    output: dict[str, Any]
    is_write: bool
    policy_violation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row(value: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(value) if value is not None else None


class CustomerServiceTools:
    """Narrow business-tool surface over one isolated SQLite task snapshot."""

    def __init__(self, database_path: Path, task: TaskSpec) -> None:
        self.database_path = database_path.resolve()
        self.task = task
        self.executions: list[ToolExecution] = []
        self.verified_customer_id: str | None = None
        self._tool_attempts: dict[str, int] = {}
        self.observed_user_messages: list[str] = []

    def observe_user_message(self, content: str) -> None:
        """Record only real user turns so write tools can verify conversational consent."""

        if content.strip():
            self.observed_user_messages.append(content.strip())

    def _require_confirmation(self, confirmed: bool, action: str) -> None:
        if not confirmed:
            raise PermissionError("customer_confirmation_required")
        if self.has_observed_confirmation(action):
            return
        raise PermissionError("customer_confirmation_not_observed")

    def has_observed_confirmation(self, action: str) -> bool:
        """Return whether a real user turn explicitly confirms one write action."""

        return self.confirmation_in_messages(self.observed_user_messages, action)

    @staticmethod
    def confirmation_in_messages(messages: Iterable[str], action: str) -> bool:
        """Apply the runtime confirmation policy to an arbitrary user-message history."""

        negative = ("先不要", "不要操作", "暂不处理", "取消操作", "没有确认", "未确认", "不确认")
        positive_by_action = {
            "cancel_order": ("确认取消",),
            "update_shipping_address": ("确认修改", "确认更改", "确认更新地址"),
            "create_refund": ("确认退", "确认退款", "确认退货"),
        }
        positive = positive_by_action[action]
        for message in reversed(list(messages)):
            normalized = message.replace(" ", "")
            if any(marker in normalized for marker in negative):
                return False
            if any(marker in normalized for marker in positive):
                return True
        return False

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        def tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }

        string = {"type": "string"}
        return [
            tool(
                "verify_customer_identity",
                "使用姓名和手机号后四位核验客户身份。成功后返回内部 customer_id，之后才允许查询订单或执行写操作。",
                {"name": string, "phone_last4": string},
                ["name", "phone_last4"],
            ),
            tool("list_orders", "列出客户订单。", {"customer_id": string}, ["customer_id"]),
            tool(
                "get_order",
                "查询订单、商品和支付摘要。",
                {"customer_id": string, "order_id": string},
                ["customer_id", "order_id"],
            ),
            tool(
                "get_shipment",
                "查询订单物流状态和运单号。",
                {"customer_id": string, "order_id": string},
                ["customer_id", "order_id"],
            ),
            tool(
                "search_refund_policy",
                "检索指定商品的退款政策，并根据订单状态判断退款资格。",
                {"customer_id": string, "order_item_id": string},
                ["customer_id", "order_item_id"],
            ),
            tool(
                "cancel_order",
                "取消尚未履约的订单；执行前必须获得客户确认。",
                {"customer_id": string, "order_id": string, "confirmed": {"type": "boolean"}},
                ["customer_id", "order_id", "confirmed"],
            ),
            tool(
                "update_shipping_address",
                "修改尚未履约订单的收货地址；执行前必须获得客户确认。",
                {
                    "customer_id": string,
                    "order_id": string,
                    "new_address": string,
                    "confirmed": {"type": "boolean"},
                },
                ["customer_id", "order_id", "new_address", "confirmed"],
            ),
            tool(
                "create_refund",
                "为符合政策的商品创建退货记录并执行原路退款。",
                {
                    "customer_id": string,
                    "order_id": string,
                    "order_item_id": string,
                    "quantity": {"type": "integer", "minimum": 1},
                    "reason": string,
                    "confirmed": {"type": "boolean"},
                },
                ["customer_id", "order_id", "order_item_id", "quantity", "reason", "confirmed"],
            ),
            tool(
                "escalate_to_human",
                "为当前客服会话创建人工工单；身份核验失败时也可安全升级。",
                {"issue": string, "reason": string},
                ["issue", "reason"],
            ),
        ]

    @classmethod
    def schemas_for_state(
        cls,
        *,
        identity_verified: bool,
        successful_tools: Iterable[str] = (),
        observed_user_messages: Iterable[str] = (),
        failed_tool_counts: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the exact tool surface exposed for one conversation state."""

        schemas = {item["function"]["name"]: item for item in cls.schemas()}
        successful = set(successful_tools)
        failures = dict(failed_tool_counts or {})
        if "escalate_to_human" in successful:
            return []
        if not identity_verified:
            names = ["verify_customer_identity"]
            if failures.get("verify_customer_identity", 0) >= 1:
                names.append("escalate_to_human")
            return [schemas[name] for name in names]

        names = [
            "list_orders",
            "get_order",
            "get_shipment",
            "search_refund_policy",
        ]
        if any(
            count >= 2
            for name, count in failures.items()
            if name not in {"verify_customer_identity", "escalate_to_human"}
        ):
            names.append("escalate_to_human")
        observed = list(observed_user_messages)
        if "get_order" in successful:
            if cls.confirmation_in_messages(observed, "cancel_order"):
                names.append("cancel_order")
            if cls.confirmation_in_messages(observed, "update_shipping_address"):
                names.append("update_shipping_address")
        if {"get_order", "search_refund_policy"}.issubset(successful) and cls.confirmation_in_messages(
            observed,
            "create_refund"
        ):
            names.append("create_refund")
        return [schemas[name] for name in names]

    def available_schemas(self) -> list[dict[str, Any]]:
        """Expose only tools that are valid in the current verified conversation state."""

        if any(item.name == "escalate_to_human" and item.ok for item in self.executions):
            return []
        schemas = self.schemas_for_state(
            identity_verified=self.verified_customer_id is not None,
            successful_tools={item.name for item in self.executions if item.ok},
            observed_user_messages=self.observed_user_messages,
            failed_tool_counts={
                name: sum(
                    not item.ok
                    and (name == "verify_customer_identity" or item.policy_violation is None)
                    for item in self.executions
                    if item.name == name
                )
                for name in {item.name for item in self.executions}
            },
        )
        minimum_attempts = {
            str(name): int(count)
            for name, count in dict(self.task.metadata.get("min_attempts_before_escalation") or {}).items()
        }
        if any(self._tool_attempts.get(name, 0) < count for name, count in minimum_attempts.items()):
            schemas = [item for item in schemas if item["function"]["name"] != "escalate_to_human"]
        return schemas

    def _owned_order(self, connection: sqlite3.Connection, customer_id: str, order_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM orders WHERE id = ? AND customer_id = ?",
            (order_id, customer_id),
        ).fetchone()
        if row is None:
            raise PermissionError("order_not_found_or_not_owned")
        return row

    def _require_identity(self, customer_id: str) -> None:
        if self.verified_customer_id != customer_id:
            raise PermissionError("identity_verification_required")

    def verify_customer_identity(self, name: str, phone_last4: str) -> dict[str, Any]:
        with connect(self.database_path) as connection:
            customer = _row(connection.execute(
                """SELECT id, name, phone_last4, tier FROM customers
                   WHERE lower(name) = lower(?) AND phone_last4 = ?""",
                (name.strip(), phone_last4.strip()),
            ).fetchone())
        if customer is None:
            raise PermissionError("identity_mismatch")
        customer_id = str(customer["id"])
        self.verified_customer_id = customer_id
        return {"verified": True, "customer_id": customer_id, "tier": customer["tier"]}

    def list_orders(self, customer_id: str) -> dict[str, Any]:
        self._require_identity(customer_id)
        with connect(self.database_path) as connection:
            orders = [dict(row) for row in connection.execute(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY created_at DESC",
                (customer_id,),
            )]
        return {"orders": orders}

    def get_order(self, customer_id: str, order_id: str) -> dict[str, Any]:
        self._require_identity(customer_id)
        with connect(self.database_path) as connection:
            order = self._owned_order(connection, customer_id, order_id)
            items = [dict(row) for row in connection.execute(
                """SELECT oi.*, p.name AS product_name, p.returnable, p.return_window_days
                   FROM order_items oi JOIN products p ON p.id = oi.product_id
                   WHERE oi.order_id = ? ORDER BY oi.id""",
                (order_id,),
            )]
            payments = [dict(row) for row in connection.execute(
                "SELECT * FROM payments WHERE order_id = ? ORDER BY id", (order_id,)
            )]
            returns = [dict(row) for row in connection.execute(
                "SELECT * FROM returns WHERE order_id = ? ORDER BY created_at, id", (order_id,)
            )]
            refunds = [dict(row) for row in connection.execute(
                "SELECT * FROM refunds WHERE order_id = ? ORDER BY created_at, id", (order_id,)
            )]
        return {
            "order": dict(order),
            "items": items,
            "payments": payments,
            "returns": returns,
            "refunds": refunds,
        }

    def get_shipment(self, customer_id: str, order_id: str) -> dict[str, Any]:
        self._require_identity(customer_id)
        with connect(self.database_path) as connection:
            self._owned_order(connection, customer_id, order_id)
            shipment = _row(connection.execute(
                "SELECT * FROM shipments WHERE order_id = ?", (order_id,)
            ).fetchone())
        if shipment is None:
            raise LookupError("shipment_not_found")
        return {"shipment": shipment}

    def search_refund_policy(self, customer_id: str, order_item_id: str) -> dict[str, Any]:
        self._require_identity(customer_id)
        with connect(self.database_path) as connection:
            row = connection.execute(
                """SELECT oi.id AS order_item_id, oi.order_id, oi.quantity, oi.status,
                          o.customer_id, s.delivered_at, p.returnable, p.return_window_days
                   FROM order_items oi
                   JOIN orders o ON o.id = oi.order_id
                   JOIN products p ON p.id = oi.product_id
                   LEFT JOIN shipments s ON s.order_id = o.id
                   WHERE oi.id = ? AND o.customer_id = ?""",
                (order_item_id, customer_id),
            ).fetchone()
        if row is None:
            raise PermissionError("order_item_not_found_or_not_owned")
        decision = can_return_item(
            delivered_at=row["delivered_at"],
            reference_time=self.task.reference_time,
            returnable=bool(row["returnable"]),
            return_window_days=int(row["return_window_days"]),
        )
        return {
            "allowed": decision.allowed,
            "code": decision.code,
            "message": decision.message,
            "details": decision.details,
            "order_id": row["order_id"],
            "order_item_id": order_item_id,
        }

    def cancel_order(self, customer_id: str, order_id: str, confirmed: bool) -> dict[str, Any]:
        self._require_identity(customer_id)
        self._require_confirmation(confirmed, "cancel_order")
        with connect(self.database_path) as connection:
            order = self._owned_order(connection, customer_id, order_id)
            decision = can_cancel_order(str(order["status"]))
            if not decision.allowed:
                raise PermissionError(decision.code)
            payment = connection.execute(
                "SELECT * FROM payments WHERE order_id = ? AND status = 'captured' ORDER BY id LIMIT 1",
                (order_id,),
            ).fetchone()
            connection.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order_id,))
            refund_id = None
            if payment is not None:
                refund_id = f"refund_cancel_{order_id}"
                connection.execute("UPDATE payments SET status = 'refunded' WHERE id = ?", (payment["id"],))
                connection.execute(
                    """INSERT OR IGNORE INTO refunds
                       (id, order_id, return_id, payment_id, amount_cents, status, created_at)
                       VALUES (?, ?, NULL, ?, ?, 'completed', ?)""",
                    (refund_id, order_id, payment["id"], payment["amount_cents"], self.task.reference_time),
                )
        return {"cancelled": True, "order_id": order_id, "refund_id": refund_id}

    def update_shipping_address(
        self,
        customer_id: str,
        order_id: str,
        new_address: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        self._require_identity(customer_id)
        self._require_confirmation(confirmed, "update_shipping_address")
        if not new_address.strip():
            raise ValueError("empty_shipping_address")
        with connect(self.database_path) as connection:
            order = self._owned_order(connection, customer_id, order_id)
            decision = can_update_address(str(order["status"]))
            if not decision.allowed:
                raise PermissionError(decision.code)
            connection.execute(
                "UPDATE orders SET shipping_address = ? WHERE id = ?",
                (new_address.strip(), order_id),
            )
        return {"updated": True, "order_id": order_id, "shipping_address": new_address.strip()}

    def create_refund(
        self,
        customer_id: str,
        order_id: str,
        order_item_id: str,
        quantity: int,
        reason: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        self._require_identity(customer_id)
        self._require_confirmation(confirmed, "create_refund")
        if int(quantity) <= 0:
            raise ValueError("invalid_return_quantity")
        with connect(self.database_path) as connection:
            self._owned_order(connection, customer_id, order_id)
            item = connection.execute(
                """SELECT oi.*, p.returnable, p.return_window_days, s.delivered_at
                   FROM order_items oi
                   JOIN products p ON p.id = oi.product_id
                   LEFT JOIN shipments s ON s.order_id = oi.order_id
                   WHERE oi.id = ? AND oi.order_id = ?""",
                (order_item_id, order_id),
            ).fetchone()
            if item is None:
                raise LookupError("order_item_not_found")
            decision = can_return_item(
                delivered_at=item["delivered_at"],
                reference_time=self.task.reference_time,
                returnable=bool(item["returnable"]),
                return_window_days=int(item["return_window_days"]),
            )
            if not decision.allowed:
                raise PermissionError(decision.code)
            already = connection.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM returns WHERE order_item_id = ?",
                (order_item_id,),
            ).fetchone()[0]
            if already + int(quantity) > int(item["quantity"]):
                raise ValueError("return_quantity_exceeds_purchased")
            return_id = f"return_{order_item_id}_{already + 1}"
            connection.execute(
                """INSERT INTO returns
                   (id, order_id, order_item_id, quantity, reason, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'approved', ?)""",
                (return_id, order_id, order_item_id, int(quantity), reason.strip(), self.task.reference_time),
            )
            payment = connection.execute(
                "SELECT * FROM payments WHERE order_id = ? ORDER BY id LIMIT 1", (order_id,)
            ).fetchone()
            if payment is None:
                raise LookupError("payment_not_found")
            amount = int(quantity) * int(item["unit_price_cents"])
            refund_id = f"refund_{return_id}"
            connection.execute(
                """INSERT OR IGNORE INTO refunds
                   (id, order_id, return_id, payment_id, amount_cents, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'completed', ?)""",
                (refund_id, order_id, return_id, payment["id"], amount, self.task.reference_time),
            )
            connection.execute("UPDATE returns SET status = 'refunded' WHERE id = ?", (return_id,))
            returned_after = int(already) + int(quantity)
            remaining_quantity = int(item["quantity"]) - returned_after
            item_status = "returned" if remaining_quantity == 0 else "partially_returned"
            connection.execute("UPDATE order_items SET status = ? WHERE id = ?", (item_status, order_item_id))
        return {
            "return_id": return_id,
            "refunded": True,
            "refund_id": refund_id,
            "refunded_quantity": int(quantity),
            "amount_cents": amount,
            "currency": "CNY",
            "remaining_quantity": remaining_quantity,
        }

    def escalate_to_human(self, issue: str, reason: str) -> dict[str, Any]:
        customer_id = self.verified_customer_id or self.task.user_id
        with connect(self.database_path) as connection:
            customer = connection.execute("SELECT 1 FROM customers WHERE id = ?", (customer_id,)).fetchone()
            if customer is None:
                raise LookupError("customer_not_found")
            count = connection.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0]
            ticket_id = f"ticket_{customer_id}_{count + 1}"
            connection.execute(
                """INSERT INTO support_tickets
                   (id, customer_id, issue, reason, status, created_at)
                   VALUES (?, ?, ?, ?, 'open', ?)""",
                (ticket_id, customer_id, issue.strip(), reason.strip(), self.task.reference_time),
            )
        return {
            "escalated": True,
            "ticket_id": ticket_id,
            "reason": reason.strip(),
            "next_action": "向用户说明失败原因和工单号并立即结束；不要再调用任何工具",
        }

    def execute(self, name: str, arguments: str | dict[str, Any] | None) -> ToolExecution:
        parsed: dict[str, Any] = {}
        violation: str | None = None
        try:
            known_names = {item["function"]["name"] for item in self.schemas()}
            if name not in known_names:
                raise ValueError("unknown_tool")
            available_names = {item["function"]["name"] for item in self.available_schemas()}
            if name not in available_names:
                raise PermissionError("tool_not_available_in_current_state")
            parsed = json.loads(arguments or "{}") if isinstance(arguments, str) else dict(arguments or {})
            attempt = self._tool_attempts.get(name, 0) + 1
            self._tool_attempts[name] = attempt
            fault = dict((self.task.metadata.get("tool_faults") or {}).get(name) or {})
            if attempt <= int(fault.get("failures") or 0):
                raise TimeoutError(str(fault.get("error") or "tool_timeout"))
            handler: Callable[..., dict[str, Any]] | None = getattr(self, name, None)
            if handler is None or name.startswith("_"):
                raise ValueError("unknown_tool")
            output = handler(**parsed)
            ok = True
        except PermissionError as exc:
            violation = str(exc)
            output = {"error": violation}
            ok = False
        except (LookupError, ValueError, TypeError, AttributeError, TimeoutError, json.JSONDecodeError) as exc:
            output = {"error": str(exc) or type(exc).__name__}
            ok = False
        execution = ToolExecution(name, parsed, ok, output, name in WRITE_TOOLS, violation)
        self.executions.append(execution)
        return execution
