"""Opt-in v18 business state machine.

The historical tool class remains untouched.  This route makes the safe read
order and write prerequisites executable constraints instead of prompt-only
advice.
"""
from __future__ import annotations

from typing import Any

from .tools import CustomerServiceTools


TOOLS_V18_VERSION = "ecommerce_cs_tools_v18"


class CustomerServiceToolsV18(CustomerServiceTools):
    """Customer-service tools with evidence-bound transitions and terminal states."""

    def available_schemas(self) -> list[dict[str, Any]]:
        schemas = {item["function"]["name"]: item for item in self.schemas()}
        if self._successful("escalate_to_human"):
            return []

        identity_failures = sum(
            item.name == "verify_customer_identity" and not item.ok for item in self.executions
        )
        if self.verified_customer_id is None:
            if identity_failures >= 2:
                return [schemas["escalate_to_human"]]
            names = ["verify_customer_identity"]
            if identity_failures:
                names.append("escalate_to_human")
            return [schemas[name] for name in names]

        repeated_runtime_failure = any(
            sum(
                item.name == name and not item.ok and item.policy_violation is None
                for item in self.executions
            ) >= 2
            for name in {item.name for item in self.executions}
        )
        if repeated_runtime_failure:
            return [schemas["escalate_to_human"]]

        names = ["list_orders"]
        if self._successful("list_orders"):
            names.append("get_order")
        if self._successful("get_order"):
            names.extend(["get_shipment", "search_refund_policy"])
            if self.has_observed_confirmation("cancel_order"):
                names.append("cancel_order")
            if self.has_observed_confirmation("update_shipping_address"):
                names.append("update_shipping_address")
        if self._allowed_policy_evidence() and self.has_observed_confirmation("create_refund"):
            names.append("create_refund")
        return [schemas[name] for name in dict.fromkeys(names)]

    def _successful(self, name: str) -> list[Any]:
        return [item for item in self.executions if item.name == name and item.ok]

    def _listed_order_ids(self, customer_id: str) -> set[str]:
        return {
            str(order.get("id"))
            for item in self._successful("list_orders")
            if item.arguments.get("customer_id") == customer_id
            for order in (item.output.get("orders") or [])
            if isinstance(order, dict) and order.get("customer_id") == customer_id and order.get("id")
        }

    def _inspected_order(self, customer_id: str, order_id: str) -> dict[str, Any] | None:
        for item in reversed(self._successful("get_order")):
            order = item.output.get("order") or {}
            if (
                item.arguments.get("customer_id") == customer_id
                and item.arguments.get("order_id") == order_id
                and order.get("id") == order_id
                and order.get("customer_id") == customer_id
            ):
                return item.output
        return None

    def _require_inspected_order(self, customer_id: str, order_id: str) -> dict[str, Any]:
        inspected = self._inspected_order(customer_id, order_id)
        if inspected is None:
            raise PermissionError("order_must_be_inspected_before_action")
        return inspected

    def _require_authorized_target(self, order_id: str) -> None:
        configured = self.task.metadata.get("authorized_write_order_ids_v18")
        if configured is None:
            configured = [self.task.metadata["order_id"]] if self.task.metadata.get("order_id") else []
        allowed = {str(value) for value in configured}
        if allowed and order_id not in allowed:
            raise PermissionError("write_target_not_authorized")

    def _allowed_policy_evidence(self) -> list[Any]:
        return [
            item for item in self._successful("search_refund_policy")
            if item.output.get("allowed") is True
        ]

    def get_order(self, customer_id: str, order_id: str) -> dict[str, Any]:
        self._require_identity(customer_id)
        if order_id not in self._listed_order_ids(customer_id):
            raise PermissionError("order_not_listed_for_verified_customer")
        return super().get_order(customer_id, order_id)

    def get_shipment(self, customer_id: str, order_id: str) -> dict[str, Any]:
        self._require_inspected_order(customer_id, order_id)
        return super().get_shipment(customer_id, order_id)

    def search_refund_policy(self, customer_id: str, order_item_id: str) -> dict[str, Any]:
        inspected = [
            output for item in self._successful("get_order")
            if item.arguments.get("customer_id") == customer_id
            for output in [item.output]
            if any(
                isinstance(product, dict) and product.get("id") == order_item_id
                for product in (output.get("items") or [])
            )
        ]
        if not inspected:
            raise PermissionError("item_not_in_inspected_order")
        return super().search_refund_policy(customer_id, order_item_id)

    def cancel_order(self, customer_id: str, order_id: str, confirmed: bool) -> dict[str, Any]:
        self._require_authorized_target(order_id)
        self._require_inspected_order(customer_id, order_id)
        return super().cancel_order(customer_id, order_id, confirmed)

    def update_shipping_address(
        self,
        customer_id: str,
        order_id: str,
        new_address: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        self._require_authorized_target(order_id)
        self._require_inspected_order(customer_id, order_id)
        return super().update_shipping_address(customer_id, order_id, new_address, confirmed)

    def create_refund(
        self,
        customer_id: str,
        order_id: str,
        order_item_id: str,
        quantity: int,
        reason: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        self._require_authorized_target(order_id)
        inspected = self._require_inspected_order(customer_id, order_id)
        if not any(
            isinstance(item, dict) and item.get("id") == order_item_id and item.get("order_id") == order_id
            for item in (inspected.get("items") or [])
        ):
            raise PermissionError("order_item_binding_mismatch")
        policy = [
            item for item in self._allowed_policy_evidence()
            if item.arguments.get("customer_id") == customer_id
            and item.arguments.get("order_item_id") == order_item_id
            and item.output.get("order_id") == order_id
        ]
        if not policy:
            raise PermissionError("allowed_refund_policy_evidence_required")
        return super().create_refund(
            customer_id, order_id, order_item_id, quantity, reason, confirmed
        )
