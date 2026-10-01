from __future__ import annotations

from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable

from .models import TaskSpec, VerificationResult
from .rewards import RewardBreakdown, score_verification, select_reward
from .simulator import SYSTEM_PROMPT
from .tools import CustomerServiceTools
from .verifier import verify_task


class EcommerceCustomerServiceEnv:
    """One stateful customer-service environment per online RL rollout.

    The public business methods are intentionally compatible with TRL's
    ``environment_factory`` tool discovery. Every reset copies the immutable
    task database, so parallel rollouts never share mutable business state.
    """

    def __init__(self, *, task_path: Path, reward_mode: str = "strict_binary") -> None:
        self._task_path = Path(task_path).resolve()
        self._reward_mode = reward_mode
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._task: TaskSpec | None = None
        self._tools: CustomerServiceTools | None = None
        self._last_verification: VerificationResult | None = None
        self._last_breakdown: RewardBreakdown | None = None

    def reset(self, **_: Any) -> None:
        """Reset the isolated SQLite snapshot before a rollout."""

        self._cleanup()
        source = TaskSpec.from_file(self._task_path)
        self._temporary = tempfile.TemporaryDirectory(prefix="ecommerce-cs-online-")
        database_path = Path(self._temporary.name) / "state.sqlite"
        shutil.copy2(source.database_path, database_path)
        self._task = replace(source, database_path=database_path)
        self._tools = CustomerServiceTools(database_path, self._task)
        self._tools.observe_user_message(self._task.user_prompt)
        self._last_verification = None
        self._last_breakdown = None
        return None

    def verify_customer_identity(self, name: str, phone_last4: str) -> dict[str, Any]:
        """Verify the customer and unlock access to that customer's resources.

        Args:
            name: Customer name supplied in the conversation.
            phone_last4: Last four digits of the customer's phone number.

        Returns:
            Verification result containing the internal customer identifier.
        """

        return self._execute("verify_customer_identity", {"name": name, "phone_last4": phone_last4})

    def list_orders(self, customer_id: str) -> dict[str, Any]:
        """List orders owned by a verified customer.

        Args:
            customer_id: Internal customer identifier returned by identity verification.

        Returns:
            The customer's orders or a safe error.
        """

        return self._execute("list_orders", {"customer_id": customer_id})

    def get_order(self, customer_id: str, order_id: str) -> dict[str, Any]:
        """Read one owned order, its items, payments, returns and refunds.

        Args:
            customer_id: Verified internal customer identifier.
            order_id: Order identifier to inspect.

        Returns:
            Order facts required before a write decision.
        """

        return self._execute("get_order", {"customer_id": customer_id, "order_id": order_id})

    def get_shipment(self, customer_id: str, order_id: str) -> dict[str, Any]:
        """Read shipment status for an owned order.

        Args:
            customer_id: Verified internal customer identifier.
            order_id: Order identifier whose shipment should be read.

        Returns:
            Shipment facts or a safe error.
        """

        return self._execute("get_shipment", {"customer_id": customer_id, "order_id": order_id})

    def search_refund_policy(self, customer_id: str, order_item_id: str) -> dict[str, Any]:
        """Evaluate the refund policy for an owned order item.

        Args:
            customer_id: Verified internal customer identifier.
            order_item_id: Item identifier returned by order lookup.

        Returns:
            Auditable refund eligibility and reason.
        """

        return self._execute(
            "search_refund_policy",
            {"customer_id": customer_id, "order_item_id": order_item_id},
        )

    def cancel_order(self, customer_id: str, order_id: str, confirmed: bool) -> dict[str, Any]:
        """Cancel an eligible unshipped order after explicit confirmation.

        Args:
            customer_id: Verified internal customer identifier.
            order_id: Eligible order identifier.
            confirmed: Whether the user explicitly confirmed cancellation.

        Returns:
            Cancellation and refund result or a safe error.
        """

        return self._execute(
            "cancel_order",
            {"customer_id": customer_id, "order_id": order_id, "confirmed": confirmed},
        )

    def update_shipping_address(
        self,
        customer_id: str,
        order_id: str,
        new_address: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        """Update an eligible unshipped order after explicit confirmation.

        Args:
            customer_id: Verified internal customer identifier.
            order_id: Eligible order identifier.
            new_address: Complete replacement shipping address.
            confirmed: Whether the user explicitly confirmed the change.

        Returns:
            Address update result or a safe error.
        """

        return self._execute(
            "update_shipping_address",
            {
                "customer_id": customer_id,
                "order_id": order_id,
                "new_address": new_address,
                "confirmed": confirmed,
            },
        )

    def create_refund(
        self,
        customer_id: str,
        order_id: str,
        order_item_id: str,
        quantity: int,
        reason: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        """Create an eligible item refund after policy lookup and confirmation.

        Args:
            customer_id: Verified internal customer identifier.
            order_id: Owned order identifier.
            order_item_id: Eligible item identifier.
            quantity: Number of units to return.
            reason: Customer-provided return reason.
            confirmed: Whether the user explicitly confirmed the refund.

        Returns:
            Return and refund identifiers or a safe error.
        """

        return self._execute(
            "create_refund",
            {
                "customer_id": customer_id,
                "order_id": order_id,
                "order_item_id": order_item_id,
                "quantity": quantity,
                "reason": reason,
                "confirmed": confirmed,
            },
        )

    def escalate_to_human(self, issue: str, reason: str) -> dict[str, Any]:
        """Create a human-support ticket when safe automation cannot continue.

        Args:
            issue: Short description of the customer issue.
            reason: Auditable reason why human support is required.

        Returns:
            Human-support ticket information or a safe error.
        """

        return self._execute("escalate_to_human", {"issue": issue, "reason": reason})

    def _execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._tools is None:
            raise RuntimeError("environment_not_reset")
        execution = self._tools.execute(name, arguments)
        return {
            "ok": execution.ok,
            "tool": execution.name,
            **execution.output,
        }

    def _evaluate(self, final_response: str) -> float:
        if self._task is None or self._tools is None:
            raise RuntimeError("environment_not_reset")
        executions = tuple(item.to_dict() for item in self._tools.executions)
        self._last_verification = verify_task(self._task, final_response, executions)
        self._last_breakdown = score_verification(self._last_verification)
        return select_reward(self._last_breakdown, self._reward_mode)

    def _cleanup(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __del__(self) -> None:
        self._cleanup()


def build_environment_factories(
    task_paths: Iterable[Path],
    *,
    reward_mode: str = "strict_binary",
) -> dict[str, Callable[[], EcommerceCustomerServiceEnv]]:
    factories: dict[str, Callable[[], EcommerceCustomerServiceEnv]] = {}
    for task_path in task_paths:
        task = TaskSpec.from_file(task_path)
        factories[task.task_id] = partial(
            EcommerceCustomerServiceEnv,
            task_path=Path(task_path).resolve(),
            reward_mode=reward_mode,
        )
    return factories


def build_online_rows(task_paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one routed TRL row per inline-dialogue training task.

    TRL's generic tool loop does not synthesize additional user turns. Tasks
    that require scripted user replies are therefore excluded rather than
    silently leaking identity or confirmation into environment state.
    """

    rows: list[dict[str, Any]] = []
    excluded: list[str] = []
    for task_path in task_paths:
        task = TaskSpec.from_file(task_path)
        if task.split == "test":
            raise ValueError("test tasks must not enter online GRPO")
        if task.metadata.get("scripted_user_replies"):
            excluded.append(task.task_id)
            continue
        rows.append(
            {
                "task_id": task.task_id,
                "environment": task.task_id,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task.user_prompt},
                ],
            }
        )
    return rows, {
        "eligible_tasks": len(rows),
        "excluded_scripted_dialogue_tasks": len(excluded),
        "excluded_task_ids": excluded,
    }


def strict_environment_reward(
    completions: list[Any],
    environments: list[EcommerceCustomerServiceEnv],
    **_: Any,
) -> list[float]:
    """Score completed online episodes against isolated database state."""

    if len(completions) != len(environments):
        raise ValueError("completion_environment_length_mismatch")
    return [
        environment._evaluate(_final_response(completion))
        for completion, environment in zip(completions, environments)
    ]


def _final_response(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content") or "")
    return json.dumps(completion, ensure_ascii=False, default=str)
