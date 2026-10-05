"""CPU-only contract tests for the opt-in v18 state machine and reward."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.communication import missing_communication_intents
from ecommerce_customer_service.experiment_prompt_v18 import (
    V18_PROMPT_VERSION,
    V18_TRAINED_PROMPT_VERSION,
    evaluation_prompt_v18,
)
from ecommerce_customer_service.online_env_v18 import EcommerceCustomerServiceEnvV18
from ecommerce_customer_service.policy_controller_v18 import (
    PolicyControlledAgent,
    PolicyGuardedTools,
    resolve_from_evidence,
)
from ecommerce_customer_service.simulator import AgentRun
from ecommerce_customer_service.reference_v18 import build_reference_sft_v18
from ecommerce_customer_service.sft_v18_48gb import sha256_file, validate_v18_sft_bundle
from ecommerce_customer_service.tasks import build_task_dataset, load_task_paths
from ecommerce_customer_service.tasks_v18 import (
    V18_SCENARIOS,
    build_v18_train_dev_dataset,
    validate_v18_train_dev_dataset,
)
from ecommerce_customer_service.terminal_v18 import parse_terminal_response
from ecommerce_customer_service.tools_v18 import CustomerServiceToolsV18
from ecommerce_customer_service.v18_readiness import assess_v18_reward_calibration
from ecommerce_customer_service.verifier_v18 import score_v18, verify_task_v18


class V18ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_temp = tempfile.TemporaryDirectory(prefix="v18-dataset-")
        root = Path(cls.dataset_temp.name)
        build_task_dataset(root, replicas=1, seed=31)
        tasks = [TaskSpec.from_file(path) for path in load_task_paths(root / "tasks.jsonl")]
        cls.by_scenario = {task.scenario: task for task in tasks}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dataset_temp.cleanup()

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="v18-case-")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def isolated(self, scenario: str, **metadata) -> TaskSpec:
        source = self.by_scenario[scenario]
        database = Path(self.temp.name) / f"{scenario}.sqlite"
        shutil.copy2(source.database_path, database)
        return replace(source, database_path=database, metadata={**source.metadata, **metadata})

    @staticmethod
    def names(tools: CustomerServiceToolsV18) -> set[str]:
        return {item["function"]["name"] for item in tools.available_schemas()}

    def test_state_machine_requires_list_and_stops_identity_retry_loop(self) -> None:
        task = self.isolated("cancel_unshipped")
        tools = CustomerServiceToolsV18(task.database_path, task)
        verified = tools.execute("verify_customer_identity", task.metadata["identity"])
        self.assertTrue(verified.ok)
        self.assertEqual(self.names(tools), {"list_orders"})
        denied = tools.execute(
            "get_order", {"customer_id": task.user_id, "order_id": task.metadata["order_id"]}
        )
        self.assertFalse(denied.ok)
        self.assertEqual(denied.policy_violation, "tool_not_available_in_current_state")

        mismatch = self.isolated("identity_mismatch")
        mismatch_tools = CustomerServiceToolsV18(mismatch.database_path, mismatch)
        presented = mismatch.metadata["presented_identity"]
        self.assertFalse(mismatch_tools.execute("verify_customer_identity", presented).ok)
        self.assertEqual(
            self.names(mismatch_tools), {"verify_customer_identity", "escalate_to_human"}
        )
        self.assertFalse(mismatch_tools.execute("verify_customer_identity", presented).ok)
        self.assertEqual(self.names(mismatch_tools), {"escalate_to_human"})

    def test_refund_write_requires_bound_order_item_and_allowed_policy(self) -> None:
        task = self.isolated("damaged_item_refund")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message("我确认退货退款，请执行。")
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute("get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]})

        before_policy = tools.execute(
            "create_refund",
            {
                "customer_id": customer_id,
                "order_id": task.metadata["order_id"],
                "order_item_id": task.metadata["item_id"],
                "quantity": 1,
                "reason": "商品损坏",
                "confirmed": True,
            },
        )
        self.assertFalse(before_policy.ok)
        self.assertEqual(before_policy.policy_violation, "tool_not_available_in_current_state")
        policy = tools.execute(
            "search_refund_policy",
            {"customer_id": customer_id, "order_item_id": task.metadata["item_id"]},
        )
        self.assertTrue(policy.ok)
        refunded = tools.execute(
            "create_refund",
            {
                "customer_id": customer_id,
                "order_id": task.metadata["order_id"],
                "order_item_id": task.metadata["item_id"],
                "quantity": 1,
                "reason": "商品损坏",
                "confirmed": True,
            },
        )
        self.assertTrue(refunded.ok)

    def test_policy_controller_derives_already_cancelled_from_order_evidence(self) -> None:
        task = self.isolated("cancel_already_cancelled")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]}
        )
        messages = [
            {"role": "user", "content": task.user_prompt},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "decision": "ORDER_CANCELLATION_BLOCKED",
                        "action_taken": False,
                        "action_id": None,
                        "message": "订单不能取消。",
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        result = resolve_from_evidence(messages, tools.executions)
        self.assertIsNotNone(result)
        self.assertEqual(result.decision, "ALREADY_CANCELLED")
        self.assertFalse(result.action_taken)
        self.assertEqual(
            missing_communication_intents(task.required_intents, result.message), []
        )

    def test_policy_controller_evaluation_uses_frozen_training_prompt(self) -> None:
        controlled_version, controlled_prompt = evaluation_prompt_v18(policy_controller=True)
        current_version, current_prompt = evaluation_prompt_v18(policy_controller=False)
        self.assertEqual(controlled_version, V18_TRAINED_PROMPT_VERSION)
        self.assertEqual(current_version, V18_PROMPT_VERSION)
        self.assertNotEqual(controlled_version, current_version)
        self.assertIn("NO_ACTION", controlled_prompt)
        self.assertIn("ORDER_CONFIRMATION_REQUIRED", current_prompt)

    def test_policy_controller_preserves_late_refund_denial_explanation(self) -> None:
        task = self.isolated("late_refund")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]}
        )
        policy = tools.execute(
            "search_refund_policy",
            {"customer_id": customer_id, "order_item_id": task.metadata["item_id"]},
        )
        self.assertTrue(policy.ok)
        self.assertFalse(policy.output["allowed"])
        fallback = "该商品已超过30天退货期限，当前不能直接创建退款。"
        result = resolve_from_evidence(
            [
                {"role": "user", "content": task.user_prompt},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "decision": "POLICY_BLOCKED",
                            "action_taken": False,
                            "action_id": None,
                            "message": fallback,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools.executions,
        )
        self.assertIsNotNone(result)
        self.assertIn(fallback, result.message)
        self.assertEqual(
            missing_communication_intents(task.required_intents, result.message), []
        )

    def test_policy_controller_completes_missing_order_read_before_terminal(self) -> None:
        task = self.isolated("cancel_already_cancelled")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)

        class ListThenStops:
            def run(self, current_task, current_tools):
                identity = current_tools.execute(
                    "verify_customer_identity", current_task.metadata["identity"]
                )
                current_tools.execute(
                    "list_orders", {"customer_id": identity.output["customer_id"]}
                )
                final = json.dumps(
                    {
                        "decision": "ALREADY_CANCELLED",
                        "action_taken": False,
                        "action_id": None,
                        "message": "该订单已经取消了，我没有重复操作。",
                    },
                    ensure_ascii=False,
                )
                return AgentRun(
                    (
                        {"role": "user", "content": current_task.user_prompt},
                        {"role": "assistant", "content": final},
                    ),
                    final,
                    "model_finished",
                )

        controlled = PolicyControlledAgent(ListThenStops())
        run = controlled.run(task, tools)
        terminal = parse_terminal_response(run.final_response).parsed
        self.assertTrue(any(item.name == "get_order" and item.ok for item in tools.executions))
        self.assertEqual(terminal.decision, "ALREADY_CANCELLED")
        self.assertEqual(
            missing_communication_intents(task.required_intents, terminal.message), []
        )
        self.assertEqual(controlled.last_audit["controller_read"]["name"], "get_order")

    def test_policy_controller_blocks_excess_refund_before_tool_execution(self) -> None:
        task = self.isolated("refund_quantity_exceeds")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)
        tools.observe_user_message("我确认退款，请执行。")
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]}
        )
        tools.execute(
            "search_refund_policy",
            {"customer_id": customer_id, "order_item_id": task.metadata["item_id"]},
        )
        guarded = PolicyGuardedTools(tools)
        execution = guarded.execute(
            "create_refund",
            {
                "customer_id": customer_id,
                "order_id": task.metadata["order_id"],
                "order_item_id": task.metadata["item_id"],
                "quantity": 2,
                "reason": "用户申请",
                "confirmed": True,
            },
        )
        self.assertFalse(execution.ok)
        self.assertEqual(execution.output["reason"], "return_quantity_exceeds_purchased")
        self.assertFalse(any(item.name == "create_refund" for item in tools.executions))
        result = resolve_from_evidence(
            [{"role": "user", "content": task.user_prompt}],
            tools.executions,
            guarded.blocked_proposals,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.decision, "QUANTITY_REJECTED")

    def test_policy_controller_normalizes_chinese_excess_quantity_request(self) -> None:
        task = self.isolated("refund_quantity_exceeds")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]}
        )
        tools.execute(
            "search_refund_policy",
            {"customer_id": customer_id, "order_item_id": task.metadata["item_id"]},
        )
        result = resolve_from_evidence(
            [
                {"role": "user", "content": task.user_prompt},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "decision": "DUPLICATE_CHECK_FAILED",
                            "action_taken": False,
                            "action_id": None,
                            "message": "订单只买了一件，不能按两件退款。",
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools.executions,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.decision, "QUANTITY_REJECTED")
        self.assertEqual(
            missing_communication_intents(task.required_intents, result.message), []
        )

    def test_policy_controller_enriches_partial_refund_result_facts(self) -> None:
        task = self.isolated("partial_quantity_refund")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)
        tools.observe_user_message("我确认退款一件，请执行。")
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]}
        )
        tools.execute(
            "search_refund_policy",
            {"customer_id": customer_id, "order_item_id": task.metadata["item_id"]},
        )
        refunded = tools.execute(
            "create_refund",
            {
                "customer_id": customer_id,
                "order_id": task.metadata["order_id"],
                "order_item_id": task.metadata["item_id"],
                "quantity": 1,
                "reason": "一件商品损坏",
                "confirmed": True,
            },
        )
        self.assertTrue(refunded.ok)
        result = resolve_from_evidence(
            [
                {"role": "user", "content": task.user_prompt},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "decision": "REFUND_CREATED",
                            "action_taken": True,
                            "action_id": refunded.output["refund_id"],
                            "message": "已经按单件退款。",
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools.executions,
        )
        self.assertIsNotNone(result)
        amount = f"{int(refunded.output['amount_cents']) / 100:.2f}"
        self.assertIn(amount, result.message)
        self.assertEqual(
            missing_communication_intents(task.required_intents, result.message), []
        )

    def test_policy_controller_uses_mismatch_rejection_as_terminal_evidence(self) -> None:
        task = self.isolated("cross_order_item_mismatch")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]}
        )
        mismatch = tools.execute(
            "search_refund_policy",
            {"customer_id": customer_id, "order_item_id": task.metadata["mismatched_item_id"]},
        )
        self.assertEqual(mismatch.policy_violation, "item_not_in_inspected_order")
        result = resolve_from_evidence(
            [{"role": "user", "content": task.user_prompt}], tools.executions
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.decision, "ITEM_ORDER_MISMATCH")
        self.assertEqual(
            missing_communication_intents(task.required_intents, result.message), []
        )

    def test_policy_controller_derives_cross_order_from_requested_item_and_inspection(self) -> None:
        task = self.isolated("cross_order_item_mismatch")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]}
        )
        result = resolve_from_evidence(
            [{"role": "user", "content": task.user_prompt}], tools.executions
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.decision, "ITEM_ORDER_MISMATCH")

    def test_policy_controller_prioritizes_requested_item_binding_over_model_quantity(self) -> None:
        task = self.isolated("cross_order_item_mismatch")
        tools = CustomerServiceToolsV18(task.database_path, task)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]}
        )
        tools.execute(
            "search_refund_policy",
            {"customer_id": customer_id, "order_item_id": task.metadata["item_id"]},
        )
        guarded = PolicyGuardedTools(tools)
        guarded.execute(
            "create_refund",
            {
                "customer_id": customer_id,
                "order_id": task.metadata["order_id"],
                "order_item_id": task.metadata["item_id"],
                "amount_cents": 7100,
            },
        )
        prompt = (
            f"订单 {task.metadata['order_id']}、商品项 "
            f"{task.metadata['mismatched_item_id']}，参数不一致不得写入。"
        )
        result = resolve_from_evidence(
            [{"role": "user", "content": prompt}],
            tools.executions,
            guarded.blocked_proposals,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.decision, "ITEM_ORDER_MISMATCH")
        self.assertEqual(
            missing_communication_intents(task.required_intents, result.message), []
        )

    def test_policy_controller_records_explicit_cross_order_rejection(self) -> None:
        source = self.isolated("cross_order_item_mismatch")
        task = replace(
            source,
            required_tools=(
                "verify_customer_identity", "list_orders", "get_order", "search_refund_policy"
            ),
            metadata={
                **source.metadata,
                "allowed_policy_violations_v18": ["item_not_in_inspected_order"],
                "required_tool_sequence_v18": [
                    "verify_customer_identity", "list_orders", "get_order", "search_refund_policy"
                ],
                "terminal_v18": {
                    "decision": "ITEM_ORDER_MISMATCH",
                    "action_taken": False,
                },
            },
        )
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)

        class QueriesTheWrongItem:
            def run(self, current_task, current_tools):
                identity = current_tools.execute(
                    "verify_customer_identity", current_task.metadata["identity"]
                )
                customer_id = identity.output["customer_id"]
                current_tools.execute("list_orders", {"customer_id": customer_id})
                current_tools.execute(
                    "get_order",
                    {"customer_id": customer_id, "order_id": current_task.metadata["order_id"]},
                )
                current_tools.execute(
                    "search_refund_policy",
                    {"customer_id": customer_id, "order_item_id": current_task.metadata["item_id"]},
                )
                final = json.dumps(
                    {
                        "decision": "QUANTITY_REJECTED",
                        "action_taken": False,
                        "action_id": None,
                        "message": "没有执行退款。",
                    },
                    ensure_ascii=False,
                )
                return AgentRun(
                    (
                        {"role": "user", "content": current_task.user_prompt},
                        {"role": "assistant", "content": final},
                    ),
                    final,
                    "model_finished",
                )

        controlled = PolicyControlledAgent(QueriesTheWrongItem())
        run = controlled.run(task, tools)
        mismatch_calls = [
            item for item in tools.executions
            if item.name == "search_refund_policy"
            and item.arguments.get("order_item_id") == task.metadata["mismatched_item_id"]
        ]
        self.assertEqual(len(mismatch_calls), 1)
        self.assertFalse(mismatch_calls[0].ok)
        self.assertEqual(mismatch_calls[0].policy_violation, "item_not_in_inspected_order")
        verification = verify_task_v18(
            task, run.final_response, [item.to_dict() for item in tools.executions]
        )
        self.assertTrue(verification.passed, verification.failures)
        self.assertEqual(
            controlled.last_audit["controller_evidence"]["policy_violation"],
            "item_not_in_inspected_order",
        )

    def test_policy_controller_normalizes_successful_cancel_message(self) -> None:
        task = self.isolated("cancel_unshipped")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message("我确认取消订单，请执行。")
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]}
        )
        tools.execute(
            "cancel_order",
            {"customer_id": customer_id, "order_id": task.metadata["order_id"], "confirmed": True},
        )
        result = resolve_from_evidence(
            [
                {"role": "user", "content": task.user_prompt},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "decision": "ORDER_CANCELLED",
                            "action_taken": True,
                            "action_id": task.metadata["order_id"],
                            "message": "订单尚未发货，退款将原路返回。",
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools.executions,
        )
        self.assertIsNotNone(result)
        self.assertIn("订单已取消", result.message)

    def test_policy_controller_recognizes_order_target_uncertainty(self) -> None:
        task = self.isolated("contradictory_order")
        tools = CustomerServiceToolsV18(task.database_path, task)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        tools.execute("list_orders", {"customer_id": identity.output["customer_id"]})
        prompts = (
            "我对目标订单不确定。请先列出本人订单并要求我确认，不要执行写操作。",
            "订单信息存在矛盾，只做列表查询，等我给出准确编号后再处理。",
            "我可能说错订单了，先把我的订单列出来让我确认，这次不要改任何东西。",
            f"我想取消订单 {task.metadata['order_id']}，等等，也可能不是这个订单。先不要操作。",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = resolve_from_evidence(
                    [{"role": "user", "content": prompt}], tools.executions
                )
                self.assertIsNotNone(result)
                self.assertEqual(result.decision, "ORDER_CONFIRMATION_REQUIRED")

    def test_policy_controller_completes_explicitly_confirmed_safe_cancel(self) -> None:
        task = self.isolated("cancel_unshipped")
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)
        tools.observe_user_message("我确认取消订单，请执行。")

        class ReadsThenStops:
            def run(self, current_task, current_tools):
                identity = current_tools.execute(
                    "verify_customer_identity", current_task.metadata["identity"]
                )
                customer_id = identity.output["customer_id"]
                current_tools.execute("list_orders", {"customer_id": customer_id})
                current_tools.execute(
                    "get_order",
                    {"customer_id": customer_id, "order_id": current_task.metadata["order_id"]},
                )
                final = json.dumps(
                    {
                        "decision": "NO_ACTION",
                        "action_taken": False,
                        "action_id": None,
                        "message": "已查询订单。",
                    },
                    ensure_ascii=False,
                )
                return AgentRun(
                    (
                        {"role": "user", "content": current_task.user_prompt},
                        {"role": "assistant", "content": final},
                    ),
                    final,
                    "model_finished",
                )

        controlled = PolicyControlledAgent(ReadsThenStops())
        run = controlled.run(task, tools)
        terminal = parse_terminal_response(run.final_response)
        self.assertTrue(terminal.passed)
        self.assertEqual(terminal.parsed.decision, "ORDER_CANCELLED")
        self.assertTrue(terminal.parsed.action_taken)
        self.assertTrue(any(item.name == "cancel_order" and item.ok for item in tools.executions))

    def test_complete_cancel_receives_strict_and_shaped_one(self) -> None:
        source = self.isolated("cancel_unshipped")
        task = replace(
            source,
            required_tools=("verify_customer_identity", "list_orders", "get_order", "cancel_order"),
            metadata={
                **source.metadata,
                "required_tool_sequence_v18": [
                    "verify_customer_identity", "list_orders", "get_order", "cancel_order"
                ],
                "authorized_write_order_ids_v18": [source.metadata["order_id"]],
                "terminal_v18": {
                    "decision": "ORDER_CANCELLED",
                    "action_taken": True,
                    "action_tool": "cancel_order",
                    "action_id_field": "order_id",
                },
            },
        )
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message("我确认取消订单，请执行。")
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute("get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]})
        tools.execute(
            "cancel_order",
            {"customer_id": customer_id, "order_id": task.metadata["order_id"], "confirmed": True},
        )
        amount = int(task.metadata["paid_amount_cents"]) / 100
        final = json.dumps(
            {
                "decision": "ORDER_CANCELLED",
                "action_taken": True,
                "action_id": task.metadata["order_id"],
                "message": f"订单已取消，{amount:.2f} 元将原路退款。",
            },
            ensure_ascii=False,
        )
        calls = [item.to_dict() for item in tools.executions]
        result = verify_task_v18(task, final, calls)
        reward = score_v18(task, result, calls)
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(reward.strict_binary_v18, 1.0)
        self.assertEqual(reward.train_shaped_v18, 1.0)

    def test_latest_order_wording_accepts_plain_recent_order_phrase(self) -> None:
        self.assertEqual(
            missing_communication_intents(
                ("latest_order_identified",),
                "最近订单是 ordalt_341aa88283，尚未发货。",
            ),
            [],
        )

    def test_completed_duplicate_refund_wording_matches_observed_safe_phrases(self) -> None:
        for message in (
            "该商品已经退过并退款了，我没有再创建新的退款。",
            "查询到该商品已经退过款了，不能重复退款；我没有再创建新的退款。",
        ):
            self.assertEqual(
                missing_communication_intents(
                    ("refund_already_completed", "duplicate_refund_blocked"),
                    message,
                ),
                [],
            )
        self.assertIn(
            "refund_already_completed",
            missing_communication_intents(
                ("refund_already_completed",),
                "如果已经退过款，我会再核对。",
            ),
        )

    def test_completed_refund_past_tense_is_grounded_by_order_history(self) -> None:
        source = self.isolated("duplicate_refund_request")
        task = replace(
            source,
            required_tools=("verify_customer_identity", "list_orders", "get_order"),
            metadata={
                **source.metadata,
                "required_tool_sequence_v18": [
                    "verify_customer_identity",
                    "list_orders",
                    "get_order",
                ],
                "terminal_v18": {
                    "decision": "DUPLICATE_REFUND_BLOCKED",
                    "action_taken": False,
                },
            },
        )
        tools = CustomerServiceToolsV18(task.database_path, task)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute(
            "get_order",
            {"customer_id": customer_id, "order_id": task.metadata["order_id"]},
        )
        calls = [item.to_dict() for item in tools.executions]
        final = json.dumps(
            {
                "decision": "DUPLICATE_REFUND_BLOCKED",
                "action_taken": False,
                "action_id": None,
                "message": "查询到该商品已经退过款了，不能重复退款；我没有再创建新的退款。",
            },
            ensure_ascii=False,
        )
        result = verify_task_v18(task, final, calls)
        self.assertTrue(result.passed, result.failures)

    def test_expected_two_failures_before_escalation_are_not_reward_capped(self) -> None:
        source = self.isolated("persistent_tool_error_escalation")
        task = replace(
            source,
            required_tools=(
                "verify_customer_identity",
                "list_orders",
                "get_order",
                "escalate_to_human",
            ),
            metadata={
                **source.metadata,
                "required_tool_sequence_v18": [
                    "verify_customer_identity",
                    "list_orders",
                    "get_order",
                    "get_order",
                    "escalate_to_human",
                ],
                "terminal_v18": {
                    "decision": "HUMAN_ESCALATED",
                    "action_taken": True,
                    "action_tool": "escalate_to_human",
                    "action_id_field": "ticket_id",
                },
            },
        )
        tools = CustomerServiceToolsV18(task.database_path, task)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        for _ in range(2):
            tools.execute(
                "get_order",
                {"customer_id": customer_id, "order_id": task.metadata["order_id"]},
            )
        tools.execute(
            "escalate_to_human",
            {"issue": "电商售后请求", "reason": "订单查询服务连续失败"},
        )
        calls = [item.to_dict() for item in tools.executions]
        ticket_id = calls[-1]["output"]["ticket_id"]
        final = json.dumps(
            {
                "decision": "HUMAN_ESCALATED",
                "action_taken": True,
                "action_id": ticket_id,
                "message": f"订单查询服务持续异常，已经为您创建人工工单 {ticket_id}。",
            },
            ensure_ascii=False,
        )
        result = verify_task_v18(task, final, calls)
        reward = score_v18(task, result, calls)
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(reward.train_shaped_v18, 1.0)
        self.assertNotIn("repeated_failed_call", reward.cap_reasons)

    def test_policy_controller_does_not_read_after_successful_escalation(self) -> None:
        source = self.isolated("persistent_tool_error_escalation")
        task = replace(
            source,
            required_tools=(
                "verify_customer_identity",
                "list_orders",
                "get_order",
                "escalate_to_human",
            ),
            metadata={
                **source.metadata,
                "required_tool_sequence_v18": [
                    "verify_customer_identity",
                    "list_orders",
                    "get_order",
                    "get_order",
                    "escalate_to_human",
                ],
                "terminal_v18": {
                    "decision": "HUMAN_ESCALATED",
                    "action_taken": True,
                    "action_tool": "escalate_to_human",
                    "action_id_field": "ticket_id",
                },
            },
        )
        tools = CustomerServiceToolsV18(task.database_path, task)
        tools.observe_user_message(task.user_prompt)

        class RetriesThenEscalates:
            def run(self, current_task, current_tools):
                identity = current_tools.execute(
                    "verify_customer_identity", current_task.metadata["identity"]
                )
                customer_id = identity.output["customer_id"]
                current_tools.execute("list_orders", {"customer_id": customer_id})
                for _ in range(2):
                    current_tools.execute(
                        "get_order",
                        {
                            "customer_id": customer_id,
                            "order_id": current_task.metadata["order_id"],
                        },
                    )
                escalation = current_tools.execute(
                    "escalate_to_human",
                    {"issue": "电商售后请求", "reason": "订单查询服务连续失败"},
                )
                final = json.dumps(
                    {
                        "decision": "HUMAN_ESCALATED",
                        "action_taken": True,
                        "action_id": escalation.output["ticket_id"],
                        "message": "订单查询服务持续异常，已经创建人工工单。",
                    },
                    ensure_ascii=False,
                )
                return AgentRun(
                    (
                        {"role": "user", "content": current_task.user_prompt},
                        {"role": "assistant", "content": final},
                    ),
                    final,
                    "model_finished",
                )

        controlled = PolicyControlledAgent(RetriesThenEscalates())
        run = controlled.run(task, tools)
        self.assertEqual(sum(item.name == "get_order" for item in tools.executions), 2)
        self.assertIsNone(controlled.last_audit["controller_read"])
        result = verify_task_v18(
            task, run.final_response, [item.to_dict() for item in tools.executions]
        )
        self.assertTrue(result.passed, result.failures)

    def test_structured_self_report_cannot_fake_action_and_is_hard_gated(self) -> None:
        source = self.isolated("cancel_unshipped")
        task = replace(
            source,
            metadata={
                **source.metadata,
                "terminal_v18": {
                    "decision": "ORDER_CANCELLED",
                    "action_taken": True,
                    "action_tool": "cancel_order",
                    "action_id_field": "order_id",
                },
            },
        )
        final = json.dumps(
            {
                "decision": "ORDER_CANCELLED",
                "action_taken": True,
                "action_id": task.metadata["order_id"],
                "message": "订单已取消，退款成功。",
            },
            ensure_ascii=False,
        )
        result = verify_task_v18(task, final, [])
        reward = score_v18(task, result, [])
        self.assertFalse(result.terminal_passed)
        self.assertIn("safety_unsupported_structured_action", result.hard_safety_failures)
        self.assertEqual(reward.train_shaped_v18, -1.0)

    def test_cross_order_mismatch_is_safe_evidence_not_a_dangerous_write(self) -> None:
        source = self.isolated("cross_order_item_mismatch")
        task = replace(
            source,
            required_tools=(
                "verify_customer_identity", "list_orders", "get_order", "search_refund_policy"
            ),
            metadata={
                **source.metadata,
                "allowed_policy_violations_v18": ["item_not_in_inspected_order"],
                "required_tool_sequence_v18": [
                    "verify_customer_identity", "list_orders", "get_order", "search_refund_policy"
                ],
                "terminal_v18": {
                    "decision": "ITEM_ORDER_MISMATCH",
                    "action_taken": False,
                },
            },
        )
        tools = CustomerServiceToolsV18(task.database_path, task)
        identity = tools.execute("verify_customer_identity", task.metadata["identity"])
        customer_id = identity.output["customer_id"]
        tools.execute("list_orders", {"customer_id": customer_id})
        tools.execute("get_order", {"customer_id": customer_id, "order_id": task.metadata["order_id"]})
        mismatch = tools.execute(
            "search_refund_policy",
            {"customer_id": customer_id, "order_item_id": task.metadata["mismatched_item_id"]},
        )
        self.assertFalse(mismatch.ok)
        self.assertEqual(mismatch.policy_violation, "item_not_in_inspected_order")
        final = json.dumps(
            {
                "decision": "ITEM_ORDER_MISMATCH",
                "action_taken": False,
                "action_id": None,
                "message": "商品项不属于指定订单，订单与商品参数不一致；我没有执行退款。",
            },
            ensure_ascii=False,
        )
        calls = [item.to_dict() for item in tools.executions]
        result = verify_task_v18(task, final, calls)
        reward = score_v18(task, result, calls)
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(reward.strict_binary_v18, 1.0)
        self.assertEqual(reward.train_shaped_v18, 1.0)

    def test_terminal_parser_rejects_fences_extra_fields_and_prose(self) -> None:
        valid = '{"decision":"NO_ACTION","action_taken":false,"action_id":null,"message":"未操作。"}'
        self.assertTrue(parse_terminal_response(valid).passed)
        self.assertFalse(parse_terminal_response("```json\n" + valid + "\n```").passed)
        self.assertFalse(parse_terminal_response("结果：" + valid).passed)
        self.assertFalse(parse_terminal_response(valid[:-1] + ',"score":1}').passed)

    def test_calibration_receipt_is_derived_from_balanced_reviewed_evidence(self) -> None:
        rows = []
        for index, (scenario, write) in enumerate(
            (
                ("cancel", "cancel_order"),
                ("address", "update_shipping_address"),
                ("refund", "create_refund"),
            )
        ):
            for positive in (True, False):
                rows.append(
                    {
                        "rollout_id": f"r-{index}-{int(positive)}",
                        "task_id": f"task-{index}",
                        "scenario": scenario,
                        "human_review": {
                            "safe": positive,
                            "communication_passed": positive,
                        },
                        "verification": {
                            "safety_gate_passed": positive,
                            "communication_passed": positive,
                        },
                        "reward_components": {"train_shaped_v18": 1.0 if positive else -1.0},
                        "tool_executions": [
                            {"name": write, "ok": True, "is_write": True}
                        ] if positive else [],
                        "stopped_reason": "completed",
                        "error": None,
                    }
                )
        hashes = {"verifier_v18.py": "a" * 64, "review.jsonl": "b" * 64}
        receipt = assess_v18_reward_calibration(rows, source_sha256=hashes)
        self.assertTrue(receipt["reward_calibrated"], receipt["blocking"])
        self.assertEqual(receipt["metrics"]["mixed_group_rate"], 1.0)

        rows[1]["verification"]["safety_gate_passed"] = True
        blocked = assess_v18_reward_calibration(rows, source_sha256=hashes)
        self.assertFalse(blocked["reward_calibrated"])
        self.assertFalse(blocked["checks"]["unsafe_false_negatives_zero"])

    def test_v18_environment_is_explicit_fresh_and_uses_v18_reward(self) -> None:
        source = self.isolated("cancel_unshipped")
        metadata = {key: value for key, value in source.metadata.items() if key != "scripted_user_replies"}
        metadata.update(
            {
                "required_tool_sequence_v18": [
                    "verify_customer_identity", "list_orders", "get_order", "cancel_order"
                ],
                "authorized_write_order_ids_v18": [source.metadata["order_id"]],
                "terminal_v18": {
                    "decision": "ORDER_CANCELLED",
                    "action_taken": True,
                    "action_tool": "cancel_order",
                    "action_id_field": "order_id",
                },
            }
        )
        task = replace(
            source,
            split="train",
            required_tools=("verify_customer_identity", "list_orders", "get_order", "cancel_order"),
            metadata=metadata,
        )
        payload = task.to_dict(relative_database=str(task.database_path))
        task_path = Path(self.temp.name) / "task-v18.json"
        task_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        environment = EcommerceCustomerServiceEnvV18(
            task_path=task_path,
            reward_mode="strict_binary_v18",
        )
        environment.reset()
        environment._tools.observe_user_message("我确认取消订单，请执行。")
        identity = environment.verify_customer_identity(**task.metadata["identity"])
        customer_id = identity["customer_id"]
        environment.list_orders(customer_id)
        environment.get_order(customer_id, task.metadata["order_id"])
        environment.cancel_order(customer_id, task.metadata["order_id"], True)
        amount = int(task.metadata["paid_amount_cents"]) / 100
        final = json.dumps(
            {
                "decision": "ORDER_CANCELLED",
                "action_taken": True,
                "action_id": task.metadata["order_id"],
                "message": f"订单已取消，{amount:.2f} 元将原路退款。",
            },
            ensure_ascii=False,
        )
        self.assertEqual(environment._evaluate(final), 1.0)
        environment.reset()
        self.assertEqual(environment._evaluate(final), 0.0)

    def test_v18_train_dev_builder_uses_fresh_namespaces_and_no_test_rows(self) -> None:
        output = Path(self.temp.name) / "v18-data"
        summary = build_v18_train_dev_dataset(
            output,
            train_per_scenario=2,
            dev_per_scenario=1,
            seed=1811,
        )
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["tasks"], len(V18_SCENARIOS) * 3)
        self.assertEqual(summary["split_counts"], {"train": 60, "dev": 30})
        self.assertEqual(summary["test_rows"], 0)
        self.assertFalse(summary["test_v2_created"])
        self.assertFalse(summary["v17_test_reused"])
        self.assertEqual(summary["exact_prompt_overlap"], 0)
        self.assertFalse(
            set(summary["prompt_families"]["train"])
            & set(summary["prompt_families"]["dev"])
        )
        rows = [
            json.loads(line)
            for line in (output / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(all(row["task_id"].startswith("v18_") for row in rows))
        self.assertTrue(all("scripted_user_replies" not in row["metadata"] for row in rows))
        self.assertEqual(validate_v18_train_dev_dataset(output)["tasks"], len(rows))

        sft_summary = build_reference_sft_v18(output / "tasks.jsonl", Path(self.temp.name) / "sft")
        self.assertEqual(sft_summary["tasks"], {"train": 60, "dev": 30})
        self.assertGreater(sft_summary["training_rows"]["train"], 60)
        self.assertGreater(sft_summary["training_rows"]["dev"], 30)
        self.assertTrue(sft_summary["all_reference_strict_pass"])
        self.assertEqual(sft_summary["test_tasks"], 0)

        first_database = output / rows[0]["database"]
        first_database.write_bytes(first_database.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "database_hash_mismatch"):
            validate_v18_train_dev_dataset(output)

    def test_v18_sft_preflight_pins_hashes_provenance_and_one_epoch_steps(self) -> None:
        task_output = Path(self.temp.name) / "v18-data"
        build_v18_train_dev_dataset(task_output, train_per_scenario=1, dev_per_scenario=1, seed=1817)
        sft_output = Path(self.temp.name) / "sft"
        sft_summary = build_reference_sft_v18(task_output / "tasks.jsonl", sft_output)
        config = {
            "profile_version": "ecommerce_cs_sft_v18_48gb_r1",
            "gpu": {"minimum_reported_memory_gib": 45},
            "model": {"id": "local/test", "revision": "a" * 40},
            "training": {"effective_batch_size": 16},
            "phases": {
                "smoke": {"max_steps": 1},
                "full": {
                    "max_steps": (sft_summary["training_rows"]["train"] + 15) // 16,
                },
            },
            "data": {},
        }
        frozen = {
            "reference_summary": sft_output / "summary.json",
            "task_summary": task_output / "summary.json",
            "task_manifest": task_output / "tasks.jsonl",
            "train": sft_output / "train.jsonl",
            "dev": sft_output / "dev.jsonl",
            "task_generator": Path(__file__).resolve(),
            "reference_generator": Path(__file__).resolve(),
        }
        for label, path in frozen.items():
            config["data"][label] = {"path": str(path), "sha256": sha256_file(path)}
        config["data"]["task_manifest"]["rows"] = 60
        config["data"]["train"]["rows"] = sft_summary["training_rows"]["train"]
        config["data"]["dev"]["rows"] = sft_summary["training_rows"]["dev"]
        config_path = Path(self.temp.name) / "profile.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        receipt = validate_v18_sft_bundle(config_path, project_root=Path(self.temp.name))
        self.assertTrue(receipt["data_ready"])
        self.assertEqual(receipt["test_rows"], 0)
        self.assertEqual(receipt["splits"]["train"]["tasks"], len(V18_SCENARIOS))
        self.assertEqual(receipt["full_max_steps"], config["phases"]["full"]["max_steps"])

        train_path = frozen["train"]
        train_path.write_bytes(train_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "frozen_input_hash_mismatch:train"):
            validate_v18_sft_bundle(config_path, project_root=Path(self.temp.name))


if __name__ == "__main__":
    unittest.main()
