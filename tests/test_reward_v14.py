"""Synthetic semantic controls, not new model samples or reward signoff."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.communication import INTENT_RULES
from ecommerce_customer_service.communication_v14 import operation_stopped_v14
from ecommerce_customer_service.database import create_database
from ecommerce_customer_service.models import StateAssertion, TaskSpec
from ecommerce_customer_service.verifier import VERIFIER_VERSION, V13_VERIFIER_VERSION, verify_task
from ecommerce_customer_service.verifier_v14 import V14_VERIFIER_VERSION, verify_task_v14


class StopSemanticsTest(unittest.TestCase):
    def test_current_non_execution_synonyms(self):
        for text in (
            "在您最后确认之前，不会进行任何退款操作。",
            "目前我不会进行任何退款操作。",
            "我尚未进行退款操作。",
            "我没有进行任何退款操作。",
            "本次未办理退款申请。",
            "我暂时不办理退款。",
            "在您明确授权前，我不会办理退款。",
            "目前 **不会进行任何退款操作**。",
            "尚未执行任何退款操作。",
            "我已暂停退款流程。",
            "我不会执行任何退款操作。",
            "目前我不会进行退款操作。只有您明确确认后我才会办理退款。",
            "我尚未进行退款操作。我们正在退款政策查询。",
        ):
            with self.subTest(text=text):
                self.assertTrue(operation_stopped_v14(text))

    def test_query_only_or_authorization_promise_is_not_a_stop(self):
        for text in (
            "您需要最后确认后才执行退款操作。目前只是查询结果。",
            "我需要等待用户最后确认后再执行退款操作。",
            "退款政策允许退款。请您确认。",
            "我不会进行退款信息查询。",
            "我不会进行退款政策核对。",
            "我不会进行退款操作说明。",
        ):
            with self.subTest(text=text):
                self.assertFalse(operation_stopped_v14(text))

    def test_denial_quotes_hypotheticals_and_future_stop_are_not_credited(self):
        for text in (
            "并不是不会进行任何退款操作。",
            "我并非不会办理退款。",
            "我不能保证不会进行任何退款操作。",
            "我没有说我不会执行任何退款。",
            "我不一定不会进行任何退款操作。",
            "我未必不会执行退款。",
            "这不代表我不会进行任何退款操作。",
            "如果我不会进行任何退款操作，您可以联系人工。",
            "假设尚未进行退款操作。",
            "用户说若我不会执行退款可以再联系。",
            "即使我不会执行退款也可以投诉。",
            "例如，我不会进行任何退款操作。",
            '示例："不会进行任何退款操作"。',
            "规则写着“不会进行任何退款操作”。",
            "提示为「不会执行任何退款」。",
            "`不会进行任何退款操作` 是一个字符串。",
            "'不会进行任何退款操作' 是一个字符串。",
            '"不会进行任何退款操作',
            '"目前\n不会进行任何退款操作\n"',
            "```\n不会进行任何退款操作\n```",
            "明天我才不会进行任何退款操作。",
            "我计划稍后暂停退款操作。",
            "其他客服不会进行任何退款操作。",
            "用户不会进行退款操作。",
            "不会进行退款操作是一个示例。",
        ):
            with self.subTest(text=text):
                self.assertFalse(operation_stopped_v14(text))

    def test_contradictory_actual_or_unconditional_future_execution(self):
        for suffix in (
            "我已经进行了退款。", "我正在办理退款。", "本次已执行退款操作。",
            "退款操作已经完成。", "但是我会执行退款。", "我将进行退款。",
            "稍后我会办理退款。", "明天我会进行退款。", "现在就进行退款。",
            "已进行了退款。", "将办理退款。", "我准备进行退款。",
            "无需您确认我就进行退款。",
            "我已经退款。", "我会退款。", "我们将为您退款。",
        ):
            with self.subTest(suffix=suffix):
                self.assertFalse(operation_stopped_v14("我不会进行任何退款操作。" + suffix))


class OverlaySafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-v14-")
        database = Path(self.temp.name) / "state.sqlite"
        create_database(database)
        self.task = TaskSpec(task_id="synthetic_unseen", split="dev", scenario="negative_confirmation",
            user_id="cust_synthetic", user_prompt="先不要退款", reference_time="2026-01-01T00:00:00+00:00",
            database_path=database, assertions=(), required_intents=("operation_stopped",))
        self.text = "在您最后确认之前，不会进行任何退款操作。"

    def tearDown(self):
        self.temp.cleanup()

    def test_v13_and_defaults_remain_frozen_and_v14_is_explicit(self):
        rules_before = deepcopy(INTENT_RULES)
        self.assertEqual(VERIFIER_VERSION, "deterministic_semantic_v12")
        self.assertEqual(V13_VERIFIER_VERSION, "deterministic_semantic_v13")
        self.assertEqual(V14_VERIFIER_VERSION, "deterministic_semantic_v14")
        self.assertFalse(verify_task(self.task, self.text, [], verifier_version=V13_VERIFIER_VERSION).passed)
        self.assertTrue(verify_task_v14(self.task, self.text, []).passed)
        self.assertFalse(verify_task(self.task, self.text, [], verifier_version=V13_VERIFIER_VERSION).passed)
        self.assertEqual(INTENT_RULES, rules_before)
        with self.assertRaisesRegex(ValueError, "unsupported_verifier_version"):
            verify_task(self.task, self.text, [], verifier_version=V14_VERIFIER_VERSION)

    def test_state_and_required_tools_cannot_be_waived(self):
        invalid = replace(self.task, assertions=(StateAssertion("orders", {"id": "missing"}, count=1),))
        state = verify_task_v14(invalid, self.text, [])
        self.assertFalse(state.passed)
        self.assertFalse(state.state_passed)
        missing = verify_task_v14(replace(self.task, required_tools=("get_order",)), self.text, [])
        self.assertFalse(missing.passed)
        self.assertFalse(missing.required_tools_passed)
        self.assertIn("required_tool_missing_or_failed:get_order", missing.failures)

    def test_policy_and_factual_checks_cannot_be_waived(self):
        call = {"name": "get_order", "ok": False, "is_write": False,
                "output": {"error": "identity_verification_required"}, "arguments": {},
                "policy_violation": "identity_verification_required"}
        result = verify_task_v14(self.task, self.text, [call])
        self.assertFalse(result.passed)
        self.assertIn("policy_violation:identity_verification_required", result.failures)
        task = replace(self.task, required_values=("89.99",))
        result = verify_task_v14(task, self.text + "金额为0元。", [])
        self.assertFalse(result.passed)
        self.assertIn("missing_value:89.99", result.failures)

    def test_real_or_failed_refund_attempt_never_satisfies_stop(self):
        for ok in (True, False):
            for is_write in (True, False):
                call = {"name": "create_refund", "ok": ok, "is_write": is_write,
                        "arguments": {}, "output": {}, "policy_violation": None}
                with self.subTest(ok=ok, is_write=is_write):
                    result = verify_task_v14(self.task, self.text, [call])
                    self.assertFalse(result.passed)
                    self.assertFalse(result.communication_passed)
                    self.assertIn("operation_stopped_contradicted_by_tool:create_refund", result.failures)

    def test_other_intents_and_inputs_are_unchanged(self):
        for text in (self.text, "订单已取消。", "需要等待确认。"):
            task = replace(self.task, required_intents=("order_cancelled",))
            self.assertEqual(verify_task_v14(task, text, []),
                             verify_task(task, text, [], verifier_version=V13_VERIFIER_VERSION))

    def test_frozen_grounded_identifier_checks_remain_failures(self):
        text = self.text + "refund_id 为 refund_deadbeef99。"
        old = verify_task(self.task, text, [], verifier_version=V13_VERIFIER_VERSION)
        new = verify_task_v14(self.task, text, [])
        facts = {failure for failure in old.failures if failure.startswith("fact_mismatch:")}
        self.assertTrue(facts)
        self.assertTrue(facts.issubset(new.failures))
        self.assertFalse(new.passed)


if __name__ == "__main__":
    unittest.main()
