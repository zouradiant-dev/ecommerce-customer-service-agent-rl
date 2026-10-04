"""CPU regressions for corrected semantics, grounded facts and strict JSON."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.communication import INTENT_RULES, INTENT_RULES_V11
from ecommerce_customer_service.communication_facts import grounded_communication_failures
from ecommerce_customer_service.database import create_database
from ecommerce_customer_service.local_hf import ToolCallParseError, parse_qwen_response
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.online_env import JsonToolObservation
from ecommerce_customer_service.verifier import (
    LEGACY_VERIFIER_VERSION, VERIFIER_VERSION, verify_task,
)


def fixture():
    task = TaskSpec(task_id="unseen-example", split="train", scenario="damaged_item_refund",
        user_id="cust_new", user_prompt="请退货", reference_time="2026-01-01T00:00:00+08:00",
        database_path=Path("unused.sqlite"), assertions=(), metadata={"order_id": "ord_new"})
    calls = [
        {"name": "get_order", "ok": True, "arguments": {"customer_id": "cust_new", "order_id": "ord_new"},
         "output": {"order": {"id": "ord_new", "customer_id": "cust_new", "status": "delivered"},
            "payments": [{"order_id": "ord_new", "method": "alipay"}], "returns": [], "refunds": []}},
        {"name": "create_refund", "ok": True, "arguments": {"customer_id": "cust_new", "order_id": "ord_new"},
         "output": {"return_id": "return_new_1", "refund_id": "refund_new_1", "refunded": True, "amount_cents": 12345}},
        {"name": "escalate_to_human", "ok": True, "arguments": {},
         "output": {"escalated": True, "ticket_id": "ticket_cust_new_1"}},
    ]
    return task, calls


class SemanticsTest(unittest.TestCase):
    def test_versions_are_distinct_and_old_synonym_misses_reproducible(self):
        self.assertEqual(LEGACY_VERIFIER_VERSION, "deterministic_semantic_v11")
        self.assertEqual(VERIFIER_VERSION, "deterministic_semantic_v12")
        for intent, text in (("refund_to_original_payment", "退款方式：支付宝（原支付渠道）。"),
                             ("human_escalation_created", "已为您升级至人工支持。")):
            self.assertFalse(INTENT_RULES_V11[intent].matches(text))
            self.assertTrue(INTENT_RULES[intent].matches(text))

    def test_refund_destination_positive_and_negative_minimal_pairs(self):
        positive = ("退款方式：支付宝（原支付渠道）。", "款项将按原支付路径退回。",
                    "退款已通过原付款渠道返回。", "退款金额123.45元将原路退回。")
        negative = ("退款未按原支付路径退回。", "退款不能原路退回。", "不是原路退款。",
                    "如果原支付渠道允许，退款将退回。", "原路退款失败。",
                    "退款方式：其他账户，不是原支付渠道。", "退款已原路退回，但改为其他账户。",
                    "已创建退款，金额123.45元。", "退款方式：支付宝。")
        for text in positive + negative:
            with self.subTest(text=text):
                self.assertEqual(INTENT_RULES["refund_to_original_payment"].matches(text), text in positive)

    def test_escalation_is_completed_not_negated_future_or_failed(self):
        positive = ("已为您升级至人工支持。", "我已成功转交至人工客服。", "已创建人工工单。")
        negative = ("尚未为您升级至人工支持。", "不是已为您升级至人工支持。",
                    "将为您升级至人工支持。", "准备升级至人工支持。", "若升级至人工支持，再通知您。",
                    "已升级至人工支持失败。", "尚未创建人工工单。", "已转人工。人工工单尚未创建。",
                    "如果已为您升级至人工支持，请稍候。", "将交由人工处理。", "若已创建人工工单，请等待。")
        for text in positive + negative:
            with self.subTest(text=text):
                self.assertEqual(INTENT_RULES["human_escalation_created"].matches(text), text in positive)


class GroundedFactsTest(unittest.TestCase):
    def setUp(self):
        self.task, self.calls = fixture()

    def failures(self, text, calls=None):
        return grounded_communication_failures(self.task, text, self.calls if calls is None else calls)

    def test_correct_typed_ids_and_units(self):
        self.assertFalse(self.failures("退款ID：refund_new_1；退货项 ID：return_new_1。工单号：ticket_cust_new_1。"
            "退款金额：123.45元（12345分）。退款状态：已完成。原路退回至支付宝。"))
        self.assertFalse(self.failures('退款金额：12345分（123.45元）。'))
        self.assertFalse(self.failures('{"refund_id":"refund_new_1", "return_id":"return_new_1"}'))

    def test_wrong_id_type_spelling_and_suffix(self):
        for text, field in (("**退款ID**：return_new_1", "refund_id"), ("退款ID：refund_new_11", "refund_id"),
                            ("退货项ID：refund_new_1", "return_id"), ("工单号：ticket_cust_other_1", "ticket_id")):
            with self.subTest(text=text):
                self.assertTrue(any(f"fact_mismatch:{field}:" in value for value in self.failures(text)))

    def test_failed_or_foreign_tools_cannot_support_an_id(self):
        for modification in ("failed", "customer", "order"):
            calls = deepcopy(self.calls)
            if modification == "failed":
                calls[1]["ok"] = False
            else:
                calls[1]["arguments"][f"{modification}_id"] = "foreign"
            self.assertTrue(self.failures("退款ID：refund_new_1", calls))
        self.assertTrue(self.failures("退款ID：refund_new_1", []))

    def test_wrong_amount_units_destination_and_state(self):
        for text in ("退款金额：12345元", "退款金额：123.45元（100分）", "退款金额：99元",
                     "退款方式：微信（原支付渠道）", "原路退回至微信", "退款状态：未退款"):
            with self.subTest(text=text):
                self.assertTrue(self.failures(text))

    def test_latest_order_status_after_write_overrides_old_read(self):
        calls = deepcopy(self.calls)
        calls.append({"name": "cancel_order", "ok": True,
            "arguments": {"customer_id": "cust_new", "order_id": "ord_new"},
            "output": {"cancelled": True, "order_id": "ord_new", "refund_id": "refund_cancel_ord_new"}})
        self.assertFalse(self.failures("订单状态：已取消。", calls))
        self.assertTrue(self.failures("订单状态：已签收。", calls))

    def test_cancellation_refund_amount_comes_from_captured_payment_not_metadata(self):
        calls = deepcopy(self.calls[:1])
        calls[0]["output"]["payments"][0].update(id="pay_new", status="captured", amount_cents=12345)
        calls.append({"name": "cancel_order", "ok": True,
            "arguments": {"customer_id": "cust_new", "order_id": "ord_new"},
            "output": {"cancelled": True, "refund_id": "refund_cancel_ord_new"}})
        self.assertFalse(self.failures("退款金额：123.45元。", calls))
        self.assertTrue(self.failures("原付款123.45元，退款金额：999元。", calls))

    def test_historical_owned_refund_ids_are_valid(self):
        calls = deepcopy(self.calls[:1])
        calls[0]["output"]["refunds"] = [{"id": "refund_old", "order_id": "ord_new", "status": "completed", "amount_cents": 500}]
        self.assertFalse(self.failures("退款ID：refund_old，退款金额：5元。", calls))
        calls[0]["output"]["refunds"][0]["order_id"] = "ord_other"
        self.assertTrue(self.failures("退款ID：refund_old", calls))

    def test_semantic_fix_cannot_waive_no_execution_or_wrong_id(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite"
            create_database(database)
            task = replace(self.task, database_path=database, required_intents=("refund_to_original_payment",),
                           required_tools=("create_refund",), allowed_write_tools=("create_refund",))
            result = verify_task(task, "款项将按原支付路径退回。", [])
            self.assertTrue(result.communication_passed)
            self.assertFalse(result.passed)
            response = "退款ID：return_new_1。退款原路退回。"
            self.assertFalse(verify_task(task, response, self.calls[:2]).passed)
            self.assertTrue(verify_task(task, response, self.calls[:2], verifier_version=LEGACY_VERIFIER_VERSION).passed)
            self.assertIn("unsupported_success_claim:escalate_to_human",
                          verify_task(task, "已为您升级至人工支持，退款将原路退回。", []).failures)


class ObservationTest(unittest.TestCase):
    def test_trl_str_result_is_strict_json_without_changing_dictionary_access(self):
        result = JsonToolObservation({"ok": True, "confirmed": False, "value": None, "nested": {"verified": True}, "text": "中文"})
        self.assertTrue(result["ok"])
        self.assertEqual(json.loads(str(result)), dict(result))
        self.assertNotIn("True", str(result))
        self.assertIn("true", str(result))
        self.assertIn("null", str(result))

    def test_nonfinite_output_is_not_silently_serialized(self):
        with self.assertRaises(ValueError):
            str(JsonToolObservation({"number": float("nan")}))

    def test_invalid_model_json_is_still_rejected(self):
        with self.assertRaises(ToolCallParseError):
            parse_qwen_response('<tool_call>{"name":"create_refund","arguments":{"confirmed":True}}</tool_call>')


if __name__ == "__main__":
    unittest.main()
