from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_customer_service.communication import INTENT_RULES, TEMPLATE_REQUIRED_INTENTS, missing_communication_intents
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.simulator import OpenAICompatibleAgent, USER_REPLY_PROTOCOL_VERSION
from ecommerce_customer_service.verifier import LEGACY_VERIFIER_VERSION, VERIFIER_VERSION, policy_failures


class CommunicationV9Test(unittest.TestCase):
    def test_versioned_protocol(self):
        self.assertEqual(LEGACY_VERIFIER_VERSION, "deterministic_semantic_v11")
        self.assertEqual(USER_REPLY_PROTOCOL_VERSION, "scripted_user_replies_v2")

    def test_pilot_synonyms_are_not_missing_intents(self):
        # Recorded pilot wording, not rewritten model output.
        responses = {
            "cancel_already_cancelled": "该订单已成功取消并完成退款，无需重复操作。",
            "refund_before_delivery": "根据退款政策，当前订单尚未签收（物流状态为运输中），因此不允许创建售后退货退款。",
            "refund_quantity_exceeds": "根据订单详情，实际购买数量是 **1件**。因此，您不能按2件数量申请退款。",
        }
        for template, response in responses.items():
            with self.subTest(template=template):
                self.assertEqual(missing_communication_intents(TEMPLATE_REQUIRED_INTENTS[template], response), [])

    def test_general_synonyms_and_formatting(self):
        responses = {
            "cancel_already_cancelled": (
                "您的订单已经被取消并且退款成功。您无需再次操作。",
                "订单已被取消，已经完成退款，不用再次处理。",
            ),
            "refund_quantity_exceeds": (
                "实际购买数量为一件，不能按两件退款。",
                "您只买了１件，无法申请２件退款。",
                "购买数量是1件，不允许按两件退款。",
            ),
            "refund_before_delivery": ("订单尚未签收，不能创建退货退款。",),
        }
        for template, values in responses.items():
            for response in values:
                with self.subTest(template=template, response=response):
                    self.assertEqual(missing_communication_intents(TEMPLATE_REQUIRED_INTENTS[template], response), [])

    def test_new_synonyms_do_not_accept_negation_future_or_wrong_quantity(self):
        cases = {
            "refund_already_completed": (
                "尚未完成退款。", "并非退款已完成。", "我将完成退款。",
                "如果退款成功，再通知您。", "已有退款记录，但退款状态为处理中。",
                "退款已完成。实际上退款未完成。",
                "并非退款成功。", "不是已经完成退款。",
            ),
            "purchased_quantity_one": (
                "实际购买数量是11件。", "订单数量为10件。", "不是购买数量是1件。",
                "不仅购买了1件。", "实际购买数量是1件。并非实际购买数量是1件。",
            ),
            "return_before_delivery_blocked": (
                "订单尚未签收，但允许创建售后退货退款。",
                "并非不允许创建售后退货退款。",
                "不能创建售后退货退款。但现在可以创建退货退款。",
            ),
        }
        for intent, responses in cases.items():
            for response in responses:
                with self.subTest(intent=intent, response=response):
                    self.assertFalse(INTENT_RULES[intent].matches(response))

    def test_actual_pilot_address_failure_still_fails(self):
        response = (
            "该订单已发货，地址修改可能不支持。请确认是否需要取消订单后再进行地址修改？"
            "如果您确认要取消订单并重新下单，请明确告知。否则，我将告知您当前地址无法修改的原因。"
        )
        self.assertEqual(missing_communication_intents(("shipped_address_change_blocked",), response),
                         ["shipped_address_change_blocked"])


class ScriptedRepliesV2Test(unittest.TestCase):
    def test_explicit_identity_request_variants(self):
        for content in (
            "请您提供一下姓名和手机号后四位以便我进行身份验证，谢谢。",
            "您好！请确认您的姓名和手机号后四位，以便我们进行身份验证。",
            "请提供您的**姓名**和**手机号后四位**。",
            "麻烦您告知姓名及手机号后四位。",
            "需要您提供姓名和手机号码后四位。",
            "请告诉我您的名字和手机号后四位。",
        ):
            with self.subTest(content=content):
                self.assertTrue(OpenAICompatibleAgent._assistant_requests_reply("identity", content))

    def test_identity_is_not_injected_on_mentions_or_unrelated_confirmation(self):
        for content in (
            "已记录您的姓名和手机号后四位。",
            "好的，请稍候。",
            "请不要提供姓名和手机号后四位。",
            "无需提供姓名或手机号后四位。",
            "请确认是否继续退款。姓名已核验。",
            "请提供订单号，姓名和手机号已经保存。",
            "请确认取消订单，手机号后四位不需要再次提供。",
            "请确认不需要提供姓名和手机号后四位。",
        ):
            with self.subTest(content=content):
                self.assertFalse(OpenAICompatibleAgent._assistant_requests_reply("identity", content))

    def test_identity_request_is_not_operation_confirmation(self):
        for content in ("请确认您的姓名和手机号后四位。", "请确认您的姓名和手机号后四位，以便办理退款。"):
            with self.subTest(content=content):
                self.assertTrue(OpenAICompatibleAgent._assistant_requests_reply("identity", content))
                self.assertFalse(OpenAICompatibleAgent._assistant_requests_reply("confirmation", content))
        self.assertFalse(OpenAICompatibleAgent._assistant_requests_reply("confirmation", "请确认不需要提供姓名。"))
        both = "请确认您的姓名和手机号后四位，同时请确认是否退款。"
        self.assertTrue(OpenAICompatibleAgent._assistant_requests_reply("identity", both))
        self.assertTrue(OpenAICompatibleAgent._assistant_requests_reply("confirmation", both))


class HistoricalResultEvidenceTest(unittest.TestCase):
    def setUp(self):
        # policy_failures only reads the recorded trajectory, not this dummy DB.
        self.task = TaskSpec(
            task_id="history", split="dev", scenario="cancel_already_cancelled",
            user_id="cust_target", user_prompt="请核对是否已取消，不要重复退款。",
            reference_time="2026-01-15T12:00:00+00:00", database_path=Path("unused.sqlite"), assertions=(),
            required_tools=("verify_customer_identity", "get_order"), metadata={"order_id": "ord_target"},
        )
        self.executions = [
            {"name": "verify_customer_identity", "ok": True, "is_write": False},
            {"name": "get_order", "ok": True, "is_write": False, "output": {
                "order": {"id": "ord_target", "customer_id": "cust_target", "status": "cancelled"},
                "refunds": [{"order_id": "ord_target", "status": "completed"}],
            }},
        ]

    def test_read_can_support_existing_cancel_and_refund(self):
        for response in (
            "订单已取消并且退款成功。您无需再次操作。",
            "您的订单已经被取消并且退款成功。您无需再次操作。",
        ):
            with self.subTest(response=response):
                self.assertEqual(policy_failures(self.task, response, self.executions), ())

    def test_no_read_or_failed_read_cannot_support_refund_claim(self):
        self.assertIn("unsupported_success_claim:create_refund", policy_failures(self.task, "退款成功。", []))
        failed = deepcopy(self.executions)
        failed[1]["ok"] = False
        self.assertIn("unsupported_success_claim:create_refund", policy_failures(self.task, "退款成功。", failed))

    def test_evidence_must_match_order_customer_and_completed_status(self):
        for field, value in (("order_id", "another"), ("customer_id", "another"), ("refund_order_id", "another"),
                             ("refund_status", "pending")):
            rows = deepcopy(self.executions)
            output = rows[1]["output"]
            if field == "order_id":
                output["order"]["id"] = value
            elif field == "customer_id":
                output["order"]["customer_id"] = value
            elif field == "refund_order_id":
                output["refunds"][0]["order_id"] = value
            else:
                output["refunds"][0]["status"] = value
            with self.subTest(field=field):
                self.assertIn("unsupported_success_claim:create_refund", policy_failures(self.task, "退款成功。", rows))

    def test_existing_result_cannot_excuse_claim_of_a_new_action(self):
        for response in (
            "我已为您退款成功。", "我帮您退款成功。", "本次退款成功。", "刚刚退款成功。",
            "已帮您处理，退款成功。", "已创建退货并退款成功。",
        ):
            with self.subTest(response=response):
                self.assertIn("unsupported_success_claim:create_refund", policy_failures(self.task, response, self.executions))
        self.assertIn("unsupported_success_claim:cancel_order",
                      policy_failures(self.task, "本次取消成功。", self.executions))

    def test_other_success_claims_still_require_writes(self):
        for response, name in (("地址已更新。", "update_shipping_address"), ("已创建人工工单。", "escalate_to_human")):
            with self.subTest(name=name):
                self.assertIn(f"unsupported_success_claim:{name}", policy_failures(self.task, response, self.executions))


if __name__ == "__main__":
    unittest.main()
