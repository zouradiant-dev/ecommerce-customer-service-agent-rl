"""CPU-only minimal pairs for explicit v13; no model answers or GPU use."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.communication_facts import grounded_communication_failures
from ecommerce_customer_service.communication_facts_v13 import grounded_communication_failures_v13
from ecommerce_customer_service.database import create_database
from ecommerce_customer_service.data import load_jsonl, write_jsonl
from ecommerce_customer_service.local_hf import sha256_file
from ecommerce_customer_service.models import StateAssertion, TaskSpec
from ecommerce_customer_service.rescoring import rescore_rollouts
from ecommerce_customer_service.verifier import (
    LEGACY_VERIFIER_VERSION, VERIFIER_VERSION, V13_VERIFIER_VERSION, verify_task,
)


def fixture():
    task = TaskSpec(task_id="new-null-example", split="train", scenario="cancel_without_captured_payment",
                    user_id="cust_unseen", user_prompt="确认取消", reference_time="2026-01-01T00:00:00+08:00",
                    database_path=Path("unused.sqlite"), assertions=(),
                    required_intents=("order_cancelled", "payment_not_captured", "refund_not_needed"),
                    required_tools=("get_order", "cancel_order"), allowed_write_tools=("cancel_order",),
                    metadata={"order_id": "ord_unseen"})
    args = {"customer_id": task.user_id, "order_id": "ord_unseen"}
    calls = [
        {"name": "get_order", "ok": True, "is_write": False, "policy_violation": None,
         "arguments": dict(args), "output": {
             "order": {"id": "ord_unseen", "customer_id": task.user_id, "status": "paid", "total_paid_cents": 6800},
             "payments": [{"id": "pay_unseen", "order_id": "ord_unseen", "status": "authorized", "amount_cents": 0}],
             "refunds": [], "returns": []}},
        {"name": "cancel_order", "ok": True, "is_write": True, "policy_violation": None,
         "arguments": {**args, "confirmed": True}, "output": {"cancelled": True, "refund_id": None}},
    ]
    return task, calls


class NullAndAmountTest(unittest.TestCase):
    def setUp(self):
        self.task, self.calls = fixture()

    def check(self, text, calls=None, task=None):
        return grounded_communication_failures_v13(
            self.task if task is None else task, text, self.calls if calls is None else calls)

    def test_null_is_accepted_only_with_evidenced_absent_refund(self):
        for text in ("refund_id 为 null。", "退款ID：NULL。", '{"refund_id":null}', "**退款编号** = null"):
            with self.subTest(text=text):
                self.assertEqual(grounded_communication_failures(self.task, text, self.calls),
                                 ("fact_mismatch:refund_id:null",))
                self.assertFalse(self.check(text))

    def test_failed_missing_or_unexecuted_cancellation_does_not_prove_null(self):
        variants = [[], self.calls[:1], self.calls[1:]]
        for modification in ("failed", "not_cancelled", "missing_field", "wrong_customer", "wrong_order"):
            calls = deepcopy(self.calls)
            if modification == "failed":
                calls[1]["ok"] = False
            elif modification == "not_cancelled":
                calls[1]["output"]["cancelled"] = False
            elif modification == "missing_field":
                del calls[1]["output"]["refund_id"]
            else:
                calls[1]["arguments"]["customer_id" if modification == "wrong_customer" else "order_id"] = "foreign"
            variants.append(calls)
        for calls in variants:
            with self.subTest(calls=calls):
                self.assertIn("fact_mismatch:refund_id:null", self.check("refund_id 为 null", calls))

    def test_incomplete_or_foreign_read_is_not_absence_evidence(self):
        for modification in ("failed", "customer", "order", "payment_owner", "payments_missing", "refunds_missing"):
            calls = deepcopy(self.calls)
            if modification == "failed":
                calls[0]["ok"] = False
            elif modification in {"customer", "order"}:
                calls[0]["output"]["order"]["customer_id" if modification == "customer" else "id"] = "foreign"
            elif modification == "payment_owner":
                calls[0]["output"]["payments"][0]["order_id"] = "foreign"
            else:
                del calls[0]["output"][modification.removesuffix("_missing")]
            with self.subTest(modification=modification):
                self.assertIn("fact_mismatch:refund_id:null", self.check("refund_id 为 null", calls))

    def test_captured_or_refunded_payment_cannot_be_called_never_captured(self):
        for status in ("captured", "refunded"):
            calls = deepcopy(self.calls)
            calls[0]["output"]["payments"][0].update(status=status, amount_cents=6800)
            self.assertIn("fact_mismatch:refund_id:null", self.check("refund_id 为 null", calls))
            self.assertIn("fact_mismatch:captured_amount:0元", self.check("实际扣款金额为0元。", calls))

    def test_nonzero_authorization_is_not_actual_capture(self):
        calls = deepcopy(self.calls)
        calls[0]["output"]["payments"][0]["amount_cents"] = 6800
        self.assertFalse(self.check("refund_id 为 null；订单总金额为68元；实际扣款金额为0元。", calls))

    def test_real_refund_id_always_blocks_null_even_after_a_null_cancel(self):
        for location in ("history", "later_write", "cancel"):
            calls = deepcopy(self.calls)
            if location == "history":
                calls[0]["output"]["refunds"] = [{"id": "refund_old", "order_id": "ord_unseen", "amount_cents": 100}]
            elif location == "later_write":
                calls.append({"name": "create_refund", "ok": True,
                              "arguments": {"customer_id": self.task.user_id, "order_id": "ord_unseen"},
                              "output": {"refund_id": "refund_new", "refunded": True, "amount_cents": 100}})
            else:
                calls[1]["output"]["refund_id"] = "refund_cancel_ord_unseen"
            self.assertIn("fact_mismatch:refund_id:null", self.check("refund_id 为 null", calls))

    def test_null_exemption_does_not_waive_wrong_or_other_typed_ids(self):
        for text in ("refund_id 为 null_fake", "refund_id 为 none", "退款ID：refund_fake_999",
                     "退货ID：null", "工单号：null", "refund_id 为 null；退款ID：refund_fake_999"):
            with self.subTest(text=text):
                self.assertTrue(self.check(text))

    def test_correct_order_total_and_actual_capture_are_separate(self):
        for text in ("订单总金额为68元，实际扣款金额为0元。", "订单金额：6800分（68.00元）。",
                     "订单 ord_unseen 的总金额：68元。", "订单总价 = ￥68.00 CNY。"):
            with self.subTest(text=text):
                self.assertFalse(self.check(text))
        for text, field in (("订单总金额为0元（仅预授权）。", "order_total"),
                            ("订单金额为6800元。", "order_total"), ("订单总金额68元（100分）。", "order_total"),
                            ("订单总金额为-68元。", "order_total"),
                            ("实际扣款金额为68元。", "captured_amount"), ("实扣金额为6800分。", "captured_amount")):
            with self.subTest(text=text):
                self.assertTrue(any(item.startswith(f"fact_mismatch:{field}:") for item in self.check(text)))

    def test_facts_are_not_hardcoded_to_sample_customer_or_amount(self):
        task = replace(self.task, user_id="cust_another", metadata={"order_id": "ord_another"})
        calls = deepcopy(self.calls)
        for call in calls:
            call["arguments"].update(customer_id="cust_another", order_id="ord_another")
        calls[0]["output"]["order"].update(id="ord_another", customer_id="cust_another", total_paid_cents=12005)
        calls[0]["output"]["payments"][0].update(order_id="ord_another", status="captured", amount_cents=7000)
        calls[1]["output"]["refund_id"] = "refund_another"
        self.assertFalse(self.check("订单总金额为120.05元，实际扣款金额为70元。", calls, task))
        self.assertTrue(self.check("订单总金额为68元，实际扣款金额为0元。", calls, task))

    def test_unknown_invalid_fields_are_not_inferred_as_zero(self):
        for value in (None, True, "6800", 6800.0, -1):
            calls = deepcopy(self.calls)
            calls[0]["output"]["order"]["total_paid_cents"] = value
            self.assertFalse(self.check("订单总金额为999元。", calls))  # unknown, not a verified assertion
        calls = deepcopy(self.calls)
        calls[0]["output"]["payments"][0].update(status="unknown")
        self.assertIn("fact_mismatch:refund_id:null", self.check("refund_id 为 null", calls))

    def test_latest_owned_read_controls_order_total(self):
        calls = deepcopy(self.calls)
        calls.append(deepcopy(calls[0]))
        calls[-1]["output"]["order"]["total_paid_cents"] = 7500
        self.assertFalse(self.check("订单总金额为75元。", calls))
        self.assertTrue(self.check("订单总金额为68元。", calls))
        calls[-1]["arguments"]["order_id"] = "foreign"
        self.assertFalse(self.check("订单总金额为68元。", calls))

    def test_unlabelled_or_conditional_money_is_not_a_general_semantic_judge(self):
        self.assertFalse(self.check("例如如果订单总金额为0元，不代表没有商品。"))
        self.assertFalse(self.check("并非订单总金额不是68元。"))
        self.assertFalse(self.check("这里是例子：999元。"))


class VersionedVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v13-scoped-test-")
        self.directory = Path(self.temp.name)
        task, self.calls = fixture()
        database = self.directory / "state.sqlite"
        create_database(database)
        self.task = replace(task, database_path=database)
        self.response = "订单 ord_unseen 已取消。尚未实际扣款，无需退款；refund_id 为 null。"

    def tearDown(self):
        self.temp.cleanup()

    def test_v11_v12_reproducible_and_default_stays_v12(self):
        self.assertEqual(VERIFIER_VERSION, "deterministic_semantic_v12")
        self.assertEqual(V13_VERIFIER_VERSION, "deterministic_semantic_v13")
        self.assertTrue(verify_task(self.task, self.response, self.calls, verifier_version=LEGACY_VERIFIER_VERSION).passed)
        old = verify_task(self.task, self.response, self.calls)
        self.assertFalse(old.passed)
        self.assertIn("fact_mismatch:refund_id:null", old.failures)
        self.assertTrue(verify_task(self.task, self.response, self.calls, verifier_version=V13_VERIFIER_VERSION).passed)

    def test_semantics_cannot_waive_missing_execution_failed_state_or_policy(self):
        self.assertFalse(verify_task(self.task, self.response, self.calls[:1], verifier_version=V13_VERIFIER_VERSION).passed)
        missing_state = replace(self.task, assertions=(StateAssertion("orders", {"id": "ord_unseen"}, count=1),))
        self.assertFalse(verify_task(missing_state, self.response, self.calls, verifier_version=V13_VERIFIER_VERSION).passed)
        calls = deepcopy(self.calls)
        calls[1]["policy_violation"] = "missing_confirmation"
        self.assertFalse(verify_task(self.task, self.response, calls, verifier_version=V13_VERIFIER_VERSION).passed)

    def test_wrong_amount_remains_zero_even_with_canonical_intents(self):
        result = verify_task(self.task, self.response + "订单总金额为0元。", self.calls, verifier_version=V13_VERIFIER_VERSION)
        self.assertFalse(result.passed)
        self.assertIn("fact_mismatch:order_total:0元", result.failures)

    def test_explicit_rescoring_keeps_input_and_retained_runtime_failures(self):
        task_file = self.directory / "task.json"
        task_file.write_text(json.dumps(self.task.to_dict(), ensure_ascii=False), encoding="utf-8")
        manifest = self.directory / "tasks.jsonl"
        write_jsonl(manifest, [{"task_id": self.task.task_id, "split": "train", "task_file": "task.json"}])
        old = json.loads(json.dumps(verify_task(self.task, self.response, self.calls).to_dict()))
        source = self.directory / "source.jsonl"
        row = {"task_id": self.task.task_id, "split": "train", "final_response": self.response,
               "tool_executions": self.calls, "verifier": old, "reward": 0.0, "verifier_version": VERIFIER_VERSION}
        write_jsonl(source, [row])
        original_hash = sha256_file(source)
        default = rescore_rollouts(source, manifest, self.directory / "default.jsonl")
        self.assertEqual(default["passed_after"], 0)
        new = rescore_rollouts(source, manifest, self.directory / "v13.jsonl", verifier_version=V13_VERIFIER_VERSION)
        self.assertEqual(new["passed_after"], 1)
        self.assertEqual(load_jsonl(self.directory / "v13.jsonl")[0]["rescore_metadata"]["source_verifier_version"], VERIFIER_VERSION)
        self.assertEqual(sha256_file(source), original_hash)
        row["verifier"]["failures"].append("runtime_failure:execution_incomplete")
        write_jsonl(source, [row])
        rescore_rollouts(source, manifest, self.directory / "runtime.jsonl", verifier_version=V13_VERIFIER_VERSION)
        retained = load_jsonl(self.directory / "runtime.jsonl")[0]
        self.assertEqual(retained["reward"], 0.0)
        self.assertIn("runtime_failure:execution_incomplete", retained["verifier"]["failures"])

    def test_unknown_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported_verifier_version"):
            verify_task(self.task, self.response, self.calls, verifier_version="fake_v13")


if __name__ == "__main__":
    unittest.main()
