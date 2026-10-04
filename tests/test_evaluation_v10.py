from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_customer_service.communication import INTENT_RULES, missing_communication_intents
from ecommerce_customer_service.data import load_jsonl, write_jsonl
from ecommerce_customer_service.evaluation import evaluate_rollouts
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.simulator import ReferenceAgent, run_task
from ecommerce_customer_service.tasks import build_task_dataset, load_task_paths
from ecommerce_customer_service.verifier import required_tool_failures, VERIFIER_VERSION

spec = importlib.util.spec_from_file_location("rescore_local_hf", ROOT / "scripts" / "rescore_local_hf.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CommunicationV10Test(unittest.TestCase):
    def test_confirmed_equivalent_expressions(self):
        cases = {
            "operation_stopped": (
                "在您明确同意之前，退款不会被执行。",
                "目前我只能提供查询服务，无法执行退款操作。",
                "您需要明确确认，目前不能执行退款操作。",
                "好的，您不需要我现在执行任何操作。目前结束本次查询。",
            ),
            "no_duplicate_action": (
                "不需要重复操作或重复退款。",
                "我们不需要再次取消或退款。",
            ),
            "refund_already_completed": (
                "退款状态：completed（已完成）。",
                "退款 ID：refund_example。\n退款金额：65.00元。\n状态：已完成。",
                "退款记录（refund_id: refund_example，金额7500分，状态为completed）。",
                "订单已经完成取消和退款流程。",
                "退款已成功完成（75元）。",
            ),
            "order_cancelled": (
                "您的订单 ord_example 已经于2024年2月18日取消。",
                "订单已于2024-02-18取消。",
            ),
            "return_before_delivery_blocked": (
                "退款政策不允许在订单尚未签收时创建退货退款。",
                "尚未送达，因此不符合立即创建退货退款的条件。",
            ),
            "order_id_confirmation_needed": (
                "您之前说不确定是哪笔，现在确认就是这个订单吗？",
                "请您确认是不是这笔订单。",
            ),
        }
        for intent, responses in cases.items():
            for response in responses:
                with self.subTest(intent=intent, response=response):
                    self.assertTrue(INTENT_RULES[intent].matches(response))

    def test_negative_and_contradictory_minimal_pairs(self):
        cases = {
            "operation_stopped": (
                "退款不会被执行，但我现在就创建退款。",
                "您不需要我现在执行退款操作，但我已经为您执行退款。",
                "在您确认之前退款不会被执行。实际上我将办理退款。",
                "我已停止操作，但退款已经被执行。",
                "不是退款不会被执行。",
                "并非无法执行退款操作。",
                "只是需要确认退款金额。",
            ),
            "refund_already_completed": (
                "退款金额：65元。物流状态：completed。",
                "支付状态：completed，退款尚未处理。",
                "退款状态：pending。",
                "退款记录（refund_id: refund_example，金额65元，状态为pending）。",
                "退款已成功完成，退款状态为处理中。",
                "如果退款状态已完成，请联系我。",
                "订单已取消，但我将完成取消和退款流程。",
            ),
            "no_duplicate_action": (
                "并非不需要重复退款。",
                "不需要重复操作，但我将再次退款。",
                "请重复取消和退款。",
            ),
            "order_cancelled": (
                "订单将于2024年2月18日取消。",
                "订单不是已经于2024年2月18日取消。",
                "订单已于2024年2月18日取消。实际上订单尚未取消。",
            ),
            "return_before_delivery_blocked": (
                "并非不符合创建退货退款的条件。",
                "不符合创建退货退款的条件，但我将创建退货退款。",
                "尚未签收，现在允许创建退货退款。",
            ),
            "order_id_confirmation_needed": (
                "无需现在确认就是这个订单吗。",
                "现在确定就是这个订单，我直接操作。",
            ),
        }
        for intent, responses in cases.items():
            for response in responses:
                with self.subTest(intent=intent, response=response):
                    self.assertFalse(INTENT_RULES[intent].matches(response))

    def test_ambiguous_real_failures_are_not_relabeled(self):
        cases = (
            ("operation_stopped", "您尚未确认执行退款操作，请明确告知是否继续执行退款操作。"),
            ("shipped_address_change_blocked", "订单已发货，地址修改可能不支持，是否取消后重新下单？"),
            ("refund_quantity_two_rejected", "您申请的1件与购买的1件一致，不存在超额退款，请确认退款。"),
            ("order_id_confirmation_needed", "取消订单的条件未满足，所以不进行取消操作。"),
        )
        for intent, response in cases:
            with self.subTest(intent=intent):
                self.assertEqual(missing_communication_intents((intent,), response), [intent])

    def test_true_missing_tools_are_not_waived(self):
        task = TaskSpec(
            task_id="missing-policy", split="dev", scenario="refund_before_delivery",
            user_id="cust_example", user_prompt="未送达，想退货。",
            reference_time="2024-02-18T12:00:00+00:00", database_path=Path("unused.sqlite"),
            assertions=(), required_tools=("get_order", "search_refund_policy"),
        )
        self.assertEqual(required_tool_failures(task, ({"name": "get_order", "ok": True},)),
                         ("required_tool_missing_or_failed:search_refund_policy",))


class ImmutableRescoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rescore-v10-test-")
        self.root = Path(self.temporary.name)
        build_task_dataset(self.root / "dataset", replicas=1, seed=7)
        self.manifest = self.root / "dataset" / "tasks.jsonl"
        tasks = [TaskSpec.from_file(path) for path in load_task_paths(self.manifest, split="dev")]
        rows = [run_task(task, ReferenceAgent()).to_dict() for task in tasks]
        for row in rows:
            row["verifier_version"] = "deterministic_semantic_v9"
            if row["scenario"] == "negative_confirmation":
                row["final_response"] = "在您明确同意之前，退款不会被执行。"
                row["verifier"]["passed"] = False
                row["verifier"]["communication_passed"] = False
                row["verifier"]["failures"] = ["missing_intent:operation_stopped"]
                row["reward"] = 0.0
        self.source = self.root / "original-v9"
        self.source.mkdir()
        self.output = self.root / "derived-v10"
        self.protocol = {
            "split": "dev", "selection": "full_dev", "tasks": len(tasks),
            "task_ids": [task.task_id for task in tasks], "trials_per_task": 1,
            "manifest_sha256": audit.sha256_file(self.manifest),
            "verifier_version": "deterministic_semantic_v9",
        }
        self.summary = {
            "complete": True, "protocol": self.protocol,
            "stages": {stage: {"evaluation": evaluate_rollouts(rows)} for stage in ("base", "sft")},
        }
        audit.write_json(self.source / "protocol.json", self.protocol)
        audit.write_json(self.source / "summary.json", self.summary)
        audit.write_json(self.source / "model_info.json", {"test_only": True})
        for stage in ("base", "sft"):
            (self.source / stage).mkdir()
            write_jsonl(self.source / stage / "rollouts.jsonl", rows)
            audit.write_json(self.source / stage / "report.json", self.summary["stages"][stage])

    def tearDown(self):
        self.temporary.cleanup()

    def test_new_report_retains_original_bytes_and_payload(self):
        before = audit.source_hashes(self.source)
        summary = audit.run(self.source, self.manifest, self.output)
        self.assertEqual(audit.source_hashes(self.source), before)
        self.assertTrue(summary["source_files_unchanged"])
        self.assertFalse(summary["model_rerun"])
        self.assertEqual(summary["verifier_version"], VERIFIER_VERSION)
        self.assertEqual(summary["changed_rows"], 2)
        self.assertEqual(summary["stages"]["sft"]["rescore"]["passed_after"], 7)
        old = load_jsonl(self.source / "sft" / "rollouts.jsonl")
        new = load_jsonl(self.output / "sft" / "rollouts.jsonl")
        for original, derived in zip(old, new):
            for key in set(original) - audit.DERIVED_FIELDS:
                self.assertEqual(original[key], derived[key])
        changes = load_jsonl(self.output / "changes.jsonl")
        self.assertTrue(all(row["removed_failures"] == ["missing_intent:operation_stopped"] for row in changes))

    def test_reject_existing_or_nested_output_without_touching_source(self):
        before = audit.source_hashes(self.source)
        for output in (self.source, self.source / "new", self.root):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                audit.run(self.source, self.manifest, output)
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            audit.run(self.source, self.manifest, self.output)
        self.assertEqual(audit.source_hashes(self.source), before)

    def test_incomplete_and_wrong_manifest_rejected_before_output(self):
        altered = deepcopy(self.summary)
        altered["complete"] = False
        (self.source / "summary.json").unlink()
        audit.write_json(self.source / "summary.json", altered)
        with self.assertRaisesRegex(ValueError, "not complete"):
            audit.run(self.source, self.manifest, self.output)
        self.assertFalse(self.output.exists())
        (self.source / "summary.json").unlink()
        audit.write_json(self.source / "summary.json", self.summary)
        other = self.root / "other.jsonl"
        write_jsonl(other, [])
        with self.assertRaisesRegex(ValueError, "Manifest hash"):
            audit.run(self.source, other, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_and_worker_error_records_rejected(self):
        path = self.source / "base" / "rollouts.jsonl"
        rows = load_jsonl(path)
        for invalid in (rows + [rows[0]], [{**rows[0], "error": "worker failure"}] + rows[1:]):
            with self.subTest(kind=invalid[0].get("error")):
                write_jsonl(path, invalid)
                with self.assertRaises(ValueError):
                    audit.run(self.source, self.manifest, self.output)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
