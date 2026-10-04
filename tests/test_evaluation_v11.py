from __future__ import annotations

import importlib.util
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
from ecommerce_customer_service.verifier import LEGACY_VERIFIER_VERSION, VERIFIER_VERSION

spec = importlib.util.spec_from_file_location("rescore_local_hf_v11", ROOT / "scripts/rescore_local_hf.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CommunicationV11Test(unittest.TestCase):
    def test_version(self):
        self.assertEqual(LEGACY_VERIFIER_VERSION, "deterministic_semantic_v11")

    def test_confirmed_and_independent_equivalent_expressions(self):
        cases = {
            "return_window_expired": (
                "退货退款请求被拒绝，原因如下：\n商品已超过30天退货期限（实际已过45天）。\n政策：退货窗口为30天。\n无法为您处理此次退款请求。",
                "退货申请被拒绝。\n原因：已超过30天退货期限。\n实际签收已过46天。",
            ),
            "operation_stopped": (
                "目前我将暂停退款操作，直到您明确确认。",
                "我会暂时暂停退款处理。在您确认之前，退款操作将保持暂停状态。",
                "我尚未执行退款操作，只是查询了政策和金额。您需要明确确认后，我才能执行退款。",
                "目前订单信息已查询完成，但未执行退款操作。如果您后续决定退款，可以再次联系我并提供明确的确认指令。",
                "在您确认后，我将执行退款。目前我将暂停退款操作，直到您明确确认。",
                "您要求先不要执行退款，我将保持当前状态，直到您明确确认执行退款操作。",
                "需要用户明确确认后才能操作。目前只查询到政策信息，尚未执行退款。",
            ),
            "order_id_confirmation_needed": (
                "您之前提到要取消的是这个订单吗？还是您想确认其他订单？请明确告知订单号。",
                "请您明确告知准确订单号。",
                "我需要您确认准确的订单号。",
                "您提到取消的是这笔订单吗？",
                "请确认是否要取消此订单。",
            ),
            "refund_already_completed": (
                "退款已处理完成（状态为 refunded），无需重复操作。",
                "退款已处理（状态为 refunded），且有退款记录，金额为75元。",
                "退款已处理（金额7500分，即75元，已退回到支付宝）。",
                "退款已经处理完成。",
            ),
            "refund_quantity_two_rejected": (
                "订单实际购买数量为1件，您只能申请1件的退款。",
                "实际只购买一件，您仅允许申请一件商品的退货。",
                "实际购买数量是1件，不允许按两件退款。",
                "您只能申请1件的退款。如果您想按1件数量申请退款，请确认。",
            ),
        }
        for intent, responses in cases.items():
            for response in responses:
                with self.subTest(intent=intent, response=response):
                    self.assertTrue(INTENT_RULES[intent].matches(response))

    def test_minimal_pair_negation_future_and_contradiction(self):
        cases = {
            "return_window_expired": (
                "退货请求被拒绝，但商品未超过30天退货期限。",
                "如果商品超过30天退货期限，退款请求会被拒绝。",
                "退货申请被拒绝，商品已超过30天退货期限。但仍可以退货。",
                "商品超过30天退货期限，不能退款。实际上现在可以退款。",
            ),
            "operation_stopped": (
                "我不会暂停退款操作。",
                "并非我将暂停退款操作。",
                "如果我将暂停退款操作，再通知您。",
                "我没有暂停退款操作。",
                "目前我将暂停退款操作。但我现在就执行退款。",
                "我将暂停退款操作，无需您确认就直接执行退款。",
                "我尚未执行退款操作，但是我将立即办理退款。",
                "并非尚未执行退款操作。",
                "我将暂停退款操作。退款操作已经执行。",
            ),
            "order_id_confirmation_needed": (
                "请不要明确告知订单号。",
                "如果我请您明确告知订单号，再答复即可。",
                "您之前提到要取消的是这个订单吗？不过我直接取消。",
                "请明确告知准确订单号，但不必确认，我直接取消。",
                "我将查询订单详情，以确认是否可以取消。",
            ),
            "refund_already_completed": (
                "退款已处理，但状态为 pending。",
                "退款已处理完成（状态为pending）。",
                "不是退款已处理完成。",
                "如果退款已处理完成，再通知我。",
                "退款将处理完成。",
                "退款已处理，订单状态为 refunded。",
                "退款已处理，物流状态为 refunded。",
                "退款已处理完成。实际上退款尚未退回。",
                "支付已处理完成，退款尚未处理。",
            ),
            "refund_quantity_two_rejected": (
                "并非只能申请1件的退款。",
                "只能申请1件的退款，但可以按2件退款。",
                "只能申请1件的退款。我将按两件退款。",
                "只能申请11件的退款。",
                "只能申请10件的退款。",
                "仅允许申请2件的退款。",
                "订单购买数量为1件，建议您申请1件退款。",
                "您希望按2件退款，是否有其他原因需要按2件处理？",
                "您提到只想按1件数量申请退款，只能申请1件退款。",
                "并非不允许按两件退款。",
            ),
        }
        for intent, responses in cases.items():
            for response in responses:
                with self.subTest(intent=intent, response=response):
                    self.assertFalse(INTENT_RULES[intent].matches(response))

    def test_ambiguous_business_failures_remain_failures(self):
        cases = (
            ("operation_stopped", "您尚未确认退款。请明确告知是否继续执行退款操作。"),
            ("refund_already_completed", "该订单已取消，有退款记录，退款情况已确认。"),
            ("refund_quantity_two_rejected", "您购买了1件，现在想申请1件的退款。实际只能按1件退款。"),
            ("refund_quantity_two_rejected", "您申请的1件与实际购买数量一致，不存在超额退款。"),
            ("order_id_confirmation_needed", "我已找到您的订单列表，我将查询订单详情。"),
            ("shipped_address_change_blocked", "订单已发货，但退款政策说未签收不能退货，因此改地址可能受限。"),
        )
        for intent, response in cases:
            with self.subTest(intent=intent):
                self.assertEqual(missing_communication_intents((intent,), response), [intent])


class CompletedV10RescoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rescore-v11-test-")
        self.root = Path(self.temp.name)
        build_task_dataset(self.root / "dataset", replicas=1, seed=7)
        self.manifest = self.root / "dataset/tasks.jsonl"
        tasks = [TaskSpec.from_file(path) for path in load_task_paths(self.manifest, split="dev")]
        rows = [run_task(task, ReferenceAgent()).to_dict() for task in tasks]
        for row in rows:
            row["verifier_version"] = "deterministic_semantic_v10"
            if row["scenario"] == "negative_confirmation":
                row["final_response"] = "目前我将暂停退款操作，直到您明确确认。"
                row["verifier"].update(passed=False, communication_passed=False,
                                       failures=["missing_intent:operation_stopped"])
                row["reward"] = 0.0
        self.source = self.root / "original-v10"
        self.source.mkdir()
        self.protocol = {
            "split": "dev", "selection": "full_dev", "tasks": len(tasks),
            "task_ids": [task.task_id for task in tasks], "trials_per_task": 1,
            "manifest_sha256": audit.sha256_file(self.manifest),
            "verifier_version": "deterministic_semantic_v10",
        }
        audit.write_json(self.source / "protocol.json", self.protocol)
        audit.write_json(self.source / "model_info.json", {"test_only": True})
        for stage in ("base", "sft"):
            (self.source / stage).mkdir()
            write_jsonl(self.source / stage / "rollouts.jsonl", rows)
        self.refresh_reports()

    def tearDown(self):
        self.temp.cleanup()

    def refresh_reports(self):
        stages = {}
        for stage in ("base", "sft"):
            stages[stage] = {"evaluation": evaluate_rollouts(load_jsonl(self.source / stage / "rollouts.jsonl"))}
            path = self.source / stage / "report.json"
            if path.exists():
                path.unlink()
            audit.write_json(path, stages[stage])
        path = self.source / "summary.json"
        if path.exists():
            path.unlink()
        audit.write_json(path, {"complete": True, "protocol": self.protocol, "stages": stages})

    def test_v10_inputs_are_preserved_and_both_models_use_the_same_rules(self):
        before = audit.source_hashes(self.source)
        output = self.root / "derived-v11"
        summary = audit.run(self.source, self.manifest, output)
        self.assertEqual(before, audit.source_hashes(self.source))
        self.assertEqual(summary["verifier_version"], VERIFIER_VERSION)
        self.assertEqual(summary["source_verifier_version"], "deterministic_semantic_v10")
        self.assertFalse(summary["model_rerun"])
        self.assertEqual(summary["changed_rows"], 2)
        self.assertEqual(summary["stages"]["base"]["rescore"]["passed_after"], 7)
        self.assertEqual(summary["stages"]["sft"]["rescore"]["passed_after"], 7)
        for stage in ("base", "sft"):
            old = load_jsonl(self.source / stage / "rollouts.jsonl")
            new = load_jsonl(output / stage / "rollouts.jsonl")
            for previous, current in zip(old, new):
                self.assertEqual({k:v for k,v in previous.items() if k not in audit.DERIVED_FIELDS},
                                 {k:v for k,v in current.items() if k not in audit.DERIVED_FIELDS})

    def test_semantic_fix_cannot_waive_missing_tools(self):
        path = self.source / "sft/rollouts.jsonl"
        rows = load_jsonl(path)
        row = next(row for row in rows if row["scenario"] == "negative_confirmation")
        row["tool_executions"] = []
        row["verifier"]["required_tools_passed"] = False
        row["verifier"]["failures"].extend((
            "required_tool_missing_or_failed:verify_customer_identity",
            "required_tool_missing_or_failed:get_order",
            "required_tool_missing_or_failed:search_refund_policy",
        ))
        write_jsonl(path, rows)
        self.refresh_reports()
        summary = audit.run(self.source, self.manifest, self.root / "missing-tools-v11")
        self.assertEqual(summary["stages"]["sft"]["rescore"]["passed_after"], 6)
        rescored = next(row for row in load_jsonl(self.root / "missing-tools-v11/sft/rollouts.jsonl")
                        if row["scenario"] == "negative_confirmation")
        self.assertTrue(rescored["verifier"]["communication_passed"])
        self.assertFalse(rescored["verifier"]["required_tools_passed"])
        self.assertEqual(rescored["reward"], 0.0)

    def test_semantic_fix_cannot_waive_retained_failures_or_failed_state(self):
        path = self.source / "sft/rollouts.jsonl"
        rows = load_jsonl(path)
        row = next(row for row in rows if row["scenario"] == "negative_confirmation")
        row["verifier"]["failures"].append("runtime_failure:isolated_execution_incomplete")
        row["verifier"]["state_passed"] = False
        write_jsonl(path, rows)
        self.refresh_reports()
        output = self.root / "failed-state-v11"
        audit.run(self.source, self.manifest, output)
        rescored = next(row for row in load_jsonl(output / "sft/rollouts.jsonl")
                        if row["scenario"] == "negative_confirmation")
        self.assertTrue(rescored["verifier"]["communication_passed"])
        self.assertFalse(rescored["verifier"]["state_passed"])
        self.assertIn("runtime_failure:isolated_execution_incomplete", rescored["verifier"]["failures"])
        self.assertEqual(rescored["reward"], 0.0)

    def test_retained_failure_prevents_success_even_when_all_components_pass(self):
        path = self.source / "sft/rollouts.jsonl"
        rows = load_jsonl(path)
        row = next(row for row in rows if row["scenario"] == "negative_confirmation")
        row["verifier"]["failures"].append("runtime_failure:isolated_execution_incomplete")
        write_jsonl(path, rows)
        self.refresh_reports()
        output = self.root / "retained-failure-v11"
        audit.run(self.source, self.manifest, output)
        rescored = next(row for row in load_jsonl(output / "sft/rollouts.jsonl")
                        if row["scenario"] == "negative_confirmation")
        for component in ("state_passed", "communication_passed", "policy_passed", "required_tools_passed"):
            self.assertTrue(rescored["verifier"][component])
        self.assertFalse(rescored["verifier"]["passed"])
        self.assertEqual(rescored["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
