from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.communication import INTENT_RULES
from ecommerce_customer_service.communication_audit import (
    build_contract, prepare, sha256_file, summarize, validate_judgment, verify_job, write_new,
)
from ecommerce_customer_service.data import load_jsonl, write_jsonl
from ecommerce_customer_service.models import TaskSpec

spec = importlib.util.spec_from_file_location("communication_audit_cli", ROOT / "scripts/audit_communication.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class CommunicationAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="communication-audit-test-")
        self.root = Path(self.temp.name)
        self.task = TaskSpec(
            task_id="refund_quantity_exceeds_000", split="dev", scenario="refund_quantity_exceeds",
            user_id="synthetic-customer", user_prompt="我买了一件，但想按两件退款。", reference_time="2026-01-15",
            database_path=self.root / "not-opened.sqlite", assertions=(),
            required_intents=("purchased_quantity_one", "refund_quantity_two_rejected"),
        )
        self.rubric_path = ROOT / "configs/communication-rubric-v1.json"
        self.rubric = json.loads(self.rubric_path.read_text(encoding="utf-8"))
        self.response = "实际只购买1件，因此不能按2件退款。我没有执行退款。"
        self.row = {
            "task_id": self.task.task_id, "split": "dev", "trial": 0, "reward": 1.0,
            "messages": [{"role": "system", "content": "HIDDEN AGENT PROMPT"},
                         {"role": "user", "content": self.task.user_prompt},
                         {"role": "assistant", "content": self.response}],
            "final_response": self.response, "tool_executions": [],
            "verifier": {"passed": True, "state_passed": True, "policy_passed": True,
                         "communication_passed": True, "required_tools_passed": True, "failures": []},
        }
        self.source = self.root / "source.jsonl"
        self.manifest = self.root / "tasks.jsonl"
        self.job = self.root / "prepared"
        self.save_task(self.task)
        write_jsonl(self.source, [self.row])

    def tearDown(self):
        self.temp.cleanup()

    def save_task(self, task):
        path = self.root / "task.json"
        if path.exists():
            path.unlink()
        write_new(path, task.to_dict())
        write_jsonl(self.manifest, [{"task_id": task.task_id, "split": task.split, "task_file": "task.json"}])

    def prepare(self):
        return prepare(self.manifest, [self.source], self.rubric_path, self.job)

    def request_and_grade(self):
        request = load_jsonl(self.job / "requests.jsonl")[0]
        grade = {"request_id": request["request_id"], "judgment": {"criteria": [
            {"id": item["id"], "status": "met", "evidence": [self.response], "reason": "测试固定判定"}
            for item in request["payload"]["requirements"]]}}
        return request, grade

    def test_rubric_covers_all_existing_intents(self):
        self.assertEqual(set(self.rubric["intent_requirements"]), set(INTENT_RULES))

    def test_unknown_or_empty_requirements_do_not_default_to_success(self):
        for requirements in ((), ("unknown_intent",)):
            with self.subTest(requirements=requirements), self.assertRaises(ValueError):
                build_contract(replace(self.task, required_intents=requirements), self.rubric)

    def test_unseen_business_accepts_an_explicit_reviewed_contract(self):
        custom = {"reviewed": True, "version": "exchange-stock-policy-v1",
                  "policy_facts": ["库存为零时不可承诺完成换货；可以说明缺货并给出政策允许的下一步。"],
                  "requirements": [{"id": "explain_stock_shortage", "description": "说明库存不足，未完成换货，不虚构到货日期。"}]}
        unseen = replace(self.task, scenario="unseen_exchange", required_intents=("new_intent",),
                         metadata={"communication_contract": custom})
        contract = build_contract(unseen, self.rubric)
        self.assertEqual(contract["version"], "exchange-stock-policy-v1")
        self.assertEqual(contract["trusted_policy"], custom["policy_facts"])
        self.assertNotIn("order_cancelled", [r["id"] for r in contract["requirements"]])
        custom["reviewed"] = False
        with self.assertRaises(ValueError):
            build_contract(unseen, self.rubric)

    def test_custom_contract_needs_policy_and_nonempty_unique_criteria(self):
        for custom in (
            {"reviewed": True, "version": "v1", "policy_facts": [], "requirements": []},
            {"reviewed": True, "version": "v1", "policy_facts": ["政策"], "requirements": []},
            {"reviewed": True, "version": "v1", "policy_facts": ["政策"],
             "requirements": [{"id": "facts_grounded", "description": "碰撞通用要求"}]},
        ):
            with self.subTest(custom=custom), self.assertRaises(ValueError):
                build_contract(replace(self.task, metadata={"communication_contract": custom}), self.rubric)

    def test_preparation_is_cpu_only_blind_and_preserves_original_bytes(self):
        before = sha256_file(self.source)
        protocol = self.prepare()
        self.assertEqual(before, sha256_file(self.source))
        self.assertFalse(self.task.database_path.exists())
        self.assertFalse(protocol["api_called"])
        self.assertTrue(protocol["shadow_only"])
        request, _ = self.request_and_grade()
        message_text = json.dumps(request["messages"], ensure_ascii=False)
        for forbidden in ("HIDDEN AGENT PROMPT", self.task.task_id, "original_reward", "checkpoint", "verifier"):
            self.assertNotIn(forbidden, message_text)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_deduplicate_same_payload_but_retain_all_provenance(self):
        other = self.root / "other-model.jsonl"
        other_row = deepcopy(self.row)
        other_row["reward"] = 0.0
        other_row["verifier"].update(passed=False, communication_passed=False,
                                     failures=["missing_intent:refund_quantity_two_rejected"])
        write_jsonl(other, [other_row])
        protocol = prepare(self.manifest, [self.source, other], self.rubric_path, self.job)
        self.assertEqual(protocol["original_rows"], 2)
        self.assertEqual(protocol["unique_requests"], 1)

    def test_test_split_rejected_before_output_creation(self):
        self.save_task(replace(self.task, split="test"))
        self.row["split"] = "test"
        write_jsonl(self.source, [self.row])
        with self.assertRaisesRegex(ValueError, "dev_only"):
            self.prepare()
        self.assertFalse(self.job.exists())

    def test_not_evaluable_is_written_separately_not_as_a_success(self):
        self.save_task(replace(self.task, required_intents=()))
        result = self.prepare()
        self.assertEqual(result["not_evaluable"], 1)
        self.assertEqual(load_jsonl(self.job / "requests.jsonl"), [])
        self.assertEqual(load_jsonl(self.job / "not-evaluable.jsonl")[0]["status"], "not_evaluable")

    def test_exact_evidence_and_complete_criteria_are_required(self):
        self.prepare()
        request, grade = self.request_and_grade()
        self.assertEqual(validate_judgment(request, grade["judgment"])["communication_status"], "pass")
        for change in ("evidence", "missing", "duplicate", "extra"):
            value = deepcopy(grade["judgment"])
            if change == "evidence":
                value["criteria"][0]["evidence"] = ["模型凭空编造的证据"]
            elif change == "missing":
                value["criteria"].pop()
            elif change == "duplicate":
                value["criteria"].append(value["criteria"][0])
            else:
                value["reward"] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_judgment(request, value)

    def test_uncertainty_invalid_output_and_missing_grade_remain_review(self):
        self.prepare()
        request, grade = self.request_and_grade()
        grade["judgment"]["criteria"][0]["status"] = "uncertain"
        self.assertEqual(validate_judgment(request, grade["judgment"])["communication_status"], "needs_review")
        for grades in ([], [{"request_id": request["request_id"], "judgment": None}], [grade]):
            report = summarize(self.job, grades)
            self.assertIsNone(report["audits"][0]["shadow_reward"])
            self.assertFalse(report["calibrated"])

    def test_semantic_success_never_waives_hard_failures(self):
        for field in ("state_passed", "policy_passed", "required_tools_passed"):
            self.row["verifier"][field] = False
        self.row["verifier"]["failures"] = ["runtime_failure:incomplete", "missing_value:69.99"]
        write_jsonl(self.source, [self.row])
        self.prepare()
        _, grade = self.request_and_grade()
        result = summarize(self.job, [grade])["audits"][0]
        self.assertEqual(result["communication_status"], "pass")
        self.assertEqual(result["shadow_reward"], 0.0)
        self.assertFalse(result["hard_passed"])
        self.assertIn("missing_value:69.99", result["hard_failures"])

    def test_job_tampering_is_detected(self):
        self.prepare()
        rows = load_jsonl(self.job / "requests.jsonl")
        rows[0]["payload"]["final_response"] = "修改的回答"
        write_jsonl(self.job / "requests.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "modified"):
            verify_job(self.job)

    def run_options(self, **kwargs):
        return {"job": self.job, "output": self.root / "api-output", "execute": False, "limit": 80,
                "budget": Decimal("5"), "input_price": Decimal("2.5"), "output_price": Decimal("10"),
                "model": "qwen3-max-2026-01-23", **kwargs}

    def test_dry_run_has_no_network_call_or_output_even_with_a_key(self):
        self.prepare()
        def forbidden(*args):
            self.fail("dry run must not send a request")
        result = cli.run_api(**self.run_options(call=forbidden, api_key="not-a-real-key"))
        self.assertTrue(result["dry_run"])
        self.assertFalse((self.root / "api-output").exists())

    def test_zero_negative_nan_or_insufficient_budget_cannot_call(self):
        self.prepare()
        def forbidden(*args):
            self.fail("invalid budget must not send a request")
        for budget in (Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("0.000001")):
            with self.subTest(budget=str(budget)), self.assertRaises(ValueError):
                cli.run_api(**self.run_options(execute=True, budget=budget, call=forbidden))

    def test_paid_run_needs_key_and_explicit_safe_endpoint(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "api_key"):
            cli.run_api(**self.run_options(execute=True))
        for endpoint in ("http://dashscope.aliyuncs.com/v1", "https://evil.example/v1", "https://user:pw@dashscope.aliyuncs.com/v1"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                cli.run_api(**self.run_options(execute=True, api_key="test", api_base=endpoint))

    def test_mock_api_keeps_originals_and_creates_only_shadow_results(self):
        self.prepare()
        request, grade = self.request_and_grade()
        before = sha256_file(self.source)
        def fake(api_base, api_key, model, row, max_tokens):
            self.assertEqual(row["request_id"], request["request_id"])
            return {"model": model, "id": "test-response", "usage": {"prompt_tokens": 100, "completion_tokens": 100},
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(grade["judgment"])}}]}
        result = cli.run_api(**self.run_options(execute=True, api_key="secret-not-in-output",
                              api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", call=fake))
        self.assertTrue(result["complete_selected_subset"])
        self.assertFalse(result["calibrated"])
        self.assertEqual(before, sha256_file(self.source))
        for path in (self.root / "api-output").iterdir():
            self.assertNotIn("secret-not-in-output", path.read_text(encoding="utf-8"))

    def test_timeout_stops_without_retry_and_reports_unknown_billing(self):
        self.prepare()
        calls = []
        def fail(*args):
            calls.append(1)
            raise TimeoutError("do-not-log-sensitive-provider-data")
        result = cli.run_api(**self.run_options(execute=True, api_key="test",
                              api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", call=fail))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["unknown_billing_requests"], 1)
        self.assertIn("TimeoutError", result["failure"])
        self.assertNotIn("do-not-log-sensitive-provider-data", json.dumps(result))

    def test_wrong_model_cannot_become_success(self):
        self.prepare()
        _, grade = self.request_and_grade()
        def wrong_model(*args):
            return {"model": "different-model", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(grade["judgment"])}}]}
        result = cli.run_api(**self.run_options(execute=True, api_key="test",
                              api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", call=wrong_model))
        self.assertIsNotNone(result["failure"])
        report = json.loads((self.root / "api-output/shadow-report.json").read_text(encoding="utf-8"))
        self.assertIsNone(report["audits"][0]["shadow_reward"])

    def test_truncation_and_invalid_json_remain_review(self):
        self.prepare()
        _, grade = self.request_and_grade()
        for number, (finish, content) in enumerate((
            ("length", json.dumps(grade["judgment"])), ("stop", "not-json"),
        )):
            def fake(api_base, api_key, model, row, max_tokens):
                return {"model": model, "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        "choices": [{"finish_reason": finish, "message": {"content": content}}]}
            output = self.root / ("truncated-" + str(number))
            with self.subTest(finish=finish):
                cli.run_api(**self.run_options(execute=True, api_key="test", output=output,
                             api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", call=fake))
                report = json.loads((output / "shadow-report.json").read_text(encoding="utf-8"))
                self.assertIsNone(report["audits"][0]["shadow_reward"])
                self.assertEqual(report["audits"][0]["communication_status"], "needs_review")

    def test_large_input_cannot_use_first_tier_price(self):
        self.prepare()
        rows = load_jsonl(self.job / "requests.jsonl")
        rows[0]["messages"][1]["content"] = "x" * 33_000
        write_jsonl(self.job / "requests.jsonl", rows)
        protocol = json.loads((self.job / "protocol.json").read_text(encoding="utf-8"))
        protocol["output_sha256"]["requests.jsonl"] = sha256_file(self.job / "requests.jsonl")
        (self.job / "protocol.json").unlink()
        write_new(self.job / "protocol.json", protocol)
        def forbidden(*args):
            self.fail("higher pricing tier must not send a request")
        with self.assertRaisesRegex(ValueError, "price_tier"):
            cli.run_api(**self.run_options(execute=True, call=forbidden))
        self.assertFalse((self.root / "api-output").exists())


if __name__ == "__main__":
    unittest.main()
