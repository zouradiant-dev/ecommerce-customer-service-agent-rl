from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ecommerce_customer_service.serving.backend import MockBusinessBackend
from ecommerce_customer_service.serving.context_builder import ContextBuilder, HFTokenCounter, complete_groups
from ecommerce_customer_service.serving.model import SYSTEM_PROMPT, tool_schemas
from ecommerce_customer_service.serving.online_cases import defaults, say, tool
from ecommerce_customer_service.serving.online_evaluation import ScriptedSmokeModel, run_suite
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.storage import ServiceError, Store, encode
from ecommerce_customer_service.serving.telemetry import Telemetry, summarize


class ContextTests(unittest.TestCase):
    def test_tool_pairs_survive_window_compaction(self):
        messages = []
        for index in range(10):
            call = tool("list_orders")
            call["role"] = "assistant"
            call["tool_calls"][0]["id"] = "call_" + str(index)
            messages += [{"role": "user", "content": "查询订单"}, call,
                         {"role": "tool", "tool_call_id": "call_" + str(index), "content": "{}"},
                         {"role": "assistant", "content": "已查询"}]
        original = deepcopy(messages)
        built = ContextBuilder(recent_turns=2).build("system", messages, [], {})
        self.assertEqual(2, built.metadata["kept_groups"])
        self.assertEqual(8, built.metadata["omitted_groups"])
        complete_groups(built.messages[2:])
        self.assertEqual(original, messages)

    def test_incomplete_protocol_is_not_silently_truncated(self):
        call = {"role": "assistant", **tool("list_orders")}
        with self.assertRaises(ServiceError) as error:
            ContextBuilder().build("system", [{"role": "user", "content": "查订单"}, call], [], {})
        self.assertEqual("incomplete_tool_history", error.exception.code)

    def test_current_turn_overflow_fails_without_cutting_user_input(self):
        with self.assertRaises(ServiceError) as error:
            ContextBuilder(max_input_tokens=512).build("system", [{"role": "user", "content": "长" * 2000}], [], {})
        self.assertEqual("current_turn_exceeds_context_budget", error.exception.code)

    def test_tokenizer_counter_uses_chat_template_and_tools(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.options = kwargs
                return list(range(17))
        tokenizer = Tokenizer()
        counter = HFTokenCounter(tokenizer)
        self.assertEqual(17, counter.count([{"role": "user", "content": "测试"}], tool_schemas()))
        self.assertTrue(tokenizer.options["tokenize"])
        self.assertTrue(tokenizer.options["add_generation_prompt"])
        self.assertEqual(4, len(tokenizer.options["tools"]))

    def test_older_quotes_and_operations_do_not_authorize_writes(self):
        messages = [{"role": "user", "content": "我已确认取消 A"}, {"role": "assistant", "content": "待确认"},
                    {"role": "user", "content": "换成 B"}]
        state = {"operations": [{"order_id": "A", "status": "superseded"}]}
        built = ContextBuilder(recent_turns=1).build("system", messages, [], state)
        self.assertIn("摘要不授予", built.messages[1]["content"])
        self.assertIn("superseded", built.messages[1]["content"])
        self.assertEqual("换成 B", built.messages[-1]["content"])

    def test_full_history_control_does_not_silently_compact(self):
        messages = [{"role": "user", "content": "历史" * 3000}, {"role": "assistant", "content": "旧回复"},
                    {"role": "user", "content": "新请求"}]
        with self.assertRaises(ServiceError) as error:
            ContextBuilder(max_input_tokens=4096, policy="full-history").build("system", messages, [], {})
        self.assertEqual("full_history_exceeds_context_budget", error.exception.code)
        result = ContextBuilder(max_input_tokens=4096, recent_turns=1).build("system", messages, [], {})
        self.assertEqual(1, result.metadata["omitted_groups"])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "sessions.sqlite")
        self.backend = MockBusinessBackend(self.root / "business.sqlite")

    def tearDown(self):
        self.temp.cleanup()

    def test_trace_tree_and_usage_do_not_persist_prompt_or_credentials(self):
        class Model:
            model = "sft-test-label"
            def complete(self, messages, tools):
                return {"content": "请说明订单", "_provider_metadata": {
                    "input_tokens": 30, "output_tokens": 5, "response_model": "sft-test-label",
                    "api_key": "SECRET_CREDENTIAL"}}
        service = CustomerService(self.store, self.backend, Model())
        conversation = service.create_conversation("customer_alice")["conversation_id"]
        service.message("customer_alice", conversation, "r1", "PRIVATE_USER_TEXT")
        rows = service.telemetry.rows()
        root = next(row for row in rows if row["kind"] == "turn")
        model = next(row for row in rows if row["kind"] == "model")
        self.assertEqual(root["trace_id"], model["trace_id"])
        self.assertEqual(root["id"], model["parent_id"])
        self.assertNotIn("PRIVATE_USER_TEXT", encode(rows))
        self.assertNotIn("SECRET_CREDENTIAL", encode(rows))
        report = summarize(rows)
        self.assertEqual(30, report["observed_input_tokens"])
        self.assertEqual(5, report["observed_output_tokens"])
        self.assertIsNone(report["task_success_rate"])
        self.assertNotIn("_provider_metadata", encode(service.history("customer_alice", conversation)))

    def test_unknown_usage_is_null_not_zero(self):
        service = CustomerService(self.store, self.backend, ScriptedSmokeModel([say("你好")]))
        conversation = service.create_conversation("customer_alice")["conversation_id"]
        service.message("customer_alice", conversation, "r1", "你好")
        report = summarize(service.telemetry.rows())
        self.assertEqual(0, report["model_usage_coverage"])
        self.assertIsNone(report["observed_input_tokens"])

    def test_model_failure_is_distinct_from_tool_failure(self):
        class BrokenModel:
            def complete(self, messages, tools):
                raise TimeoutError("private endpoint detail")
        service = CustomerService(self.store, self.backend, BrokenModel())
        conversation = service.create_conversation("customer_alice")["conversation_id"]
        service.message("customer_alice", conversation, "r1", "查订单")
        rows = service.telemetry.rows()
        self.assertTrue(any(row["kind"] == "model" and row["status"] == "error" for row in rows))
        self.assertFalse(any(row["kind"] == "tool" for row in rows))
        self.assertNotIn("private endpoint detail", encode(rows))

    def test_interrupted_spans_are_not_counted_as_success(self):
        telemetry = Telemetry(self.store)
        span = telemetry.span("model", "unfinished")
        span.__enter__()
        try:
            self.assertEqual(1, telemetry.recover())
            self.assertEqual(1, summarize(telemetry.rows())["by_kind"]["model"]["unfinished"])
        finally:
            span.__exit__(None, None, None)

    def test_long_history_no_longer_hits_fixed_message_count_limit(self):
        service = CustomerService(self.store, self.backend, ScriptedSmokeModel([say("继续")]))
        conversation = service.create_conversation("customer_alice")["conversation_id"]
        for index in range(80):
            self.store.append(conversation, "history_" + str(index), {"role": "user", "content": "历史问题"}, turn_message=False)
            self.store.append(conversation, "history_" + str(index), {"role": "assistant", "content": "历史回答"}, turn_message=False)
        self.assertNotIn("error", service.message("customer_alice", conversation, "latest", "继续"))
        contexts = [row for row in service.telemetry.rows() if row["kind"] == "context"]
        self.assertGreater(contexts[-1]["attributes"]["omitted_groups"], 0)

    def test_context_state_is_owner_checked_and_expiry_aware(self):
        service = CustomerService(self.store, self.backend)
        conversation = service.create_conversation("customer_alice")["conversation_id"]
        operation = service.preview("customer_alice", conversation, "order_A")
        with self.store.transaction() as db:
            db.execute("UPDATE operations SET expires_at=0 WHERE id=?", (operation["id"],))
        self.assertEqual("expired", self.store.context_state("customer_alice", conversation)["operations"][0]["status"])
        with self.assertRaises(ServiceError):
            self.store.context_state("customer_bob", conversation)


class EvaluationTests(unittest.TestCase):
    def test_engineering_smoke_is_separately_labelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"run"
            result = run_suite(defaults(), output, mode="smoke")
            self.assertTrue(result["complete"])
            self.assertEqual(15, result["passed_cases"])
            self.assertFalse(result["real_model_execution"])
            self.assertFalse(result["production_quality_claim"])
            self.assertTrue((output/"cases/change_target/spans.json").exists())
            with self.assertRaises(FileExistsError):
                run_suite(defaults(), output, mode="smoke")

    def test_gold_labels_and_smoke_fixtures_are_not_sent_to_model(self):
        case = deepcopy(defaults()[4])
        case["expected"]["secret_gold_marker"] = "HIDDEN_GOLD"
        case["smoke_responses"] = [{"content": "HIDDEN_FIXTURE"}]
        class Model:
            model = "real-interface-test-double"
            def complete(self, messages, tools):
                content = encode(messages)
                assert "HIDDEN_GOLD" not in content
                assert "HIDDEN_FIXTURE" not in content
                return say("订单已发货，无法直接取消。")
        with tempfile.TemporaryDirectory() as temporary:
            result = run_suite([case], Path(temporary)/"run", mode="model", model=Model())
            self.assertEqual(1, result["passed_cases"])

    def test_broken_model_endpoint_leaves_incomplete_result_without_success_rate(self):
        class BrokenModel:
            def complete(self, messages, tools):
                raise ConnectionError("offline")
        with tempfile.TemporaryDirectory() as temporary:
            result = run_suite(defaults(), Path(temporary)/"run", mode="model", model=BrokenModel())
            self.assertFalse(result["complete"])
            self.assertEqual(1, result["evaluated_cases"])
            self.assertIsNone(result["case_pass_rate"])

    def test_false_success_reply_is_a_failure_even_if_database_unchanged(self):
        case = deepcopy(defaults()[4])
        case["smoke_responses"] = [say("订单已取消，退款已到账。")]
        with tempfile.TemporaryDirectory() as temporary:
            result = run_suite([case], Path(temporary)/"run", mode="smoke")
            self.assertEqual(0, result["passed_cases"])
            self.assertIn("unsupported_cancel_success_phrase", result["failures"])

    def test_test_split_and_duplicate_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"run"
            case = deepcopy(defaults()[0])
            case["split"] = "test"
            with self.assertRaises(ValueError):
                run_suite([case], output, mode="smoke")
            with self.assertRaises(ValueError):
                run_suite([defaults()[0], defaults()[0]], output, mode="smoke")
            self.assertFalse(output.exists())

    def test_context_ablation_is_scored_on_business_result_not_compaction(self):
        case = defaults()[12]
        with tempfile.TemporaryDirectory() as temporary:
            result = run_suite([case], Path(temporary)/"run", mode="smoke",
                context=ContextBuilder(policy="full-history", max_input_tokens=100000))
            self.assertEqual(1, result["passed_cases"])


if __name__ == "__main__":
    unittest.main()
