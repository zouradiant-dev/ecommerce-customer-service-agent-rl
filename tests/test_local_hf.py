from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_customer_service.local_hf import (
    LocalTransformersAgent, ToolCallParseError, TransformersCompleter, paired_seed,
    parse_qwen_response, select_dev_tasks, sha256_file, validate_lora_adapter,
)
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.simulator import ReferenceAgent, run_task
from ecommerce_customer_service.tasks import build_task_dataset

spec = importlib.util.spec_from_file_location("evaluate_local_hf", ROOT / "scripts" / "evaluate_local_hf.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ParserTest(unittest.TestCase):
    def test_plain_response(self):
        self.assertEqual(parse_qwen_response("  您好。 "), {"role": "assistant", "content": "您好。"})

    def test_multiple_calls_preserve_content_and_unique_ids(self):
        text = ('先查询。<tool_call>{"name":"list_orders","arguments":{"customer_id":"甲"}}</tool_call>'
                '<tool_call>{"name":"get_order","arguments":{"order_id":"乙"}}</tool_call>之后处理。')
        message = parse_qwen_response(text, call_prefix="round_4")
        self.assertEqual(message["content"], "先查询。之后处理。")
        self.assertEqual([call["id"] for call in message["tool_calls"]], ["round_4_0", "round_4_1"])
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"customer_id": "甲"})

    def test_tag_inside_json_string_is_not_a_delimiter(self):
        message = parse_qwen_response('<tool_call>{"name":"x","arguments":{"reason":"含</tool_call>和<tool_call>"}}</tool_call>')
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"])["reason"], "含</tool_call>和<tool_call>")

    def test_malformed_calls_are_not_repaired(self):
        invalid = [
            '<tool_call>{"name":"x","arguments":{}}', '</tool_call>', '<tool_call',
            '<tool_call>{"name":"x","arguments":{}} junk </tool_call>',
            "<tool_call>{'name':'x','arguments':{}}</tool_call>",
            '<tool_call>{"name":"x","arguments":[]}</tool_call>',
            '<tool_call>{"name":"x","arguments":"{}"}</tool_call>',
            '<tool_call>{"name":null,"arguments":{}}</tool_call>',
            '<tool_call>{"name":"x","name":"y","arguments":{}}</tool_call>',
            '<tool_call>{"name":"x","arguments":{"v":NaN}}</tool_call>',
            '<tool_call>{"name":"x","arguments":{"v":1e999}}</tool_call>',
            '<tool_call>{"name":"x","arguments":{"v":1,"v":2}}</tool_call>',
            '<tool_call>{"name":"x","arguments":{},"extra":1}</tool_call>',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ToolCallParseError) as caught:
                parse_qwen_response(text)
            self.assertEqual(caught.exception.raw_text, text)


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="local-hf-test-")
        self.root = Path(self.temporary.name)
        build_task_dataset(self.root / "dataset", replicas=1, seed=7)
        self.manifest = self.root / "dataset" / "tasks.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def task(self, name):
        return TaskSpec.from_file(self.root / "dataset" / "tasks" / f"{name}_000" / "task.json")

    def test_selection_only_dev_and_deterministic(self):
        selected = select_dev_tasks(self.manifest)
        self.assertEqual(len(selected), 7)
        self.assertTrue(all(task.split == "dev" for _, task in selected))
        self.assertEqual([t.task_id for _, t in selected], [t.task_id for _, t in select_dev_tasks(self.manifest)])
        with self.assertRaises(ValueError):
            select_dev_tasks(self.manifest, per_template=2)

    def test_manifest_cannot_disguise_test_task_as_dev(self):
        rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines()]
        row = next(row for row in rows if row["split"] == "test")
        row["split"] = "dev"
        bad = self.manifest.with_name("bad.jsonl")
        bad.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest_task_mismatch"):
            select_dev_tasks(bad)

    def test_no_runtime_oracle_corrections(self):
        seen = []
        def complete(messages, tools):
            seen.append(list(messages))
            return parse_qwen_response("请稍候。")
        row = run_task(self.task("late_refund"), LocalTransformersAgent(complete)).to_dict()
        self.assertEqual(len(seen), 1)
        self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
        self.assertEqual(row["reward"], 0)

    def test_identity_request_variants_receive_a_user_reply_without_oracle(self):
        task = self.task("refund_before_delivery")
        identity = task.metadata["scripted_user_replies"][0]["content"]
        for request in ("请您提供一下姓名和手机号后四位。", "请确认您的姓名和手机号后四位。"):
            seen = []
            def complete(messages, tools):
                seen.append(list(messages))
                if len(seen) == 1:
                    return parse_qwen_response(request)
                self.assertEqual(messages[-1], {"role": "user", "content": identity})
                return parse_qwen_response("已收到身份信息。")
            with self.subTest(request=request):
                row = run_task(task, LocalTransformersAgent(complete)).to_dict()
                self.assertEqual(len(seen), 2)
                self.assertEqual([m["role"] for m in row["messages"]],
                                 ["system", "user", "assistant", "user", "assistant"])
                self.assertEqual(row["reward"], 0)  # Still no actual identity/query tools.

    def test_malformed_model_json_is_failure_not_worker_error(self):
        text = '<tool_call>{"name":"verify_customer_identity","arguments":'
        def complete(messages, tools):
            return parse_qwen_response(text)
        row = run_task(self.task("late_refund"), LocalTransformersAgent(complete)).to_dict()
        self.assertEqual(row["stopped_reason"], "invalid_tool_call")
        self.assertIsNone(row["error"])
        self.assertEqual(row["reward"], 0)
        self.assertEqual(row["messages"][-1]["content"], text)
        self.assertIn("local_parse_error", row["messages"][-1])

    def test_inherited_loop_executes_tools_and_keeps_source_database_unchanged(self):
        task = self.task("cancel_unshipped")
        original = sha256_file(task.database_path)
        reference = run_task(task, ReferenceAgent())
        fixtures = [m for m in reference.messages if m["role"] == "assistant"]
        for trial in range(2):
            remaining = iter(fixtures)
            def complete(messages, tools):
                message = next(remaining)
                raw = message.get("content") or ""
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    raw += '<tool_call>' + json.dumps({"name": function["name"],
                            "arguments": json.loads(function["arguments"])}, ensure_ascii=False) + '</tool_call>'
                return parse_qwen_response(raw)
            row = run_task(task, LocalTransformersAgent(complete), trial=trial).to_dict()
            self.assertEqual(row["reward"], 1.0, row["verifier"])
            self.assertTrue(any(item["name"] == "cancel_order" and item["ok"] for item in row["tool_executions"]))
            self.assertEqual(sha256_file(task.database_path), original)

    def make_adapter(self):
        path = self.root / "adapter"
        path.mkdir()
        config = {"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct", "bias": "none", "r": 16}
        (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        header = json.dumps({"base.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        (path / "adapter_model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (path / filename).write_text("{}", encoding="utf-8")
        return path

    def args(self, adapter, output):
        return SimpleNamespace(manifest=self.manifest, adapter=adapter, output=output,
            base_model="Qwen/Qwen2.5-7B-Instruct", full_dev=False, tasks_per_template=1,
            trials_per_task=1, temperature=0.0, max_steps=20, max_new_tokens=512,
            max_input_tokens=8192, device=0, seed=7, allow_download=False, preflight_only=False)

    def test_adapter_rejects_non_lora_or_truncated_weights(self):
        path = self.make_adapter()
        validate_lora_adapter(path, "Qwen/Qwen2.5-7B-Instruct")
        weights = path / "adapter_model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "offsets"):
            validate_lora_adapter(path, "Qwen/Qwen2.5-7B-Instruct")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_lora_adapter(path, "other-model")

    def test_paired_driver_disables_only_base_and_uses_matched_seeds(self):
        args = self.args(self.make_adapter(), self.root / "out")
        class FakeModel:
            enabled = True
            @contextmanager
            def disable_adapter(self):
                self.enabled = False
                try:
                    yield
                finally:
                    self.enabled = True
        model = FakeModel()
        seen = []
        class FakeCompleter:
            def seed(self, seed):
                seen.append((model.enabled, seed))
            def __call__(self, messages, tools):
                return parse_qwen_response("您好。")
        with patch.object(driver, "parse_args", return_value=args), patch.object(driver, "load_backend", return_value=(model, FakeCompleter(), {"mock": True})), redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(), 0)
        self.assertEqual([enabled for enabled, _ in seen], [False] * 7 + [True] * 7)
        self.assertEqual([seed for _, seed in seen[:7]], [seed for _, seed in seen[7:]])
        summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["paired_comparison"]["common_tasks"], 7)
        self.assertTrue(summary["protocol"]["pilot"])
        self.assertEqual(summary["protocol"]["verifier_version"], "deterministic_semantic_v12")
        self.assertEqual(summary["protocol"]["user_reply_protocol_version"], "scripted_user_replies_v2")
        self.assertEqual(summary["protocol"]["sampling_top_k"], 50)
        self.assertIn("src/ecommerce_customer_service/communication_facts.py", summary["protocol"]["evaluator_source_sha256"])
        self.assertTrue(summary["stages"]["base"]["source_databases_unchanged"])
        self.assertTrue(model.enabled)

    def test_worker_error_preserves_partial_output_and_stops(self):
        args = self.args(self.make_adapter(), self.root / "out")
        selected = select_dev_tasks(self.manifest)
        args.output.mkdir()
        args.database_hashes = {task.task_id: sha256_file(task.database_path) for _, task in selected}
        class FailingCompleter:
            def seed(self, seed):
                pass
            def __call__(self, messages, tools):
                raise RuntimeError("simulated CUDA error")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Infrastructure error"):
            driver.collect_stage("base", selected, FailingCompleter(), args)
        rows = (args.output / "base" / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["stopped_reason"], "worker_error")
        self.assertFalse((args.output / "summary.json").exists())

    def test_preflight_creates_no_output_and_existing_output_is_not_overwritten(self):
        args = self.args(self.make_adapter(), self.root / "out")
        args.preflight_only = True
        with patch.object(driver, "parse_args", return_value=args), patch.object(driver, "load_backend") as loader, redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(), 0)
            loader.assert_not_called()
        self.assertFalse(args.output.exists())
        args.preflight_only = False
        args.output.mkdir()
        sentinel = args.output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with patch.object(driver, "parse_args", return_value=args), patch.object(driver, "load_backend") as loader, redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            driver.main()
        loader.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep")


class BackendTest(unittest.TestCase):
    def test_seed_stable_and_task_trial_specific(self):
        self.assertEqual(paired_seed(7, "a", 0), paired_seed(7, "a", 0))
        self.assertNotEqual(paired_seed(7, "a", 0), paired_seed(7, "b", 0))
        self.assertNotEqual(paired_seed(7, "a", 0), paired_seed(7, "a", 1))

    def test_prompt_normalization_context_guard_and_generation_metadata(self):
        class Batch(dict):
            def to(self, device):
                return self
        class Result:
            def __getitem__(self, key):
                return [21, 22]
        class Tokenizer:
            eos_token_id = 22
            length = 10
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return "prompt"
            def __call__(self, prompt, **kwargs):
                self.tokenizer_kwargs = kwargs
                return Batch(input_ids=SimpleNamespace(shape=(1, self.length)))
            def decode(self, tokens, **kwargs):
                return '完成。'
        class Model:
            calls = 0
            def generate(self, **kwargs):
                self.calls += 1
                return Result()
        tokenizer, model = Tokenizer(), Model()
        backend = TransformersCompleter(model, tokenizer, SimpleNamespace(inference_mode=nullcontext),
                                        SimpleNamespace(max_new_tokens=2), device="cuda:0", max_input_tokens=12)
        previous = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x", "arguments": '{"b":1,"a":2}'}}]}
        response = backend([previous], [{"type": "function", "function": {"name": "x"}}])
        self.assertEqual(tokenizer.messages[0]["tool_calls"][0]["function"]["arguments"], {"b": 1, "a": 2})
        self.assertFalse(tokenizer.tokenizer_kwargs["add_special_tokens"])
        self.assertEqual(response["local_finish_reason"], "stop")
        self.assertEqual(response["local_input_tokens"], 10)
        tokenizer.length = 13
        with self.assertRaisesRegex(RuntimeError, "no truncation"):
            backend([], [])
        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
