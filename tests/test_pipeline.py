from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from http.client import RemoteDisconnected

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.data import (
    assert_no_template_leakage,
    build_grpo_dataset,
    build_sft_dataset,
    combine_rollout_files,
    load_jsonl,
    merge_successful_retries,
    write_jsonl,
)
from ecommerce_customer_service.database import connect, create_database
from ecommerce_customer_service.cli import build_parser
from ecommerce_customer_service.communication import (
    INTENT_RULES,
    TEMPLATE_REQUIRED_INTENTS,
    missing_communication_intents,
)
from ecommerce_customer_service.diagnostics import analyze_bad_cases, assess_rl_gate
from ecommerce_customer_service.evaluation import compare_rollouts, evaluate_rollouts
from ecommerce_customer_service.external_data import convert_abcd_dataset
from ecommerce_customer_service.model_config import qwen_7b_profile
from ecommerce_customer_service.models import TaskSpec, VerificationResult
from ecommerce_customer_service.online_env import EcommerceCustomerServiceEnv, build_online_rows
from ecommerce_customer_service.rewards import score_verification
from ecommerce_customer_service.rescoring import rescore_rollouts
from ecommerce_customer_service.simulator import OpenAICompatibleAgent, ReferenceAgent, WeakBaselineAgent, collect_rollouts, run_task
from ecommerce_customer_service.tasks import (
    TEMPLATE_SPLITS,
    build_failed_task_manifest,
    build_split_manifest,
    build_task_dataset,
    build_template_pilot_manifest,
    load_task_paths,
)
from ecommerce_customer_service.tools import CustomerServiceTools
from ecommerce_customer_service.training import (
    canonicalize_tool_schemas,
    compute_group_advantages,
    normalize_messages,
    save_base_model_configs,
    validate_training_row,
)
from ecommerce_customer_service.verifier import VERIFIER_VERSION, required_tool_failures, verify_task
from ecommerce_customer_service.webapp import DemoApplication


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ecommerce-cs-test-")
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.summary = build_task_dataset(self.dataset, replicas=1, seed=7)
        self.manifest = self.dataset / "tasks.jsonl"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_database_context_releases_windows_file_handle(self) -> None:
        database_path = self.root / "connection-lifecycle.sqlite"
        create_database(database_path)
        with connect(database_path) as connection:
            connection.execute(
                "INSERT INTO customers (id, name, phone_last4, tier) VALUES (?, ?, ?, ?)",
                ("cust_test", "测试用户", "1234", "standard"),
            )
        database_path.unlink()
        self.assertFalse(database_path.exists())

    def test_dataset_has_template_isolation_and_independent_databases(self) -> None:
        self.assertEqual(self.summary["tasks"], 30)
        self.assertEqual(self.summary["split_counts"], {"train": 15, "dev": 7, "test": 8})
        assignments = assert_no_template_leakage(self.manifest)
        self.assertEqual(len(assignments), 30)
        paths = load_task_paths(self.manifest)
        database_paths = {TaskSpec.from_file(path).database_path for path in paths}
        self.assertEqual(len(database_paths), 30)
        self.assertTrue(all(path.exists() for path in database_paths))

    def test_v2_tasks_separate_values_from_semantic_intents(self) -> None:
        rows = load_jsonl(self.manifest)
        self.assertTrue(all(row["schema_version"] == "ecommerce_cs_task_v2" for row in rows))
        self.assertTrue(all("required_facts" not in row for row in rows))
        self.assertTrue(all(isinstance(row["required_values"], list) for row in rows))
        self.assertTrue(all(row["required_intents"] for row in rows))
        self.assertEqual(
            {row["metadata"]["template_id"] for row in rows},
            set(TEMPLATE_REQUIRED_INTENTS),
        )

    def test_communication_intent_examples_cover_every_template(self) -> None:
        examples = json.loads(
            (PROJECT_ROOT / "configs" / "communication-intent-examples.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(TEMPLATE_REQUIRED_INTENTS), set(TEMPLATE_SPLITS))
        self.assertEqual(set(examples), set(TEMPLATE_SPLITS))
        for template, required in TEMPLATE_REQUIRED_INTENTS.items():
            self.assertTrue(required)
            self.assertTrue(all(intent in INTENT_RULES for intent in required))
            self.assertGreaterEqual(len(examples[template]["positive"]), 2)
            self.assertGreaterEqual(len(examples[template]["negative"]), 2)
            for response in examples[template]["positive"]:
                self.assertEqual(
                    missing_communication_intents(required, response),
                    [],
                    msg=f"positive example rejected for {template}: {response}",
                )
            for response in examples[template]["negative"]:
                self.assertTrue(
                    missing_communication_intents(required, response),
                    msg=f"negative example accepted for {template}: {response}",
                )

    def test_communication_intents_reject_negation_and_cross_clause_keyword_games(self) -> None:
        adversarial = {
            "cancel_unshipped": "订单取消失败，取消流程已取消。",
            "cancel_after_shipment": "订单已发货，但并非不能取消，现在仍可取消。",
            "address_change_same_value": "当前地址不同，另一地址相同；无需等待，必须修改地址。",
            "full_quantity_refund": "订单有2件商品，本次只退款1件并原路退回。",
            "late_refund": "订单超过30天，虽然不能说已过期，但仍支持退货。",
        }
        for template, response in adversarial.items():
            self.assertTrue(
                missing_communication_intents(TEMPLATE_REQUIRED_INTENTS[template], response),
                msg=f"adversarial response accepted for {template}: {response}",
            )
        self.assertEqual(
            missing_communication_intents(
                TEMPLATE_REQUIRED_INTENTS["address_change_same_value"],
                "当前是您提供的收货地址。二者完全一致，因此无需重复修改。",
            ),
            [],
        )

    def test_pilot_manifest_selects_one_task_per_requested_template(self) -> None:
        train_dev = self.root / "pilots" / "train-dev.jsonl"
        summary = build_template_pilot_manifest(self.manifest, train_dev)
        self.assertEqual(summary["tasks"], 22)
        self.assertEqual(summary["templates"], 22)
        rows = load_jsonl(train_dev)
        self.assertEqual(len({row["metadata"]["template_id"] for row in rows}), 22)
        self.assertEqual({row["split"] for row in rows}, {"train", "dev"})
        self.assertTrue(all(path.exists() for path in load_task_paths(train_dev)))

        all_templates = self.root / "pilots" / "all.jsonl"
        all_summary = build_template_pilot_manifest(
            self.manifest,
            all_templates,
            splits=("train", "dev", "test"),
        )
        self.assertEqual(all_summary["tasks"], 30)
        self.assertEqual(len(load_task_paths(all_templates)), 30)

        retry_templates = (
            "contradictory_order",
            "late_refund",
            "persistent_tool_error_escalation",
            "refund_policy_tool_transient_error",
        )
        retry_manifest = self.root / "pilots" / "retry.jsonl"
        retry_summary = build_template_pilot_manifest(
            self.manifest,
            retry_manifest,
            templates=retry_templates,
        )
        self.assertEqual(retry_summary["tasks"], 4)
        self.assertEqual(
            {row["metadata"]["template_id"] for row in load_jsonl(retry_manifest)},
            set(retry_templates),
        )

        dev_manifest = self.root / "pilots" / "all-dev.jsonl"
        dev_summary = build_split_manifest(self.manifest, dev_manifest, splits=("dev",))
        self.assertEqual(dev_summary["tasks"], 7)
        self.assertEqual(dev_summary["split_counts"], {"dev": 7})
        self.assertEqual({row["split"] for row in load_jsonl(dev_manifest)}, {"dev"})
        self.assertTrue(all(path.exists() for path in load_task_paths(dev_manifest)))

    def test_failed_task_manifest_selects_only_tasks_without_a_success(self) -> None:
        source = load_jsonl(self.manifest)
        train_rows = [row for row in source if row["split"] == "train"][:3]
        rollouts = self.root / "retry-source.jsonl"
        write_jsonl(
            rollouts,
            [
                {"task_id": train_rows[0]["task_id"], "reward": 1.0, "verifier": {"failures": []}},
                {"task_id": train_rows[1]["task_id"], "reward": 0.0, "verifier": {"failures": ["missing_intent:x"]}},
                {
                    "task_id": train_rows[2]["task_id"],
                    "reward": 0.0,
                    "stopped_reason": "worker_error",
                    "error": "TimeoutError",
                    "verifier": {"failures": ["worker_error:TimeoutError"]},
                },
            ],
        )
        all_retry = self.root / "retry" / "all.jsonl"
        summary = build_failed_task_manifest(
            self.manifest,
            rollouts,
            all_retry,
            splits=("train",),
        )
        self.assertEqual(summary["tasks"], 2)
        self.assertEqual(
            {row["task_id"] for row in load_jsonl(all_retry)},
            {train_rows[1]["task_id"], train_rows[2]["task_id"]},
        )
        self.assertTrue(all(path.exists() for path in load_task_paths(all_retry)))

        worker_retry = self.root / "retry" / "worker.jsonl"
        worker_summary = build_failed_task_manifest(
            self.manifest,
            rollouts,
            worker_retry,
            splits=("train",),
            failure_mode="worker_error",
        )
        self.assertEqual(worker_summary["task_ids"], [train_rows[2]["task_id"]])

    def test_merge_retries_replaces_only_failed_rows_with_successes(self) -> None:
        base = self.root / "merge-base.jsonl"
        retry_one = self.root / "merge-retry-one.jsonl"
        retry_two = self.root / "merge-retry-two.jsonl"
        output = self.root / "merge-output.jsonl"
        write_jsonl(
            base,
            [
                {"task_id": "a", "reward": 1.0, "value": "base-a"},
                {"task_id": "b", "reward": 0.0, "value": "base-b"},
                {"task_id": "c", "reward": 0.0, "value": "base-c"},
            ],
        )
        write_jsonl(
            retry_one,
            [
                {"task_id": "a", "reward": 1.0, "value": "retry-a"},
                {"task_id": "b", "reward": 1.0, "value": "retry-b"},
                {"task_id": "c", "reward": 0.0, "value": "retry-c-failed"},
            ],
        )
        write_jsonl(retry_two, [{"task_id": "c", "reward": 1.0, "value": "retry-c"}])

        summary = merge_successful_retries(base, [retry_one, retry_two], output)
        rows = {row["task_id"]: row for row in load_jsonl(output)}
        self.assertEqual(summary["replacements"], 2)
        self.assertEqual(summary["passed"], 3)
        self.assertEqual(summary["still_failed"], 0)
        self.assertEqual(rows["a"]["value"], "base-a")
        self.assertEqual(rows["b"]["value"], "retry-b")
        self.assertEqual(rows["c"]["value"], "retry-c")
        self.assertEqual(rows["a"]["selection_metadata"]["selected_source"], str(base.resolve()))

    def test_combine_rollouts_requires_disjoint_task_ids(self) -> None:
        train = self.root / "combine-train.jsonl"
        dev = self.root / "combine-dev.jsonl"
        output = self.root / "combined.jsonl"
        write_jsonl(train, [{"task_id": "train-a", "split": "train", "reward": 1.0}])
        write_jsonl(dev, [{"task_id": "dev-a", "split": "dev", "reward": 1.0}])
        summary = combine_rollout_files([train, dev], output)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["split_counts"], {"dev": 1, "train": 1})
        self.assertEqual(summary["failed"], 0)
        rows = load_jsonl(output)
        self.assertEqual({row["task_id"] for row in rows}, {"train-a", "dev-a"})
        self.assertTrue(all(row["combination_metadata"]["source"] for row in rows))

        duplicate = self.root / "combine-duplicate.jsonl"
        write_jsonl(duplicate, [{"task_id": "train-a", "split": "dev", "reward": 1.0}])
        with self.assertRaisesRegex(ValueError, "Duplicate task ID"):
            combine_rollout_files([train, duplicate], self.root / "invalid-combined.jsonl")

    def test_qwen_7b_is_the_default_real_model(self) -> None:
        args = build_parser().parse_args(
            [
                "run-model",
                "--manifest",
                str(self.manifest),
                "--output",
                str(self.root / "model-rollouts.jsonl"),
            ]
        )
        self.assertEqual(args.model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(args.scenario)
        self.assertIsNone(args.limit)
        file_profile = json.loads((PROJECT_ROOT / "configs" / "qwen2.5-7b-qlora.json").read_text(encoding="utf-8"))
        code_profile = qwen_7b_profile()
        self.assertEqual(file_profile["profile_version"], code_profile["profile_version"])
        self.assertEqual(file_profile["sft"]["lora_rank"], code_profile["sft"]["lora"]["rank"])

    def test_openai_agent_merges_streaming_tool_call_deltas(self) -> None:
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "verify_customer_identity"},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"name":"小王",'}},
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"phone_last4":"1234"}'}},
                            ]
                        }
                    }
                ]
            },
        ]
        body = "\n\n".join(f"data: {json.dumps(event, ensure_ascii=False)}" for event in events)
        body += "\n\ndata: [DONE]\n"
        message = OpenAICompatibleAgent._parse_sse_message(body)
        call = message["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "verify_customer_identity")
        self.assertEqual(json.loads(call["function"]["arguments"])["phone_last4"], "1234")

    def test_qwen37_plus_disables_thinking_without_affecting_other_endpoints(self) -> None:
        messages = [{"role": "user", "content": "测试"}]
        qwen = OpenAICompatibleAgent("qwen3.7-plus-2026-05-26")
        self.assertIs(qwen._request_body(messages, [])["enable_thinking"], False)

        local = OpenAICompatibleAgent("Qwen/Qwen2.5-7B-Instruct")
        self.assertNotIn("enable_thinking", local._request_body(messages, []))

        thinking_qwen = OpenAICompatibleAgent("qwen3.7-plus-2026-05-26", enable_thinking=True)
        self.assertIs(thinking_qwen._request_body(messages, [])["enable_thinking"], True)

    def test_openai_agent_retries_remote_disconnect(self) -> None:
        class Response:
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            @staticmethod
            def read() -> bytes:
                return json.dumps({"choices": [{"message": {"role": "assistant", "content": "完成"}}]}).encode()

        agent = OpenAICompatibleAgent("stub-model", retries=2)
        with (
            patch(
                "ecommerce_customer_service.simulator.urlopen",
                side_effect=[RemoteDisconnected("remote closed connection"), Response()],
            ) as mocked_urlopen,
            patch("ecommerce_customer_service.simulator.time.sleep"),
        ):
            message = agent._complete([{"role": "user", "content": "测试"}], [])

        self.assertEqual(message["content"], "完成")
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_tool_schema_canonicalization_is_recursive_and_stable(self) -> None:
        schemas = canonicalize_tool_schemas(CustomerServiceTools.schemas())
        self.assertEqual(list(schemas[0]), sorted(schemas[0]))
        function = schemas[0]["function"]
        self.assertEqual(list(function), sorted(function))
        parameters = function["parameters"]
        self.assertEqual(list(parameters), sorted(parameters))
        self.assertEqual(schemas, canonicalize_tool_schemas(schemas))

    def test_model_visible_identifiers_are_opaque_and_seeded(self) -> None:
        rows = load_jsonl(self.manifest)
        for row in rows:
            metadata = row["metadata"]
            scenario = row["scenario"]
            for key in ("customer_id", "order_id", "product_id", "item_id", "payment_id", "shipment_id"):
                self.assertNotIn(scenario, metadata[key])
                self.assertRegex(metadata[key], r"^[a-z]+_[0-9a-f]{10}$")

        same_a = self.root / "same-a"
        same_b = self.root / "same-b"
        different = self.root / "different"
        build_task_dataset(same_a, replicas=2, seed=19)
        build_task_dataset(same_b, replicas=2, seed=19)
        build_task_dataset(different, replicas=2, seed=20)
        self.assertEqual((same_a / "tasks.jsonl").read_text(encoding="utf-8"), (same_b / "tasks.jsonl").read_text(encoding="utf-8"))
        self.assertNotEqual((same_a / "tasks.jsonl").read_text(encoding="utf-8"), (different / "tasks.jsonl").read_text(encoding="utf-8"))

    def test_identity_tool_discovers_internal_customer_id(self) -> None:
        task = TaskSpec.from_file(load_task_paths(self.manifest, split="train")[0])
        schema = next(
            item["function"]
            for item in CustomerServiceTools.schemas()
            if item["function"]["name"] == "verify_customer_identity"
        )
        self.assertNotIn("customer_id", schema["parameters"]["properties"])
        tools = CustomerServiceTools(task.database_path, task)
        result = tools.execute("verify_customer_identity", task.metadata["identity"])
        self.assertTrue(result.ok)
        self.assertEqual(result.output["customer_id"], task.user_id)

    def test_invalid_model_tool_argument_does_not_crash_rollout(self) -> None:
        task = TaskSpec.from_file(load_task_paths(self.manifest, split="train")[0])
        tools = CustomerServiceTools(task.database_path, task)
        result = tools.execute(
            "verify_customer_identity",
            {"name": None, "phone_last4": None},
        )
        self.assertFalse(result.ok)
        self.assertIn("strip", result.output["error"])
        self.assertEqual(len(tools.executions), 1)

    def test_available_tools_follow_verified_state_and_confirmation(self) -> None:
        task = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "cancel_unshipped"
        )
        isolated_db = self.root / "available-tools.sqlite"
        shutil.copy2(task.database_path, isolated_db)
        isolated = replace(task, database_path=isolated_db)
        tools = CustomerServiceTools(isolated_db, isolated)

        names = {item["function"]["name"] for item in tools.available_schemas()}
        self.assertEqual(names, {"verify_customer_identity"})

        tools.observe_user_message(task.metadata["scripted_user_replies"][0]["content"])
        verified = tools.execute("verify_customer_identity", task.metadata["identity"])
        self.assertTrue(verified.ok)
        names = {item["function"]["name"] for item in tools.available_schemas()}
        self.assertIn("get_order", names)
        self.assertNotIn("escalate_to_human", names)
        self.assertNotIn("cancel_order", names)

        tools.execute("get_order", {"customer_id": task.user_id, "order_id": task.metadata["order_id"]})
        self.assertNotIn("cancel_order", {item["function"]["name"] for item in tools.available_schemas()})
        self.assertNotIn("escalate_to_human", {item["function"]["name"] for item in tools.available_schemas()})
        confirmation = next(
            item["content"] for item in task.metadata["scripted_user_replies"] if item["kind"] == "confirmation"
        )
        tools.observe_user_message(confirmation)
        self.assertIn("cancel_order", {item["function"]["name"] for item in tools.available_schemas()})

    def test_scripted_replies_require_an_explicit_model_request(self) -> None:
        self.assertFalse(OpenAICompatibleAgent._assistant_requests_reply("identity", "好的，请稍候。"))
        self.assertTrue(
            OpenAICompatibleAgent._assistant_requests_reply("identity", "请提供姓名和手机号后四位。")
        )
        self.assertFalse(OpenAICompatibleAgent._assistant_requests_reply("confirmation", "已经为您处理。"))
        self.assertTrue(OpenAICompatibleAgent._assistant_requests_reply("confirmation", "请确认是否继续退款。"))
        self.assertTrue(OpenAICompatibleAgent._assistant_requests_reply("confirmation", "请您确认是否退款。"))
        self.assertTrue(OpenAICompatibleAgent._assistant_requests_reply("confirmation", "您确认要取消该订单吗？"))
        self.assertTrue(OpenAICompatibleAgent._assistant_requests_reply("confirmation", "请确认是否取消订单。"))
        self.assertTrue(OpenAICompatibleAgent._assistant_requests_reply("confirmation", "您是否确定要修改地址？"))

    def test_escalation_appears_only_after_a_real_failure_path(self) -> None:
        task = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "cancel_unshipped"
        )
        tools = CustomerServiceTools(task.database_path, task)
        failed = tools.execute(
            "verify_customer_identity",
            {"name": "错误姓名", "phone_last4": "0000"},
        )
        self.assertFalse(failed.ok)
        self.assertIn("escalate_to_human", {item["function"]["name"] for item in tools.available_schemas()})

    def test_correction_targets_missing_read_and_requested_write(self) -> None:
        multi = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "multi_order_disambiguation"
        )
        tools = CustomerServiceTools(multi.database_path, multi)
        tools.observe_user_message(multi.user_prompt)
        tools.execute("verify_customer_identity", multi.metadata["identity"])
        tools.execute("list_orders", {"customer_id": multi.user_id})
        correction = OpenAICompatibleAgent._correction_for_state(tools, [])
        self.assertIsNotNone(correction)
        assert correction is not None
        self.assertEqual(correction[0], "required_read:get_order")
        self.assertIn("get_order", correction[1])

        staged = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "update_address"
        )
        staged_tools = CustomerServiceTools(staged.database_path, staged)
        staged_tools.observe_user_message(staged.user_prompt)
        staged_tools.execute("verify_customer_identity", staged.metadata["identity"])
        staged_tools.execute(
            "get_order",
            {"customer_id": staged.user_id, "order_id": staged.metadata["order_id"]},
        )
        correction = OpenAICompatibleAgent._correction_for_state(
            staged_tools,
            [
                item
                for item in staged.metadata["scripted_user_replies"]
                if item["kind"] == "confirmation"
            ],
        )
        self.assertIsNotNone(correction)
        assert correction is not None
        self.assertEqual(correction[0], "confirmation")
        self.assertIn("修改收货地址", correction[1])

    def test_escalation_visibility_enforces_minimum_attempts_and_stops_after_success(self) -> None:
        task = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "persistent_tool_error_escalation"
        )
        tools = CustomerServiceTools(task.database_path, task)
        tools.observe_user_message(task.user_prompt)
        tools.execute("verify_customer_identity", task.metadata["identity"])
        self.assertNotIn("escalate_to_human", {item["function"]["name"] for item in tools.available_schemas()})
        hidden_escalation = tools.execute(
            "escalate_to_human",
            {"issue": "订单查询失败", "reason": "尚未达到重试次数"},
        )
        self.assertFalse(hidden_escalation.ok)
        self.assertEqual(hidden_escalation.policy_violation, "tool_not_available_in_current_state")
        arguments = {"customer_id": task.user_id, "order_id": task.metadata["order_id"]}
        tools.execute("get_order", arguments)
        self.assertNotIn("escalate_to_human", {item["function"]["name"] for item in tools.available_schemas()})
        still_hidden = tools.execute(
            "escalate_to_human",
            {"issue": "订单查询失败", "reason": "只失败了一次"},
        )
        self.assertFalse(still_hidden.ok)
        self.assertEqual(still_hidden.policy_violation, "tool_not_available_in_current_state")
        tools.execute("get_order", arguments)
        self.assertIn("escalate_to_human", {item["function"]["name"] for item in tools.available_schemas()})
        escalated = tools.execute("escalate_to_human", {"issue": "订单查询失败", "reason": "连续超时"})
        self.assertTrue(escalated.ok)
        self.assertEqual(
            escalated.output["next_action"],
            "向用户说明失败原因和工单号并立即结束；不要再调用任何工具",
        )
        self.assertEqual(escalated.output["reason"], "连续超时")
        self.assertEqual(tools.available_schemas(), [])
        self.assertIsNone(OpenAICompatibleAgent._correction_for_state(tools, []))
        post_escalation = tools.execute("get_order", arguments)
        self.assertFalse(post_escalation.ok)
        self.assertEqual(post_escalation.policy_violation, "tool_not_available_in_current_state")

    def test_empty_tool_surface_explicitly_disables_model_tool_calls(self) -> None:
        agent = OpenAICompatibleAgent("local-model")
        body = agent._request_body([{"role": "user", "content": "完成"}], [])
        self.assertNotIn("tools", body)
        self.assertEqual(body["tool_choice"], "none")

    def test_write_requires_observed_confirmation_and_teacher_is_multiturn(self) -> None:
        task = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "cancel_unshipped"
        )
        isolated_db = self.root / "confirmation.sqlite"
        shutil.copy2(task.database_path, isolated_db)
        isolated = replace(task, database_path=isolated_db)
        tools = CustomerServiceTools(isolated_db, isolated)
        tools.observe_user_message(isolated.user_prompt)
        tools.execute("verify_customer_identity", isolated.metadata["identity"])
        tools.execute("get_order", {"customer_id": isolated.user_id, "order_id": isolated.metadata["order_id"]})
        denied = tools.execute(
            "cancel_order",
            {"customer_id": isolated.user_id, "order_id": isolated.metadata["order_id"], "confirmed": True},
        )
        self.assertFalse(denied.ok)
        self.assertEqual(denied.output["error"], "tool_not_available_in_current_state")
        self.assertEqual(denied.policy_violation, "tool_not_available_in_current_state")

        tools.observe_user_message("我确认退款，取消订单需要稍后再说。")
        wrong_action = tools.execute(
            "cancel_order",
            {"customer_id": isolated.user_id, "order_id": isolated.metadata["order_id"], "confirmed": True},
        )
        self.assertFalse(wrong_action.ok)
        self.assertEqual(wrong_action.output["error"], "tool_not_available_in_current_state")
        self.assertEqual(wrong_action.policy_violation, "tool_not_available_in_current_state")

        confirmation = next(
            item["content"]
            for item in isolated.metadata["scripted_user_replies"]
            if item["kind"] == "confirmation"
        )
        tools.observe_user_message(confirmation)
        accepted = tools.execute(
            "cancel_order",
            {"customer_id": isolated.user_id, "order_id": isolated.metadata["order_id"], "confirmed": True},
        )
        self.assertTrue(accepted.ok)

        rollout = run_task(task, ReferenceAgent())
        self.assertEqual(rollout.reward, 1.0)
        self.assertTrue(
            any(message.get("role") == "user" and message.get("content") == confirmation for message in rollout.messages)
        )
        write_index = next(
            index
            for index, message in enumerate(rollout.messages)
            if message.get("tool_calls")
            and message["tool_calls"][0]["function"]["name"] == "cancel_order"
        )
        confirmation_index = next(
            index for index, message in enumerate(rollout.messages) if message.get("content") == confirmation
        )
        self.assertLess(confirmation_index, write_index)

    def test_verifier_rejects_unsupported_success_and_premature_escalation(self) -> None:
        tasks = [TaskSpec.from_file(path) for path in load_task_paths(self.manifest)]
        shipment = next(task for task in tasks if task.scenario == "shipment_status")
        unsupported = verify_task(shipment, "订单已取消。", ())
        self.assertIn("unsupported_success_claim:cancel_order", unsupported.failures)
        self.assertFalse(unsupported.policy_passed)

        persistent = next(task for task in tasks if task.scenario == "persistent_tool_error_escalation")
        premature = verify_task(
            persistent,
            "查询服务持续异常，已经为您创建人工工单。",
            (
                {"name": "verify_customer_identity", "ok": True, "is_write": False},
                {"name": "get_order", "ok": False, "is_write": False, "output": {"error": "timeout"}},
                {"name": "escalate_to_human", "ok": True, "is_write": True, "output": {"escalated": True}},
            ),
        )
        self.assertTrue(any(item.startswith("premature_escalation:get_order") for item in premature.failures))
        self.assertFalse(premature.policy_passed)

        blocked_after_escalation = verify_task(
            persistent,
            "订单查询持续超时，已为您创建人工工单。",
            (
                {"name": "verify_customer_identity", "ok": True, "is_write": False},
                {"name": "get_order", "ok": False, "is_write": False, "output": {"error": "timeout"}},
                {"name": "get_order", "ok": False, "is_write": False, "output": {"error": "timeout"}},
                {"name": "escalate_to_human", "ok": True, "is_write": True, "output": {"escalated": True}},
                {
                    "name": "get_order",
                    "ok": False,
                    "is_write": False,
                    "policy_violation": "tool_not_available_in_current_state",
                    "output": {"error": "tool_not_available_in_current_state"},
                },
            ),
        )
        self.assertIn(
            "policy_violation:tool_not_available_in_current_state",
            blocked_after_escalation.failures,
        )
        self.assertFalse(
            any(item.startswith("unrecovered_tool_error:get_order") for item in blocked_after_escalation.failures)
        )

    def test_verifier_accepts_deterministic_communication_aliases(self) -> None:
        task = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "address_change_same_value"
        )
        response = (
            f"当前收货地址已经是 **{task.metadata['target_address']}**。"
            "新地址与当前地址完全一致，因此无需重复写入。"
        )
        verification = verify_task(
            task,
            response,
            (
                {"name": "verify_customer_identity", "ok": True},
                {"name": "get_order", "ok": True},
            ),
        )
        self.assertTrue(verification.communication_passed)
        self.assertTrue(verification.passed)

        contradictory = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "contradictory_order"
        )
        executions = (
            {"name": "verify_customer_identity", "ok": True},
            {"name": "list_orders", "ok": True},
        )
        for teacher_response in (
            "您名下目前只有一个订单。请问您确认要取消这个订单吗？",
            f"请问您是要取消这个订单 {contradictory.metadata['order_id']} 吗？",
        ):
            with self.subTest(teacher_response=teacher_response):
                result = verify_task(contradictory, teacher_response, executions)
                self.assertTrue(result.communication_passed)
                self.assertTrue(result.passed)
        self.assertEqual(
            missing_communication_intents(
                ("order_id_confirmation_needed",),
                "订单已取消，无需确认。",
            ),
            ["order_id_confirmation_needed"],
        )
        for escalation_response in (
            "系统持续超时，已生成人工工单 ticket_001。",
            "订单系统持续超时，已为您升级人工处理，工单号为 ticket_001。",
            "多次查询失败，已为您创建人工工单 ticket_001。",
            "连续两次查询超时，已为您升级至人工客服处理，工单号为 ticket_001。",
        ):
            with self.subTest(escalation_response=escalation_response):
                self.assertEqual(
                    missing_communication_intents(
                        ("query_persistent_failure", "human_escalation_created"),
                        escalation_response,
                    ),
                    [],
                )

        self.assertEqual(
            missing_communication_intents(
                TEMPLATE_REQUIRED_INTENTS["cancel_without_captured_payment"],
                "订单已取消，预授权将被释放，不会产生实际退款。",
            ),
            [],
        )
        self.assertEqual(
            missing_communication_intents(
                TEMPLATE_REQUIRED_INTENTS["full_quantity_refund"],
                "退款已成功，剩余可退数量为0，款项原路退回。",
            ),
            [],
        )
        for template, response in (
            ("address_change_after_shipment", "订单已经发货，无法直接修改收货地址。"),
            ("cancel_already_cancelled", "订单已取消且已退款，无需进行任何重复操作。"),
            ("contradictory_order", "您目前只有一个订单，请问您确认要取消订单 ord_001 吗？"),
            ("late_refund", "商品已超过30天退货期限，因此不允许退货退款。"),
            ("negative_confirmation", "好的，我不会执行任何退款操作。"),
            ("refund_before_delivery", "订单尚未签收，目前不允许创建退货退款。"),
            ("refund_quantity_exceeds", "实际购买数量为1件，不允许按2件申请退款。"),
        ):
            with self.subTest(template=template):
                self.assertEqual(
                    missing_communication_intents(TEMPLATE_REQUIRED_INTENTS[template], response),
                    [],
                )

    def test_required_money_value_accepts_equivalent_display_precision(self) -> None:
        task = TaskSpec.from_file(load_task_paths(self.manifest)[0])
        for expected, response in (
            ("219.00", "本次金额为219元。"),
            ("69.90", "本次金额为69.9元。"),
            ("0.50", "本次金额为0.5元。"),
        ):
            communication_only = replace(
                task,
                assertions=(),
                required_values=(expected,),
                required_intents=(),
                required_tools=(),
            )
            verification = verify_task(communication_only, response, ())
            self.assertTrue(verification.communication_passed)

    def test_latest_order_accepts_shipment_lookup_as_order_detail_evidence(self) -> None:
        task = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "latest_order_lookup"
        )
        executions = (
            {"name": "verify_customer_identity", "ok": True},
            {"name": "list_orders", "ok": True},
            {"name": "get_shipment", "ok": True},
        )
        self.assertEqual(required_tool_failures(task, executions), ())

    def test_rescore_preserves_raw_rollout_and_versions_derived_result(self) -> None:
        task = next(
            TaskSpec.from_file(path)
            for path in load_task_paths(self.manifest)
            if TaskSpec.from_file(path).scenario == "cancel_already_cancelled"
        )
        source = run_task(task, ReferenceAgent()).to_dict()
        source["schema_version"] = "ecommerce_cs_rollout_v1"
        source.pop("verifier_version", None)
        source["reward"] = 0.0
        source["verifier"]["passed"] = False
        source["verifier"]["communication_passed"] = False
        source["verifier"]["failures"] = ["missing_intent:no_duplicate_action"]
        raw_path = self.root / "raw-for-rescore.jsonl"
        output_path = self.root / "rescored.jsonl"
        write_jsonl(raw_path, [source])
        raw_before = raw_path.read_text(encoding="utf-8")

        summary = rescore_rollouts(raw_path, self.manifest, output_path)
        rescored = load_jsonl(output_path)[0]
        self.assertEqual(summary["passed_before"], 0)
        self.assertEqual(summary["passed_after"], 1)
        self.assertEqual(rescored["schema_version"], "ecommerce_cs_rollout_v2")
        self.assertEqual(rescored["verifier_version"], VERIFIER_VERSION)
        self.assertTrue(rescored["verifier"]["passed"])
        self.assertEqual(raw_path.read_text(encoding="utf-8"), raw_before)

    def test_abcd_converter_is_language_only_and_excludes_test(self) -> None:
        source = self.root / "abcd.json"
        source.write_text(
            json.dumps(
                {
                    "train": [
                        {
                            "convo_id": 17,
                            "scenario": {"flow": "product_defect", "subflow": "return_size"},
                            "delexed": [
                                {"speaker": "customer", "text": "i need a return", "targets": []},
                                {"speaker": "agent", "text": "may i verify your account?", "targets": []},
                                {
                                    "speaker": "action",
                                    "text": "account <account_id> verified",
                                    "targets": ["return_size", "take_action", "pull-up-account", [], -1],
                                },
                                {"speaker": "agent", "text": "your account is verified", "targets": []},
                            ],
                        }
                    ],
                    "dev": [],
                    "test": [{"convo_id": 99, "delexed": []}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        output = self.root / "abcd-sft"
        summary = convert_abcd_dataset(source, output)
        self.assertEqual(summary["splits"], {"train": 1, "dev": 0})
        self.assertRegex(summary["source_sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse((output / "test.jsonl").exists())
        row = load_jsonl(output / "train.jsonl")[0]
        self.assertEqual(row["metadata"]["training_scope"], "language_sft_only")
        self.assertEqual(row["metadata"]["excluded_action_turns"][0]["action"], "pull-up-account")
        self.assertNotIn("action", {message["role"] for message in row["messages"]})
        with self.assertRaises(ValueError):
            convert_abcd_dataset(source, self.root / "bad-abcd", splits=("test",))

    def test_reference_closes_state_loop_and_weak_baseline_exposes_failures(self) -> None:
        paths = load_task_paths(self.manifest)
        teacher_path = self.root / "teacher.jsonl"
        baseline_path = self.root / "baseline.jsonl"
        teacher = collect_rollouts(paths, lambda task, trial: ReferenceAgent(), teacher_path)
        baseline = collect_rollouts(paths, lambda task, trial: WeakBaselineAgent(), baseline_path)
        self.assertEqual(teacher, {"attempted": 30, "successes": 30})
        self.assertGreater(baseline["successes"], 0)
        self.assertLess(baseline["successes"], 30)

        teacher_eval = evaluate_rollouts(load_jsonl(teacher_path))
        baseline_eval = evaluate_rollouts(load_jsonl(baseline_path))
        self.assertEqual(teacher_eval["pass_1"], 1.0)
        self.assertLess(baseline_eval["pass_1"], teacher_eval["pass_1"])
        badcases = analyze_bad_cases(load_jsonl(baseline_path))
        self.assertGreater(badcases["failed_rollouts"], 0)
        comparison = compare_rollouts(load_jsonl(baseline_path), load_jsonl(teacher_path))
        self.assertGreater(comparison["delta"], 0.0)

    def test_sft_excludes_test_and_grpo_keeps_only_mixed_groups(self) -> None:
        paths = load_task_paths(self.manifest)
        teacher_path = self.root / "teacher.jsonl"
        collect_rollouts(paths, lambda task, trial: ReferenceAgent(), teacher_path)
        sft_summary = build_sft_dataset(teacher_path, self.root / "sft")
        self.assertEqual(sft_summary, {"train": 15, "dev": 7, "skipped": 8, "tool_scope": "all"})
        sft_rows = load_jsonl(self.root / "sft" / "train.jsonl") + load_jsonl(self.root / "sft" / "dev.jsonl")
        self.assertTrue(all(row["metadata"]["split"] != "test" for row in sft_rows))
        self.assertTrue(all(len(row["tools"]) == 9 for row in sft_rows))
        self.assertTrue(any(sum(message.get("role") == "user" for message in row["messages"]) > 1 for row in sft_rows))

        base = load_jsonl(teacher_path)
        mixed = []
        for row in base:
            if row["split"] == "test":
                continue
            good = dict(row)
            good["trial"] = 0
            bad = json.loads(json.dumps(row, ensure_ascii=False))
            bad["trial"] = 1
            bad["reward"] = 0.0
            bad["verifier"]["passed"] = False
            bad["verifier"]["failures"] = ["state_not_updated:synthetic"]
            mixed.extend([good, bad])
        mixed_path = self.root / "mixed.jsonl"
        write_jsonl(mixed_path, mixed)
        grpo_path = self.root / "grpo.jsonl"
        grpo_summary = build_grpo_dataset(mixed_path, grpo_path)
        self.assertEqual(grpo_summary["tasks"], 22)
        self.assertEqual(grpo_summary["rollouts"], 44)
        self.assertTrue(all(row["split"] != "test" for row in load_jsonl(grpo_path)))

    def test_rl_gate_reports_each_condition(self) -> None:
        paths = load_task_paths(self.manifest)
        teacher_path = self.root / "teacher.jsonl"
        collect_rollouts(paths, lambda task, trial: ReferenceAgent(), teacher_path)
        rows = load_jsonl(teacher_path)
        duplicated = []
        for row in rows:
            good = json.loads(json.dumps(row, ensure_ascii=False))
            bad = json.loads(json.dumps(row, ensure_ascii=False))
            good["trial"] = 0
            bad["trial"] = 1
            bad["reward"] = 0.0
            bad["verifier"]["passed"] = False
            bad["verifier"]["failures"] = ["state_value:orders.status:synthetic"]
            duplicated.extend([good, bad])
        critical = {
            "verify_customer_identity",
            "get_order",
            "search_refund_policy",
            "update_shipping_address",
            "cancel_order",
            "create_refund",
            "escalate_to_human",
        }
        report = assess_rl_gate(duplicated, [0.50, 0.51, 0.51], critical_tools=critical)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))

    def test_grpo_advantage_gate_keeps_only_mixed_non_test_groups(self) -> None:
        rows = [
            {"task_id": "mixed", "split": "train", "trial": 0, "reward": 1.0},
            {"task_id": "mixed", "split": "train", "trial": 1, "reward": 0.0},
            {"task_id": "all-good", "split": "train", "trial": 0, "reward": 1.0},
            {"task_id": "all-good", "split": "train", "trial": 1, "reward": 1.0},
        ]
        selected, summary = compute_group_advantages(rows)
        self.assertEqual(len(selected), 2)
        self.assertEqual(summary["groups"], 2)
        self.assertEqual(summary["mixed_groups"], 1)
        self.assertAlmostEqual(sum(row["advantage"] for row in selected), 0.0)
        self.assertNotIn("advantage", rows[0])
        with self.assertRaises(ValueError):
            compute_group_advantages([{"task_id": "heldout", "split": "test", "reward": 0.0}])

    def test_generated_sft_rows_have_valid_tool_transcripts(self) -> None:
        paths = load_task_paths(self.manifest)
        teacher_path = self.root / "teacher-for-validation.jsonl"
        collect_rollouts(paths, lambda task, trial: ReferenceAgent(), teacher_path)
        build_sft_dataset(teacher_path, self.root / "validated-sft")
        rows = load_jsonl(self.root / "validated-sft" / "train.jsonl")
        self.assertTrue(rows)
        self.assertTrue(all(not validate_training_row(row) for row in rows))

    def test_training_normalizes_tool_arguments_to_objects(self) -> None:
        source = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "verify_customer_identity",
                            "arguments": '{"name":"刘敏","phone_last4":"2684"}',
                        },
                    }
                ],
            }
        ]
        normalized = normalize_messages(source)
        arguments = normalized[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments, {"name": "刘敏", "phone_last4": "2684"})
        self.assertIsInstance(arguments, dict)
        self.assertIsInstance(source[0]["tool_calls"][0]["function"]["arguments"], str)

        invalid = json.loads(json.dumps(source, ensure_ascii=False))
        invalid[0]["tool_calls"][0]["function"]["arguments"] = "[]"
        with self.assertRaisesRegex(ValueError, "tool_arguments_must_be_object"):
            normalize_messages(invalid)

    def test_adapter_checkpoints_save_base_model_configs(self) -> None:
        class SavedArtifact:
            def __init__(self, filename: str, value: str) -> None:
                self.filename = filename
                self.value = value

            def save_pretrained(self, path: Path) -> None:
                Path(path, self.filename).write_text(self.value, encoding="utf-8")

        class BaseModel:
            config = SavedArtifact("config.json", "base-config")
            generation_config = SavedArtifact("generation_config.json", "generation-config")

        class AdapterModel:
            config = SavedArtifact("adapter_config.json", "adapter-config")

            @staticmethod
            def get_base_model() -> BaseModel:
                return BaseModel()

        target = self.root / "checkpoint"
        target.mkdir()
        save_base_model_configs(AdapterModel(), target)
        self.assertEqual((target / "config.json").read_text(encoding="utf-8"), "base-config")
        self.assertEqual(
            (target / "generation_config.json").read_text(encoding="utf-8"),
            "generation-config",
        )
        self.assertFalse((target / "adapter_config.json").exists())

    def test_trajectory_scoped_sft_declares_only_executed_tools(self) -> None:
        paths = load_task_paths(self.manifest)
        teacher_path = self.root / "trajectory-teacher.jsonl"
        collect_rollouts(paths, lambda task, trial: ReferenceAgent(), teacher_path)
        output = self.root / "trajectory-sft"
        summary = build_sft_dataset(teacher_path, output, tool_scope="trajectory")
        self.assertEqual(summary["tool_scope"], "trajectory")
        rows = load_jsonl(output / "train.jsonl") + load_jsonl(output / "dev.jsonl")
        for row in rows:
            declared = {item["function"]["name"] for item in row["tools"]}
            executed = {
                (call.get("function") or {}).get("name")
                for message in row["messages"]
                for call in message.get("tool_calls") or []
            }
            self.assertEqual(declared, executed)

    def test_stateful_turn_sft_matches_runtime_tool_visibility(self) -> None:
        paths = load_task_paths(self.manifest)
        teacher_path = self.root / "stateful-turn-teacher.jsonl"
        collect_rollouts(paths, lambda task, trial: ReferenceAgent(), teacher_path)
        output = self.root / "stateful-turn-sft"
        summary = build_sft_dataset(teacher_path, output, tool_scope="stateful_turn")
        self.assertEqual(summary["tool_scope"], "stateful_turn")
        self.assertGreater(summary["train"], 15)

        rows = load_jsonl(output / "train.jsonl") + load_jsonl(output / "dev.jsonl")
        self.assertTrue(all(sum(row["loss_mask"]) == 1 for row in rows))
        self.assertTrue(all(not validate_training_row(row) for row in rows))
        for row in rows:
            target = next(
                message
                for message, selected in zip(row["messages"], row["loss_mask"])
                if selected
            )
            visible = {item["function"]["name"] for item in row["tools"]}
            requested = {
                (call.get("function") or call).get("name")
                for call in target.get("tool_calls") or []
            }
            self.assertTrue(requested.issubset(visible))

        shipment = [row for row in rows if row["scenario"] == "shipment_status"]
        verify_turn = next(
            row
            for row in shipment
            if any(
                (call.get("function") or call).get("name") == "verify_customer_identity"
                for message in row["messages"]
                for call in message.get("tool_calls") or []
                if row["loss_mask"][row["messages"].index(message)]
            )
        )
        shipment_turn = next(
            row
            for row in shipment
            if any(
                (call.get("function") or call).get("name") == "get_shipment"
                for message in row["messages"]
                for call in message.get("tool_calls") or []
                if row["loss_mask"][row["messages"].index(message)]
            )
        )
        self.assertEqual(
            {item["function"]["name"] for item in verify_turn["tools"]},
            {"verify_customer_identity"},
        )
        self.assertEqual(
            {item["function"]["name"] for item in shipment_turn["tools"]},
            {"list_orders", "get_order", "get_shipment", "search_refund_policy"},
        )

    def test_deeper_tasks_cover_partial_refund_disambiguation_and_negative_confirmation(self) -> None:
        tasks = [TaskSpec.from_file(path) for path in load_task_paths(self.manifest)]
        by_scenario = {task.scenario: task for task in tasks}

        partial = run_task(by_scenario["partial_quantity_refund"], ReferenceAgent())
        self.assertEqual(partial.reward, 1.0)
        refund = next(item for item in partial.tool_executions if item["name"] == "create_refund")
        self.assertEqual(refund["arguments"]["quantity"], 1)
        self.assertEqual(refund["output"]["remaining_quantity"], 1)

        disambiguation = run_task(by_scenario["multi_order_disambiguation"], ReferenceAgent())
        self.assertEqual(disambiguation.reward, 1.0)
        self.assertEqual(
            [item["name"] for item in disambiguation.tool_executions],
            ["verify_customer_identity", "list_orders", "get_order", "cancel_order"],
        )

        negative = run_task(by_scenario["negative_confirmation"], ReferenceAgent())
        self.assertEqual(negative.reward, 1.0)
        self.assertNotIn("create_refund", {item["name"] for item in negative.tool_executions})

        exceeds = by_scenario["refund_quantity_exceeds"]
        self.assertEqual(exceeds.required_tools, ("verify_customer_identity", "get_order"))

    def test_lightweight_demo_lists_and_runs_isolated_tasks(self) -> None:
        app = DemoApplication(self.root / "web-demo", replicas=1, seed=7)
        tasks = app.list_tasks()
        self.assertEqual(len(tasks), 30)
        selected = next(row for row in tasks if row["scenario"] == "multi_order_disambiguation")
        rollout = app.run(selected["task_id"], "reference")
        self.assertEqual(rollout["reward"], 1.0)
        self.assertTrue(rollout["verifier"]["passed"])
        self.assertGreaterEqual(len(rollout["tool_executions"]), 4)

        second = app.run(selected["task_id"], "reference")
        self.assertEqual(second["reward"], 1.0)

    def test_thirty_template_expansion_covers_recovery_idempotence_and_parameter_binding(self) -> None:
        tasks = [TaskSpec.from_file(path) for path in load_task_paths(self.manifest)]
        by_scenario = {task.scenario: task for task in tasks}

        full = run_task(by_scenario["full_quantity_refund"], ReferenceAgent())
        self.assertEqual(full.reward, 1.0)
        full_refund = next(item for item in full.tool_executions if item["name"] == "create_refund")
        self.assertEqual(full_refund["arguments"]["quantity"], 2)
        self.assertEqual(full_refund["output"]["remaining_quantity"], 0)

        identity_retry = run_task(by_scenario["identity_tool_transient_error"], ReferenceAgent())
        identity_calls = [
            item for item in identity_retry.tool_executions if item["name"] == "verify_customer_identity"
        ]
        self.assertEqual(identity_retry.reward, 1.0)
        self.assertEqual([item["ok"] for item in identity_calls], [False, True])

        mismatch = run_task(by_scenario["cross_order_item_mismatch"], ReferenceAgent())
        self.assertEqual(mismatch.reward, 1.0)
        self.assertNotIn("create_refund", {item["name"] for item in mismatch.tool_executions})

        bulk = run_task(by_scenario["mixed_order_bulk_cancel"], ReferenceAgent())
        self.assertEqual(bulk.reward, 1.0)
        self.assertEqual(sum(item["name"] == "cancel_order" for item in bulk.tool_executions), 1)

    def test_online_environment_uses_fresh_state_and_verifier_reward(self) -> None:
        online_dataset = self.root / "online-dataset"
        build_task_dataset(online_dataset, replicas=2, seed=23)
        task_path = next(
            path
            for path in load_task_paths(online_dataset / "tasks.jsonl", split="train")
            if TaskSpec.from_file(path).scenario == "cancel_unshipped"
            and not TaskSpec.from_file(path).metadata.get("scripted_user_replies")
        )
        task = TaskSpec.from_file(task_path)
        rows, summary = build_online_rows([task_path])
        self.assertEqual(summary["eligible_tasks"], 1)
        self.assertEqual(rows[0]["environment"], task.task_id)

        environment = EcommerceCustomerServiceEnv(task_path=task_path)
        environment.reset()
        identity = environment.verify_customer_identity(**task.metadata["identity"])
        customer_id = identity["customer_id"]
        environment.get_order(customer_id, task.metadata["order_id"])
        environment.cancel_order(customer_id, task.metadata["order_id"], True)
        amount = int(task.metadata["paid_amount_cents"]) / 100
        self.assertEqual(environment._evaluate(f"订单已取消，{amount:.2f} 元将原路退款。"), 1.0)

        environment.reset()
        self.assertEqual(environment._evaluate("订单已取消。"), 0.0)

    def test_audit_shaping_never_compensates_for_policy_failure(self) -> None:
        unsafe = VerificationResult(
            passed=False,
            state_passed=True,
            communication_passed=True,
            policy_passed=False,
            required_tools_passed=True,
        )
        breakdown = score_verification(unsafe)
        self.assertEqual(breakdown.strict_binary, 0.0)
        self.assertEqual(breakdown.audit_shaped, 0.0)

        matrix = json.loads((PROJECT_ROOT / "configs" / "algorithm-experiments.json").read_text(encoding="utf-8"))
        methods = {row["method"] for row in matrix["experiments"]}
        self.assertIn("online_environment_grpo", methods)
        self.assertIn("offline_group_advantage_weighted_regression", methods)


if __name__ == "__main__":
    unittest.main()
