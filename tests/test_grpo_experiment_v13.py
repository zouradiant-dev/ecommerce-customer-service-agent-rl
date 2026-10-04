"""CPU-only routing and guard fixtures. No CUDA, optimizer, or human signoff claims."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ecommerce_customer_service.database import create_database
from ecommerce_customer_service.grpo_experiment_v13 import (
    select_experiment_tasks, validate_review, validate_qualification, source_hashes,
)
from ecommerce_customer_service.local_hf import sha256_file
from ecommerce_customer_service.online_env import EcommerceCustomerServiceEnv, build_environment_factories, strict_environment_reward
from ecommerce_customer_service.simulator import AgentRun, run_task
from ecommerce_customer_service.tasks import _task_for_template
from ecommerce_customer_service.verifier import VERIFIER_VERSION, V13_VERIFIER_VERSION
import train_grpo_experiment_v13 as runner
import evaluate_grpo_v13 as evaluator


class OnlineRoutingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        database = create_database(self.directory / "state.sqlite")
        self.task = _task_for_template("cancel_without_captured_payment", 1, database, seed=93)
        self.path = self.directory / "task.json"
        self.path.write_text(json.dumps(self.task.to_dict(), ensure_ascii=False), encoding="utf-8")
        self.final = f"订单 {self.task.metadata['order_id']} 已取消。未实际扣款，无需退款；refund_id 为 null。"
        self.envs = []

    def tearDown(self):
        for env in self.envs:
            env._cleanup()
        self.temp.cleanup()

    def completed_env(self, version):
        env = EcommerceCustomerServiceEnv(task_path=self.path, verifier_version=version)
        self.envs.append(env)
        env.reset()
        task = self.task
        identity = env.verify_customer_identity(**task.metadata["identity"])
        env.get_order(customer_id=identity["customer_id"], order_id=task.metadata["order_id"])
        env.cancel_order(customer_id=identity["customer_id"], order_id=task.metadata["order_id"], confirmed=True)
        return env

    def test_actual_online_reward_uses_constructor_version_not_global(self):
        before = sha256_file(self.task.database_path)
        old, new = self.completed_env(VERIFIER_VERSION), self.completed_env(V13_VERIFIER_VERSION)
        self.assertEqual(strict_environment_reward([self.final, self.final], [old, new]), [0.0, 1.0])
        self.assertEqual(VERIFIER_VERSION, "deterministic_semantic_v12")
        self.assertEqual(sha256_file(self.task.database_path), before)
        self.assertNotEqual(old._task.database_path, new._task.database_path)

    def test_factory_version_survives_reset_and_default_stays_v12(self):
        for kwargs, expected in (({}, VERIFIER_VERSION), ({"verifier_version": V13_VERIFIER_VERSION}, V13_VERIFIER_VERSION)):
            env = build_environment_factories([self.path], **kwargs)[self.task.task_id]()
            self.envs.append(env)
            env.reset()
            self.assertEqual(env._verifier_version, expected)
            env.reset()
            self.assertEqual(env._verifier_version, expected)

    def test_unknown_version_is_rejected_before_rollout(self):
        with self.assertRaisesRegex(ValueError, "unsupported_verifier"):
            EcommerceCustomerServiceEnv(task_path=self.path, verifier_version="arbitrary")

    def test_manual_override_does_not_change_old_environment_default(self):
        env = self.completed_env(VERIFIER_VERSION)
        self.assertEqual(env._evaluate(self.final, verifier_version=V13_VERIFIER_VERSION), 1.0)
        self.assertEqual(env._evaluate(self.final), 0.0)

    def test_no_execution_still_cannot_receive_v13_success(self):
        env = EcommerceCustomerServiceEnv(task_path=self.path, verifier_version=V13_VERIFIER_VERSION)
        self.envs.append(env)
        env.reset()
        self.assertEqual(strict_environment_reward([self.final], [env]), [0.0])
        self.assertFalse(env._last_verification.state_passed)
        self.assertFalse(env._last_verification.required_tools_passed)

    def test_native_dev_runtime_explicit_version_keeps_default_unchanged(self):
        outer = self

        class Agent:
            def run(self, task, tools):
                identity = tools.execute("verify_customer_identity", task.metadata["identity"])
                args = {"customer_id": identity.output["customer_id"], "order_id": task.metadata["order_id"]}
                tools.execute("get_order", args)
                tools.execute("cancel_order", {**args, "confirmed": True})
                return AgentRun((), outer.final, "model_finished")

        old = run_task(self.task, Agent())
        new = run_task(self.task, Agent(), verifier_version=V13_VERIFIER_VERSION)
        self.assertEqual((old.reward, new.reward), (0.0, 1.0))
        self.assertEqual((old.verifier_version, new.verifier_version), (VERIFIER_VERSION, V13_VERIFIER_VERSION))


class ReviewGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.rollouts = self.directory / "synthetic-rows.jsonl"
        self.rollouts.write_text("synthetic test data, not readiness\n", encoding="utf-8")
        self.rows = [{"task_id": "fixture", "trial": index, "reward": float(index)} for index in (0, 1)]
        self.review = {"human_signoff": True, "reviewer": "synthetic unit-test reviewer, NOT real approval",
            "reviewed_at_utc": "2026-10-02T00:00:00+00:00", "verifier_version": V13_VERIFIER_VERSION,
            "rollouts_sha256": sha256_file(self.rollouts),
            "rows": [{**row, "recorded_reward": row["reward"], "expected_reward": row["reward"],
                      "status": "confirmed", "evidence": "synthetic fixture"} for row in self.rows]}
        self.path = self.directory / "review.json"

    def tearDown(self):
        self.temp.cleanup()

    def save(self, review=None):
        self.path.write_text(json.dumps(self.review if review is None else review), encoding="utf-8")

    def test_all_positive_and_negative_review_keys_and_exact_hash_required(self):
        self.save()
        self.assertTrue(validate_review(self.path, self.rows, self.rollouts)["human_review_confirmed"])
        mutations = [lambda r: r.update(human_signoff=False), lambda r: r.update(rollouts_sha256="changed"),
                     lambda r: r.update(verifier_version=VERIFIER_VERSION), lambda r: r["rows"].pop(0),
                     lambda r: r["rows"][0].update(status="pending"), lambda r: r["rows"][0].update(expected_reward=1),
                     lambda r: r["rows"].append(deepcopy(r["rows"][0])), lambda r: r.update(reviewed_at_utc="2026-10-02T00:00:00")]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                review = deepcopy(self.review)
                mutation(review)
                self.save(review)
                with self.assertRaises(ValueError):
                    validate_review(self.path, self.rows, self.rollouts)

    def test_pending_agent_audit_is_not_approval(self):
        self.review["human_signoff"] = False
        self.review["reviewer"] = "Codex agent replay, not human signoff"
        self.save()
        with self.assertRaisesRegex(ValueError, "human_reward_review_required"):
            validate_review(self.path, self.rows, self.rollouts)


class RunnerGuardTest(unittest.TestCase):
    def parse(self, *extra):
        return runner.build_parser().parse_args(["--manifest", "m", "--adapter", "a", "--evidence", "e",
            "--review", "r", "--output", "o", "--phase", "qualification", *extra])

    def test_explicit_bounded_experiment_acknowledgement_required(self):
        with self.assertRaisesRegex(ValueError, "experimental_only"):
            runner.validate_args(self.parse())
        runner.validate_args(self.parse("--experimental-only"))
        for value in ("0", "nan", "7"):
            with self.assertRaises(ValueError):
                runner.validate_args(self.parse("--experimental-only", "--max-hours", value))

    def test_hundred_step_phase_requires_qualification_and_no_step_override_exists(self):
        args = self.parse("--experimental-only")
        args.phase = "experiment"
        with self.assertRaisesRegex(ValueError, "qualification_required"):
            runner.validate_args(args)
        args.qualification = Path("fixture-qualification")
        runner.validate_args(args)
        self.assertNotIn("--max-steps", runner.build_parser().format_help())

    def test_pinned_generation_batch_and_frozen_reference_settings(self):
        config = runner.training_config()
        self.assertEqual((config["per_device_train_batch_size"], config["gradient_accumulation_steps"],
                          config["generation_batch_size"], config["num_generations"]), (1, 8, 8, 4))
        self.assertGreater(config["beta"], 0)
        self.assertTrue(config["mask_truncated_completions"])
        self.assertFalse(config["use_vllm"])

    def test_only_broad_balanced_train_selection_can_expand(self):
        tasks = [(Path("fixture"), SimpleNamespace(task_id=f"family{family}_{index}", scenario=f"family{family}"))
                 for family in range(5) for index in range(10)]
        reviewed = [f"family{family}_0" for family in range(5)]
        self.assertEqual(len(select_experiment_tasks(tasks, reviewed, "qualification")), 5)
        self.assertEqual(len(select_experiment_tasks(tasks, reviewed, "experiment")), 50)
        with self.assertRaisesRegex(ValueError, "broad_balanced"):
            select_experiment_tasks(tasks[:5], reviewed, "experiment")

    def fake_environments(self, version=V13_VERIFIER_VERSION):
        return [SimpleNamespace(_verifier_version=version, _task=SimpleNamespace(task_id="fixture", split="train"),
            _last_verification=SimpleNamespace(to_dict=lambda: {"passed": True}),
            _last_breakdown=SimpleNamespace(to_dict=lambda: {}), _tools=SimpleNamespace(executions=[])) for _ in range(8)]

    def test_reward_audit_stops_zero_variance_without_retry_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rewards.jsonl"
            reward = runner.AuditedV13Reward(path, time.monotonic() + 60)
            with patch.object(runner, "strict_environment_reward", return_value=[1.0] * 8):
                for _ in range(2):
                    reward(["fixture"] * 8, self.fake_environments())
                with self.assertRaisesRegex(RuntimeError, "three_zero_variance"):
                    reward(["fixture"] * 8, self.fake_environments())
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 24)

    def test_reward_audit_rejects_v12_duplicate_environments_and_wrong_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            reward = runner.AuditedV13Reward(Path(directory) / "rewards.jsonl", time.monotonic() + 60)
            with self.assertRaisesRegex(RuntimeError, "version_or_split"):
                reward(["fixture"] * 8, self.fake_environments(VERIFIER_VERSION))
            envs = self.fake_environments()
            with self.assertRaisesRegex(RuntimeError, "isolated"):
                reward(["fixture"] * 8, [envs[0]] * 8)
            envs[1]._task.task_id = "different"
            with patch.object(runner, "strict_environment_reward", return_value=[0., 1.] * 4):
                with self.assertRaisesRegex(RuntimeError, "different_tasks"):
                    reward(["fixture"] * 8, envs)

    def test_reward_time_cap_is_not_silent_success(self):
        with tempfile.TemporaryDirectory() as directory:
            reward = runner.AuditedV13Reward(Path(directory) / "rewards.jsonl", time.monotonic() - 1)
            with patch.object(runner, "strict_environment_reward", return_value=[0., 1.] * 4):
                with self.assertRaisesRegex(RuntimeError, "wall_time_limit"):
                    reward(["fixture"] * 8, self.fake_environments())

    def test_qualification_cannot_be_authorized_by_complete_flag_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            context = {"verifier_version": V13_VERIFIER_VERSION}
            (path / "preflight.json").write_text(json.dumps({**context, "phase": "qualification", "max_steps": 10}))
            (path / "complete.json").write_text(json.dumps({"complete": True, "phase": "qualification", "optimizer_steps": 10}))
            with self.assertRaisesRegex(ValueError, "not_successful"):
                validate_qualification(path, context=context, paths={}, root=ROOT)


class EvaluationGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text("synthetic preflight fixture\n")
        self.sft, self.candidate = self.root / "sft", self.root / "candidate"
        for path in (self.sft, self.candidate):
            path.mkdir()
            (path / "adapter_model.safetensors").write_bytes(b"synthetic guard fixture, not model weights")
        self.database = self.root / "state.sqlite"
        self.database.write_bytes(b"synthetic guard fixture, not business DB")
        self.args = SimpleNamespace(tasks_per_template=2, output=self.root / "new-evaluation", sft=self.sft,
            candidate=[self.candidate], manifest=self.manifest, full_dev=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_dev_preflight_records_same_scoring_prompt_and_no_test_or_training(self):
        task = SimpleNamespace(task_id="dev-fixture", split="dev", database_path=self.database)
        with patch.object(evaluator, "validate_lora_adapter"), patch.object(evaluator, "select_dev_tasks", return_value=[(self.manifest, task)]):
            protocol, selected = evaluator.preflight(self.args)
        self.assertEqual(protocol["verifier_version"], V13_VERIFIER_VERSION)
        self.assertEqual(protocol["temperature"], 0)
        self.assertEqual(protocol["split"], "dev")
        self.assertFalse(protocol["held_out_test_run"])
        self.assertFalse(self.args.output.exists())

    def test_eval_rejects_train_tasks_or_duplicate_adapter_paths(self):
        task = SimpleNamespace(task_id="train-fixture", split="train", database_path=self.database)
        with patch.object(evaluator, "validate_lora_adapter"), patch.object(evaluator, "select_dev_tasks", return_value=[(self.manifest, task)]):
            with self.assertRaisesRegex(ValueError, "dev_only"):
                evaluator.preflight(self.args)
        self.args.candidate = [self.sft]
        with self.assertRaisesRegex(ValueError, "distinct"):
            evaluator.preflight(self.args)

    def test_eval_cannot_overwrite_previous_or_nested_results(self):
        self.args.output = self.sft / "new"
        with self.assertRaisesRegex(ValueError, "new_evaluation"):
            evaluator.preflight(self.args)


if __name__ == "__main__":
    unittest.main()
