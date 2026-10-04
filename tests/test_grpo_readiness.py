from __future__ import annotations
from copy import deepcopy
from contextlib import redirect_stdout
from datetime import datetime, timezone
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
sys.path.insert(0, str(ROOT / "scripts"))
from ecommerce_customer_service.experiment_prompt import EXPERIMENT_SYSTEM_PROMPT, ONLINE_SYSTEM_PROMPT
from ecommerce_customer_service.grpo_readiness import DEFAULT_TEMPLATES, sampling_checks, select_train_tasks, validate_smoke_evidence
from ecommerce_customer_service.local_hf import LocalTransformersAgent, parse_qwen_response, sha256_file
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.online_env import EcommerceCustomerServiceEnv, build_online_rows, _final_response
from ecommerce_customer_service.simulator import SYSTEM_PROMPT, ReferenceAgent, run_task
from ecommerce_customer_service.tasks import build_task_dataset, load_task_paths
import sample_train_readiness as pilot
import train_grpo as grpo
import install_train_readiness_v12 as installer


class ReadinessTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="grpo-readiness-test-")
        self.root = Path(self.temporary.name)
        build_task_dataset(self.root / "dataset", replicas=2, seed=7)
        self.manifest = self.root / "dataset" / "tasks.jsonl"
        self.selected = select_train_tasks(self.manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def test_selection_is_inline_train_balanced_and_deterministic(self):
        self.assertEqual(len(self.selected), 8)
        self.assertEqual([task.scenario for _, task in self.selected], list(DEFAULT_TEMPLATES))
        self.assertTrue(all(task.split == "train" and not task.metadata.get("scripted_user_replies") for _, task in self.selected))
        self.assertEqual([task.task_id for _, task in self.selected], [task.task_id for _, task in select_train_tasks(self.manifest)])
        with self.assertRaisesRegex(ValueError, "not_enough_inline"):
            select_train_tasks(self.manifest, templates=("late_refund",))
        with self.assertRaises(ValueError):
            select_train_tasks(self.manifest, per_template=2)

    def test_manifest_cannot_relabel_dev_or_test_as_train(self):
        source = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines()]
        for split in ("dev", "test"):
            row = deepcopy(next(row for row in source if row["split"] == split))
            row["split"] = "train"
            bad = self.manifest.with_name(f"bad-{split}.jsonl")
            bad.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest_task_mismatch"):
                select_train_tasks(bad)
            with self.assertRaisesRegex(ValueError, "only train"):
                build_online_rows([(bad.parent / row["task_file"]).resolve()])

    def test_prompt_opt_in_preserves_legacy_and_records_actual_new_prompt(self):
        task = self.selected[0][1]
        original_hash = sha256_file(task.database_path)
        for supplied in (None, EXPERIMENT_SYSTEM_PROMPT):
            captured = []
            def complete(messages, tools):
                captured.extend(messages)
                return parse_qwen_response("请稍候。")
            result = run_task(task, LocalTransformersAgent(complete, system_prompt=supplied))
            self.assertEqual(result.messages[0]["content"], SYSTEM_PROMPT if supplied is None else supplied)
            self.assertEqual(captured[0]["content"], result.messages[0]["content"])
        self.assertEqual(sha256_file(task.database_path), original_hash)
        self.assertIn("不确定或要求先不要操作", SYSTEM_PROMPT)
        self.assertNotIn("不确定或要求先不要操作", EXPERIMENT_SYSTEM_PROMPT)
        self.assertIn("pending 或 paid", EXPERIMENT_SYSTEM_PROMPT)
        self.assertIn("本次未执行、已暂停", EXPERIMENT_SYSTEM_PROMPT)
        self.assertNotIn("写工具在确认前会暂时隐藏", ONLINE_SYSTEM_PROMPT)

    def mock_rows(self):
        rows = []
        for _, task in self.selected:
            good = run_task(task, ReferenceAgent(), trial=0).to_dict()
            self.assertEqual(good["reward"], 1.0, good["verifier"])
            bad = run_task(task, LocalTransformersAgent(lambda *a: parse_qwen_response("请稍候。")), trial=1).to_dict()
            rows.extend([good, bad])
        return rows

    def test_sampling_requires_real_writes_not_merely_state_policy_100pct(self):
        rows = self.mock_rows()
        checks = sampling_checks(rows, trials_per_task=2, expected_task_ids=[task.task_id for _, task in self.selected])
        self.assertTrue(checks["engineering_sampling_checks_passed"], checks)
        self.assertFalse(checks["formal_rl_gate_passed"])
        self.assertFalse(checks["reward_calibrated"])
        no_writes = deepcopy(rows)
        for row in no_writes:
            row["tool_executions"] = [call for call in row["tool_executions"] if not call["is_write"]]
        checks = sampling_checks(no_writes, trials_per_task=2, expected_task_ids=[task.task_id for _, task in self.selected])
        self.assertFalse(checks["checks"]["positive_write_coverage"])
        rows[0]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "actual train"):
            sampling_checks(rows, trials_per_task=2, expected_task_ids=[])

    def test_duplicate_trial_invalid_json_and_all_equal_rewards_block(self):
        rows = self.mock_rows()
        ids = [task.task_id for _, task in self.selected]
        rows.append(deepcopy(rows[0]))
        self.assertFalse(sampling_checks(rows, trials_per_task=2, expected_task_ids=ids)["checks"]["complete_train_groups"])
        rows.pop()
        rows[1]["stopped_reason"] = "invalid_tool_call"
        self.assertFalse(sampling_checks(rows, trials_per_task=2, expected_task_ids=ids)["checks"]["no_invalid_calls_or_truncation"])
        for row in rows:
            row["reward"] = 0.0
            row["verifier"]["passed"] = False
        self.assertFalse(sampling_checks(rows, trials_per_task=2, expected_task_ids=ids)["checks"]["mixed_groups_at_least_20pct"])

    def test_review_pending_disputed_missing_or_tampered_blocks_smoke(self):
        directory = self.root / "pilot"
        (directory / "sft").mkdir(parents=True)
        rows = self.mock_rows()
        path = directory / "sft" / "rollouts.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        protocol = {"split": "train", "adapter_sha256": "adapter", "manifest_sha256": "manifest",
                    "trials_per_task": 2, "task_ids": [task.task_id for _, task in self.selected]}
        checks = sampling_checks(rows, trials_per_task=2, expected_task_ids=protocol["task_ids"])
        review = {"reviewer": None, "reviewed_at_utc": None, "rollouts_sha256": sha256_file(path),
                  "rows": [{"task_id": row["task_id"], "trial": row["trial"], "status": "confirmed",
                            "recorded_reward": row["reward"], "expected_reward": row["reward"], "evidence": "test fixture"} for row in rows]}
        for name, value in (("protocol.json", protocol), ("summary.json", {"complete": True}), ("readiness.json", checks), ("reward_review.json", review)):
            (directory / name).write_text(json.dumps(value), encoding="utf-8")
        def check():
            return validate_smoke_evidence(directory, adapter_sha256="adapter", manifest_sha256="manifest")
        with self.assertRaisesRegex(ValueError, "pending"):
            check()
        review["reviewer"], review["reviewed_at_utc"] = "unit-test", datetime.now(timezone.utc).isoformat()
        (directory / "reward_review.json").write_text(json.dumps(review), encoding="utf-8")
        self.assertFalse(check()["formal_rl_gate_passed"])
        review["rows"][0]["expected_reward"] = 0
        (directory / "reward_review.json").write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "disputed"):
            check()
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            check()

    def make_adapter(self):
        path = self.root / "adapter"
        path.mkdir()
        (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct", "bias": "none", "r": 16}), encoding="utf-8")
        header = json.dumps({"base.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        (path / "adapter_model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (path / name).write_text("{}", encoding="utf-8")
        return path

    def pilot_args(self):
        return SimpleNamespace(manifest=self.manifest, adapter=self.make_adapter(), output=self.root / "out",
                               base_model="Qwen/Qwen2.5-7B-Instruct", templates=list(DEFAULT_TEMPLATES),
                               tasks_per_template=1, trials_per_task=2, temperature=0.4, max_steps=16,
                               max_new_tokens=512, max_input_tokens=8192, seed=7, device=0,
                               allow_download=False, preflight_only=False)

    def test_pilot_preflight_is_cpu_only_without_creating_outputs(self):
        args = self.pilot_args()
        args.preflight_only = True
        with patch.object(pilot, "parse_args", return_value=args), patch.object(pilot, "load_backend") as load, redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main(), 0)
            load.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_pilot_only_loads_sft_and_preserves_existing_output(self):
        args = self.pilot_args()
        class FakeCompleter:
            def seed(self, seed):
                pass
            def __call__(self, messages, tools):
                self.assert_prompt = messages[0]["content"]
                return parse_qwen_response("请稍候。")
        completer = FakeCompleter()
        with patch.object(pilot, "parse_args", return_value=args), patch.object(pilot, "load_backend", return_value=(object(), completer, {"mock": True})), redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main(), 0)
            with self.assertRaises(FileExistsError):
                pilot.main()
        self.assertEqual(completer.assert_prompt, EXPERIMENT_SYSTEM_PROMPT)
        summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["protocol"]["split"], "train")
        self.assertFalse((args.output / "base").exists())
        review = json.loads((args.output / "reward_review.json").read_text(encoding="utf-8"))
        self.assertTrue(all(row["status"] == "pending" for row in review["rows"]))

    def test_online_group_isolation_zero_signal_stop_and_premature_write_denied(self):
        path, task = self.selected[0]
        before = sha256_file(task.database_path)
        envs = [EcommerceCustomerServiceEnv(task_path=path) for _ in range(2)]
        for env in envs:
            env.reset()
        identity = envs[0].verify_customer_identity(**task.metadata["identity"])
        denied = envs[0].cancel_order(identity["customer_id"], task.metadata["order_id"], True)
        self.assertFalse(denied["ok"])
        self.assertEqual(denied["error"], "tool_not_available_in_current_state")
        reward = grpo.AuditedReward(self.root / "online.jsonl", 2)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(reward(["请稍候。"] * 2, envs), [0.0, 0.0])
            self.assertEqual(reward(["请稍候。"] * 2, envs), [0.0, 0.0])
            with self.assertRaisesRegex(RuntimeError, "three_zero_variance"):
                reward(["请稍候。"] * 2, envs)
        self.assertEqual(len((self.root / "online.jsonl").read_text(encoding="utf-8").splitlines()), 6)
        self.assertEqual(sha256_file(task.database_path), before)
        for env in envs:
            env._cleanup()

    def test_incomplete_tool_turn_cannot_be_a_final_answer(self):
        self.assertEqual(_final_response([{"role": "assistant", "content": "订单已取消。", "tool_calls": [{}]}, {"role": "tool", "content": "{}"}]), "")
        self.assertEqual(_final_response([{"role": "assistant", "content": "已完成。"}]), "已完成。")

    def test_grpo_caps_steps_rejects_dev_and_requires_sft_adapter(self):
        parser = grpo.build_parser()
        argv = ["--manifest", str(self.manifest), "--model", "model", "--adapter", "adapter", "--output", "out"]
        with redirect_stdout(io.StringIO()):
            args = parser.parse_args(argv + ["--smoke-only", "--max-steps", "21"])
            with self.assertRaisesRegex(ValueError, "20 optimizer"):
                grpo.validate_args(args)
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--manifest", str(self.manifest), "--output", "out"])
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(argv + ["--split", "dev"])

    def test_grpo_without_review_blocks_before_gpu_import_or_output(self):
        adapter = self.make_adapter()
        output = self.root / "blocked-grpo"
        args = grpo.build_parser().parse_args(["--manifest", str(self.manifest), "--adapter", str(adapter),
            "--output", str(output), "--smoke-only", "--max-steps", "10", "--preflight-only"])
        with patch.object(grpo, "build_parser") as parser, patch.object(grpo, "check_runtime") as runtime, redirect_stdout(io.StringIO()) as captured:
            parser.return_value.parse_args.return_value = args
            self.assertEqual(grpo.main(), 2)
            runtime.assert_not_called()
        self.assertIn("reviewed_train_readiness_dir_required", captured.getvalue())
        self.assertFalse(output.exists())


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="readiness-deploy-test-")
        self.root = Path(self.temporary.name)
        self.repo, self.package = self.root / "repo", self.root / "package"
        (self.repo / "src" / "ecommerce_customer_service").mkdir(parents=True)
        (self.repo / "src" / "ecommerce_customer_service" / "simulator.py").write_bytes(b"old")
        (self.repo / "frozen.py").write_bytes(b"frozen")
        (self.package / "payload" / "src" / "ecommerce_customer_service").mkdir(parents=True)
        (self.package / "payload" / "src" / "ecommerce_customer_service" / "simulator.py").write_bytes(b"new")
        (self.package / "payload" / "new.py").write_bytes(b"new-file")
        self.manifest = {"frozen_files": {"frozen.py": installer.digest(self.repo / "frozen.py")},
            "payload_sha256": {"src/ecommerce_customer_service/simulator.py": installer.digest(self.package / "payload" / "src" / "ecommerce_customer_service" / "simulator.py"),
                               "new.py": installer.digest(self.package / "payload" / "new.py")},
            "previous_sha256": {"src/ecommerce_customer_service/simulator.py": installer.digest(self.repo / "src" / "ecommerce_customer_service" / "simulator.py")}}
        (self.package / "bundle.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_install_backs_up_old_code_preserves_frozen_and_is_idempotent(self):
        result = installer.install(self.package, self.repo)
        self.assertTrue(result["installed"])
        self.assertEqual(result["changed_files"], 2)
        backup = Path(result["backup"])
        self.assertEqual((backup / "src" / "ecommerce_customer_service" / "simulator.py").read_bytes(), b"old")
        self.assertEqual((self.repo / "frozen.py").read_bytes(), b"frozen")
        self.assertTrue(installer.install(self.package, self.repo)["already_installed"])

    def test_unexpected_existing_code_stops_before_any_write(self):
        (self.repo / "src" / "ecommerce_customer_service" / "simulator.py").write_bytes(b"user-work")
        with self.assertRaisesRegex(ValueError, "unexpected existing"):
            installer.install(self.package, self.repo)
        self.assertFalse((self.repo / "new.py").exists())
        self.assertFalse((self.repo / "outputs").exists())

    def test_changed_frozen_code_and_tampered_payload_are_rejected(self):
        (self.package / "payload" / "new.py").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "payload"):
            installer.install(self.package, self.repo)
        self.assertFalse((self.repo / "outputs").exists())
        (self.repo / "frozen.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "frozen v11"):
            installer.install(self.package, self.repo)


if __name__ == "__main__":
    unittest.main()
