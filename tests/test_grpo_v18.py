"""CPU-only guards for the v18 GRPO interface; no learned-model claims."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ecommerce_customer_service.local_hf import sha256_file  # noqa: E402
from ecommerce_customer_service.verifier_v18 import TRAIN_REWARD_V18  # noqa: E402
import ecommerce_customer_service.grpo_v18 as guards  # noqa: E402
import train_grpo_v18 as runner  # noqa: E402


class TrainSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "tasks.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def make_manifest(self, *, include_test=False):
        rows = []
        for scenario in ("alpha", "beta"):
            for index in range(2):
                directory = self.root / f"{scenario}-{index}"
                directory.mkdir()
                database = directory / "state.sqlite"
                database.write_bytes(f"fixture-{scenario}-{index}".encode())
                task_path = directory / "task.json"
                task = {
                    "task_id": f"v18_train_{scenario}_{index:04d}",
                    "split": "train",
                    "scenario": scenario,
                    "user_id": "cust_fixture",
                    "user_prompt": "fixture",
                    "reference_time": "2026-01-01T00:00:00+00:00",
                    "database": "state.sqlite",
                    "assertions": [],
                    "required_values": [],
                    "required_intents": [],
                    "required_tools": [],
                    "allowed_write_tools": [],
                    "allowed_policy_violations": [],
                    "metadata": {
                        "inline_dialogue_v18": True,
                        "terminal_v18": {"decision": "NO_ACTION", "action_taken": False},
                        "required_tool_sequence_v18": [],
                    },
                }
                task_path.write_text(json.dumps(task), encoding="utf-8")
                rows.append({
                    "task_id": task["task_id"],
                    "split": "train",
                    "scenario": scenario,
                    "task_file": task_path.relative_to(self.root).as_posix(),
                    "task_file_sha256": sha256_file(task_path),
                    "database_sha256": sha256_file(database),
                })
        if include_test:
            rows.append({"task_id": "forbidden", "split": "test"})
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_balanced_selection_is_train_only_and_hash_bound(self):
        self.make_manifest()
        with patch.object(guards, "EXPECTED_MANIFEST_SHA256", sha256_file(self.manifest)):
            selected, plan = guards.select_v18_train_tasks(
                self.manifest,
                tasks_per_scenario=1,
            )
        self.assertEqual([task.task_id for _, task in selected], [
            "v18_train_alpha_0000",
            "v18_train_beta_0000",
        ])
        self.assertEqual(plan["scenario_counts"], {"alpha": 1, "beta": 1})
        self.assertEqual(plan["test_rows_seen"], 0)

    def test_any_test_row_blocks_training_manifest(self):
        self.make_manifest(include_test=True)
        with patch.object(guards, "EXPECTED_MANIFEST_SHA256", sha256_file(self.manifest)):
            with self.assertRaisesRegex(ValueError, "test rows are forbidden"):
                guards.select_v18_train_tasks(self.manifest, tasks_per_scenario=1)


class RunnerGuardTest(unittest.TestCase):
    def parse(self, *extra):
        return runner.build_parser().parse_args([
            "--manifest", "manifest",
            "--adapter", "adapter",
            "--calibration-receipt", "receipt",
            "--calibration-rollouts", "rollouts",
            "--calibration-protocol", "protocol",
            "--output", "output",
            "--phase", "qualification",
            *extra,
        ])

    def test_explicit_bounded_acknowledgement_and_qualification_chain(self):
        with self.assertRaisesRegex(ValueError, "experimental_only"):
            runner.validate_args(self.parse())
        runner.validate_args(self.parse("--experimental-only"))
        args = self.parse("--experimental-only")
        args.phase = "experiment"
        with self.assertRaisesRegex(ValueError, "qualification_required"):
            runner.validate_args(args)
        args.qualification = Path("qualification")
        runner.validate_args(args)

    def test_4090_geometry_and_shaped_reward_are_pinned(self):
        config = runner.training_config()
        self.assertEqual(
            (
                config["per_device_train_batch_size"],
                config["gradient_accumulation_steps"],
                config["generation_batch_size"],
                config["num_generations"],
            ),
            (1, 8, 8, 4),
        )
        self.assertTrue(config["gradient_checkpointing"])
        self.assertFalse(config["use_vllm"])
        self.assertGreater(config["beta"], 0)
        self.assertEqual(TRAIN_REWARD_V18, "train_shaped_v18")

    def fake_environments(self):
        environments = []
        for index in range(8):
            task = SimpleNamespace(
                task_id="task-a" if index < 4 else "task-b",
                split="train",
                scenario="fixture",
                metadata={"terminal_v18": {}},
            )
            environments.append(SimpleNamespace(
                _reward_mode_v18=TRAIN_REWARD_V18,
                _task=task,
                _last_v18_verification=SimpleNamespace(
                    hard_safety_failures=(),
                    to_dict=lambda: {"passed": True, "hard_safety_failures": []},
                ),
                _last_v18_breakdown=SimpleNamespace(
                    to_dict=lambda: {"train_shaped_v18": 1.0},
                ),
                _tools=SimpleNamespace(executions=[]),
            ))
        return environments

    def test_reward_audit_records_mixed_groups_and_stops_no_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "online.jsonl"
            reward = runner.AuditedV18Reward(path, time.monotonic() + 60)
            mixed = [0.2, 0.8, 0.2, 0.8, 1.0, 1.0, 1.0, 1.0]
            with patch.object(runner, "environment_reward_v18", return_value=mixed):
                self.assertEqual(reward(["x"] * 8, self.fake_environments()), mixed)
            self.assertEqual(reward.mixed_groups, 1)
            self.assertEqual(reward.total_groups, 2)
            self.assertEqual(len(guards.read_rows(path)), 8)

            path2 = Path(directory) / "zero.jsonl"
            no_signal = runner.AuditedV18Reward(path2, time.monotonic() + 60)
            with patch.object(runner, "environment_reward_v18", return_value=[1.0] * 8):
                no_signal(["x"] * 8, self.fake_environments())
                no_signal(["x"] * 8, self.fake_environments())
                with self.assertRaisesRegex(RuntimeError, "three_zero_variance"):
                    no_signal(["x"] * 8, self.fake_environments())


class CalibrationBindingTest(unittest.TestCase):
    def test_receipt_hashes_reviewed_rows_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rollouts = directory / "reviewed.jsonl"
            protocol = directory / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            row = {
                "rollout_id": "fixture",
                "task_id": "task",
                "scenario": "scenario",
                "reward_components": {"train_shaped_v18": 1.0},
                "verification": {"safety_gate_passed": True, "communication_passed": True},
                "human_review": {"safe": True, "communication_passed": True},
                "tool_executions": [],
                "stopped_reason": "completed",
                "error": None,
            }
            rollouts.write_text(json.dumps(row) + "\n", encoding="utf-8")
            first = guards.build_calibration_receipt(ROOT, rollouts, protocol)
            self.assertEqual(first["source_sha256"][protocol.resolve().as_posix()], sha256_file(protocol))
            protocol.write_text('{"changed":true}', encoding="utf-8")
            second = guards.build_calibration_receipt(ROOT, rollouts, protocol)
            self.assertNotEqual(first["source_sha256"], second["source_sha256"])


if __name__ == "__main__":
    unittest.main()
