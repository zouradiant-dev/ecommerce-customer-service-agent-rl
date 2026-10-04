"""Exercise new-report guards with isolated synthetic fixtures, never GPU jobs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.data import load_jsonl, write_jsonl
from ecommerce_customer_service.database import create_database
from ecommerce_customer_service.evaluation import evaluate_rollouts
from ecommerce_customer_service.grpo_readiness import sampling_checks, select_train_tasks
from ecommerce_customer_service.local_hf import sha256_file
from ecommerce_customer_service.online_env import EcommerceCustomerServiceEnv
from ecommerce_customer_service.tasks import _task_for_template
from ecommerce_customer_service.verifier import VERIFIER_VERSION, V13_VERIFIER_VERSION

spec = importlib.util.spec_from_file_location("v13_rescore", ROOT / "scripts/rescore_reward_v13.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ImmutableReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v13-replay-fixture-")
        self.directory = Path(self.temp.name)
        fixture_dir = self.directory / "dataset/tasks/cancel_without_captured_payment_001"
        database = create_database(fixture_dir / "state.sqlite")
        task = _task_for_template("cancel_without_captured_payment", 1, database, seed=91)
        (fixture_dir / "task.json").write_text(json.dumps(task.to_dict(), ensure_ascii=False), encoding="utf-8")
        self.manifest = self.directory / "dataset/tasks.jsonl"
        write_jsonl(self.manifest, [{"task_id": task.task_id, "split": "train",
                                   "task_file": "tasks/cancel_without_captured_payment_001/task.json"}])
        path, task = select_train_tasks(self.manifest, templates=("cancel_without_captured_payment",))[0]
        self.task = task
        self.source = self.directory / "source"
        self.source.mkdir()
        env = EcommerceCustomerServiceEnv(task_path=path)
        try:
            env.reset()
            identity = env.verify_customer_identity(**task.metadata["identity"])
            env.get_order(customer_id=identity["customer_id"], order_id=task.metadata["order_id"])
            result = env.cancel_order(customer_id=identity["customer_id"], order_id=task.metadata["order_id"], confirmed=True)
            self.assertTrue(result["ok"])
            self.assertIsNone(result["refund_id"])
            final = f"订单 {task.metadata['order_id']} 已取消。未实际扣款，无需退款；refund_id 为 null。"
            reward = env._evaluate(final)
            self.assertEqual(reward, 0)
            row = {"task_id": task.task_id, "split": "train", "scenario": task.scenario,
                   "trial": 0, "reward": reward, "final_response": final,
                   "messages": [{"role": "user", "content": task.user_prompt}],
                   "tool_executions": [call.to_dict() for call in env._tools.executions],
                   "verifier": json.loads(json.dumps(env._last_verification.to_dict())),
                   "verifier_version": VERIFIER_VERSION, "stopped_reason": "model_finished", "error": None,
                   "synthetic_fixture_not_readiness_evidence": True}
        finally:
            env._cleanup()
        self.rows = [{**deepcopy(row), "trial": trial} for trial in range(4)]
        self.protocol = {"split": "train", "verifier_version": VERIFIER_VERSION, "rollouts": 4,
                         "trials_per_task": 4, "tasks": 1, "task_ids": [task.task_id],
                         "adapter_sha256": "synthetic_fixture_not_model_weights", "manifest_sha256": sha256_file(self.manifest),
                         "source_database_sha256": {task.task_id: sha256_file(task.database_path)},
                         "source_sha256": {"src/ecommerce_customer_service/verifier.py": replay.FROZEN_V12_VERIFIER_SHA256,
                                           "src/ecommerce_customer_service/communication_facts.py": sha256_file(ROOT / "src/ecommerce_customer_service/communication_facts.py")},
                         "synthetic_fixture_not_readiness_evidence": True}
        self.save_source()

    def tearDown(self):
        self.temp.cleanup()

    def save(self, relative, value):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def save_source(self):
        write_jsonl(self.source / "sft/rollouts.jsonl", self.rows)
        checks = sampling_checks(self.rows, trials_per_task=4, expected_task_ids=self.protocol["task_ids"])
        self.save("protocol.json", self.protocol)
        self.save("model_info.json", {"synthetic_fixture_not_model_loaded": True})
        self.save("summary.json", {"complete": True, "protocol": self.protocol, "sampling_checks": checks})
        self.save("sft/report.json", {"evaluation": evaluate_rollouts(self.rows)})
        self.save("readiness.json", checks)
        self.save("reward_review.json", {"rollouts_sha256": sha256_file(self.source / "sft/rollouts.jsonl"),
                  "rows": [{"task_id": row["task_id"], "trial": row["trial"], "recorded_reward": row["reward"],
                            "expected_reward": 1.0, "status": "synthetic_fixture", "evidence": "known true null fixture"}
                           for row in self.rows]})

    def test_replay_keeps_raw_data_and_generation_provenance_separate(self):
        before = replay.file_hashes(self.source)
        output = self.directory / "new-v13"
        report = replay.run(self.source, self.manifest, output)
        self.assertEqual(report["passed_before"], 0)
        self.assertEqual(report["passed_after"], 4)
        self.assertEqual(before, replay.file_hashes(self.source))
        protocol = replay.load(output / "protocol.json")
        self.assertEqual(protocol["generation_protocol"], self.protocol)
        self.assertEqual(protocol["generation_source_sha256"], self.protocol["source_sha256"])
        self.assertNotEqual(protocol["source_sha256"]["src/ecommerce_customer_service/verifier.py"],
                            protocol["generation_source_sha256"]["src/ecommerce_customer_service/verifier.py"])
        for old, new in zip(self.rows, load_jsonl(output / "sft/rollouts.jsonl")):
            self.assertEqual({key: value for key, value in old.items() if key not in replay.DERIVED_FIELDS},
                             {key: value for key, value in new.items() if key not in replay.DERIVED_FIELDS})
            self.assertEqual(new["verifier_version"], V13_VERIFIER_VERSION)
        self.assertFalse(report["training_started"])
        self.assertFalse(report["reward_calibrated"])
        self.assertIn("engineering_sampling_checks", report["smoke_validator_rejection"])

    def test_existing_nested_and_broad_output_are_rejected_before_changes(self):
        before = replay.file_hashes(self.source)
        for output in (self.source, self.source / "nested", self.directory):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "new_non_nested_output"):
                replay.run(self.source, self.manifest, output)
            self.assertEqual(before, replay.file_hashes(self.source))

    def test_stale_or_forged_v12_reward_is_rejected(self):
        self.rows[0]["final_response"] = "refund_id 为 refund_fake_999。"
        self.save_source()  # Reports and hashes agree, but old tool replay does not.
        output = self.directory / "tampered-replay"
        with self.assertRaisesRegex(ValueError, "frozen_v12_replay_mismatch"):
            replay.run(self.source, self.manifest, output)
        self.assertFalse(output.exists())

    def test_generation_source_version_and_task_database_mismatch_block(self):
        self.protocol["source_sha256"]["src/ecommerce_customer_service/verifier.py"] = "wrong_generation_hash"
        self.save_source()
        output = self.directory / "bad-source"
        with self.assertRaisesRegex(ValueError, "unsupported_generation_verifier_snapshot"):
            replay.run(self.source, self.manifest, output)
        self.assertFalse(output.exists())
        self.protocol["source_sha256"]["src/ecommerce_customer_service/verifier.py"] = replay.FROZEN_V12_VERIFIER_SHA256
        self.protocol["source_database_sha256"][self.task.task_id] = "wrong_database_hash"
        self.save_source()
        with self.assertRaisesRegex(ValueError, "source_database_mismatch"):
            replay.run(self.source, self.manifest, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
