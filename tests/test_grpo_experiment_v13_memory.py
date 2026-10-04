"""Synthetic CUDA counters and qualification evidence, never real GPU claims."""
from copy import deepcopy
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
from ecommerce_customer_service.grpo_memory_v13 import CudaMemoryAudit, GIB, MEMORY_POLICY, summarize_memory
from ecommerce_customer_service.grpo_experiment_v13 import validate_qualification
from ecommerce_customer_service.local_hf import sha256_file
import train_grpo_experiment_v13 as runner


def measurements(steps=10):
    events = [("initialization", "begin", None), ("initialization", "end", None), ("training", "begin", None)]
    for step in range(1, steps + 1):
        events.extend((("training", "reward", step), ("training", "step_end", step)))
    events.extend((('training', 'end', None), ('saving', 'begin', None), ('saving', 'end', None),
                   ('reload', 'begin', None), ('reload', 'end', None)))
    return [{"event": "V13_MEMORY_SAMPLE", "policy": MEMORY_POLICY, "stage": stage, "point": point,
             "step": step, "cache_released": point in ("begin", "end"),
             "total_bytes": 24 * GIB, "driver_free_bytes": 4 * GIB,
             "allocated_bytes": 18 * GIB, "reserved_bytes": 19 * GIB,
             "peak_allocated_bytes": 20 * GIB, "peak_reserved_bytes": 22 * GIB,
             "non_allocator_used_bytes": GIB, "peak_inactive_split_bytes": GIB // 2}
            for stage, point, step in events]


class MemoryPolicyTest(unittest.TestCase):
    def test_reserved_upper_bound_keeps_one_gib_margin(self):
        report = summarize_memory(measurements(), 10)
        self.assertTrue(report["memory_safety_passed"])
        self.assertEqual(report["memory_headroom_gib"], 1.0)
        self.assertEqual(report["peak_reserved_gib"], 22.0)
        self.assertTrue(report["capacity_bound_is_conservative_not_simultaneous_peak"])

    def test_inactive_split_is_reported_but_not_double_counted(self):
        rows = measurements()
        rows[3]["peak_inactive_split_bytes"] = 8 * GIB
        report = summarize_memory(rows, 10)
        self.assertTrue(report["memory_safety_passed"])
        self.assertEqual(report["memory_headroom_gib"], 1.0)
        self.assertEqual(report["stages"]["training"]["peak_inactive_split_bytes"], 8 * GIB)
        self.assertTrue(report["fragmentation_diagnostic_only_already_in_reserved"])
        self.assertNotIn("live_demand_headroom_bytes", report["stages"]["training"])

    def test_real_reserved_peak_and_external_usage_cannot_be_waived(self):
        for field, value in (("peak_reserved_bytes", 23 * GIB), ("peak_reserved_bytes", 24 * GIB - 1)):
            rows = measurements()
            rows[3][field] = value
            report = summarize_memory(rows, 10)
            self.assertFalse(report["memory_safety_passed"])
        rows = measurements()
        rows[3].update(driver_free_bytes=GIB // 2, non_allocator_used_bytes=4 * GIB + GIB // 2)
        self.assertFalse(summarize_memory(rows, 10)["memory_safety_passed"])

    def test_device_free_after_cache_release_is_an_independent_required_margin(self):
        rows = measurements()
        rows[-1].update(reserved_bytes=23 * GIB, peak_reserved_bytes=23 * GIB, driver_free_bytes=GIB // 2,
                        non_allocator_used_bytes=GIB // 2)
        self.assertFalse(summarize_memory(rows, 10)["memory_safety_passed"])

    def test_each_stage_is_reported_and_reload_can_independently_block(self):
        rows = measurements()
        rows[-1].update(peak_allocated_bytes=23 * GIB, peak_reserved_bytes=23 * GIB)
        report = summarize_memory(rows, 10)
        self.assertGreaterEqual(report["stages"]["training"]["reserved_upper_bound_headroom_bytes"], GIB)
        self.assertLess(report["stages"]["reload"]["reserved_upper_bound_headroom_bytes"], GIB)
        self.assertFalse(report["memory_safety_passed"])

    def test_external_usage_is_stage_specific_not_borrowed_from_another_stage(self):
        rows = measurements()
        rows[0].update(driver_free_bytes=3 * GIB, non_allocator_used_bytes=2 * GIB)
        report = summarize_memory(rows, 10)
        self.assertEqual(report["stages"]["initialization"]["peak_observed_non_allocator_bytes"], 2 * GIB)
        self.assertEqual(report["stages"]["training"]["peak_observed_non_allocator_bytes"], GIB)
        self.assertEqual(report["stages"]["training"]["reserved_upper_bound_headroom_bytes"], GIB)
        self.assertFalse(report["memory_safety_passed"])

    def test_exact_one_gib_boundary_passes_and_one_byte_less_fails(self):
        rows = measurements()
        self.assertTrue(summarize_memory(rows, 10)["memory_safety_passed"])
        rows[3]["peak_reserved_bytes"] += 1
        self.assertFalse(summarize_memory(rows, 10)["memory_safety_passed"])

    def test_mid_generation_driver_free_is_recorded_not_only_boundary_free(self):
        rows = measurements()
        rows[3].update(driver_free_bytes=GIB // 2, non_allocator_used_bytes=4 * GIB + GIB // 2)
        report = summarize_memory(rows, 10)
        self.assertEqual(report["stages"]["training"]["min_observed_driver_free_bytes"], GIB // 2)
        self.assertEqual(report["stages"]["training"]["boundary_driver_free_bytes"], 4 * GIB)
        self.assertFalse(report["memory_safety_passed"])

    def test_old_r3_policy_cannot_authorize_new_training(self):
        rows = measurements()
        for row in rows:
            row["policy"] = "stage_live_external_and_driver_free_v1"
        with self.assertRaisesRegex(ValueError, "unknown_memory_policy"):
            summarize_memory(rows, 10)

    def test_missing_events_duplicate_steps_wrong_policy_and_missing_cache_cleanup_rejected(self):
        for mutation in (lambda r: r.pop(), lambda r: r[3].update(step=2),
                         lambda r: r[0].update(policy="legacy"), lambda r: r[0].update(cache_released=False),
                         lambda r: r[0].update(stage="other"), lambda r: r[0].update(total_bytes=25 * GIB)):
            rows = measurements()
            mutation(rows)
            with self.assertRaises(ValueError):
                summarize_memory(rows, 10)

    def test_bad_or_inconsistent_counter_cannot_pass(self):
        for field, value in (("total_bytes", 0), ("driver_free_bytes", float("nan")),
                             ("reserved_bytes", -1), ("peak_allocated_bytes", True),
                             ("non_allocator_used_bytes", 0), ("peak_reserved_bytes", 25 * GIB),
                             ("peak_inactive_split_bytes", 25 * GIB)):
            rows = measurements()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                summarize_memory(rows, 10)

    def test_experiment_requires_its_own_hundred_step_memory_measurements(self):
        self.assertTrue(summarize_memory(measurements(100), 100)["memory_safety_passed"])
        with self.assertRaises(ValueError):
            summarize_memory(measurements(), 100)

    def test_real_qualification_requires_bound_memory_artifacts_not_edited_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            context = {"memory_policy": MEMORY_POLICY}
            data = measurements()
            report = summarize_memory(data, 10)
            (path / "preflight.json").write_text(json.dumps({**context, "phase": "qualification", "max_steps": 10,
                                                           "task_ids": ["synthetic"], "source_sha256": {}}))
            (path / "memory_audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            (path / "memory_report.json").write_text(json.dumps(report))
            (path / "gradient_audit.jsonl").write_text("".join(json.dumps({"next_optimizer_step": s, "finite": True}) + "\n" for s in range(1, 11)))
            (path / "online_rewards.jsonl").write_text("".join(json.dumps({"task_id": "synthetic", "batch": b,
                "verifier_version": "deterministic_semantic_v13", "split": "train"}) + "\n" for b in range(1, 11) for _ in range(8)))
            for name in ("initialization.json", "adapter_model.safetensors", "ref/adapter_model.safetensors"):
                target = path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"synthetic fixture, never model evidence")
            result = {"complete": True, "phase": "qualification", "optimizer_steps": 10, "nonzero_gradient_steps": 10,
                "actual_policy_update_verified": True, "adapter_save_reload_verified": True, "sft_reference_unchanged": True,
                "source_databases_unchanged": True, "memory_policy": MEMORY_POLICY,
                **{key: report[key] for key in ("memory_headroom_gib", "peak_allocated_gib", "peak_reserved_gib")},
                "artifacts_sha256": {p.relative_to(path).as_posix(): sha256_file(p) for p in path.rglob("*") if p.is_file()}}
            complete = path / "complete.json"
            complete.write_text(json.dumps(result))
            with patch("ecommerce_customer_service.grpo_experiment_v13.replay_row") as replay, patch(
                    "ecommerce_customer_service.grpo_experiment_v13.source_hashes", return_value={}):
                self.assertEqual(validate_qualification(path, context=context, paths={"synthetic": Path("unused")}, root=ROOT)["online_rollouts_replayed"], 80)
                self.assertEqual(replay.call_count, 80)
                # Rehashing an edited report alone cannot bypass recomputation.
                report["memory_headroom_gib"] = 99
                (path / "memory_report.json").write_text(json.dumps(report))
                result["memory_headroom_gib"] = 99
                result["artifacts_sha256"]["memory_report.json"] = sha256_file(path / "memory_report.json")
                complete.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "memory_evidence_mismatch"):
                    validate_qualification(path, context=context, paths={"synthetic": Path("unused")}, root=ROOT)


class CollectorTest(unittest.TestCase):
    def fake_cuda(self):
        calls = []
        cuda = SimpleNamespace(memory=SimpleNamespace(get_allocator_backend=lambda: "native"),
            synchronize=lambda: calls.append("sync"), empty_cache=lambda: calls.append("empty"),
            reset_peak_memory_stats=lambda: calls.append("reset"), mem_get_info=lambda: (4 * GIB, 24 * GIB),
            memory_allocated=lambda: 18 * GIB, memory_reserved=lambda: 19 * GIB,
            max_memory_allocated=lambda: 20 * GIB, max_memory_reserved=lambda: 23 * GIB,
            memory_stats=lambda: {"inactive_split_bytes.all.peak": GIB // 2})
        return cuda, calls

    def test_stage_begin_resets_only_after_cleanup_and_records_durable_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda, calls = self.fake_cuda()
            path = Path(directory) / "memory.jsonl"
            audit = CudaMemoryAudit(cuda, path)
            audit.begin("training")
            audit.sample("reward", step=1)
            audit.end()
            self.assertLess(calls.index("empty"), calls.index("reset"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["point"] for r in rows], ["begin", "reward", "end"])
            self.assertEqual([r["cache_released"] for r in rows], [True, False, True])

    def test_unsupported_backend_and_missing_fragmentation_counter_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda, _ = self.fake_cuda()
            cuda.memory.get_allocator_backend = lambda: "cudaMallocAsync"
            with self.assertRaisesRegex(RuntimeError, "native_cuda_allocator"):
                CudaMemoryAudit(cuda, Path(directory) / "a.jsonl")
            cuda.memory.get_allocator_backend = lambda: "native"
            cuda.memory_stats = lambda: {}
            audit = CudaMemoryAudit(cuda, Path(directory) / "b.jsonl")
            with self.assertRaisesRegex(RuntimeError, "fragmentation_measurement_missing"):
                audit.begin("training")

    def test_configuration_and_reward_rules_unchanged(self):
        self.assertEqual(runner.training_config()["generation_batch_size"], 8)
        self.assertEqual(runner.training_config()["max_completion_length"], 2048)
        self.assertEqual(runner.training_config()["num_generations"], 4)


if __name__ == "__main__":
    unittest.main()
