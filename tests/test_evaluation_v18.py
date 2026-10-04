"""CPU integration tests for the learned-model v18 evaluation route."""
from __future__ import annotations

import json
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shutil
from contextlib import redirect_stdout
from unittest.mock import patch

from ecommerce_customer_service.evaluation_v18 import (
    audit_tool_call_format,
    run_task_v18,
    select_v18_dev_tasks,
    sha256_file,
    summarize_v18_rollouts,
)
from ecommerce_customer_service.experiment_prompt_v18 import (
    V18_SYSTEM_PROMPT,
    V18_TERMINAL_DECISIONS,
)
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.reference_v18 import _arguments, _final_response
from ecommerce_customer_service.simulator import AgentRun
from ecommerce_customer_service.tasks_v18 import V18_SCENARIOS, build_v18_train_dev_dataset


ROOT = Path(__file__).resolve().parents[1]


class DeterministicV18Agent:
    """Test double that follows the verified controller through the real tools."""

    def run(self, task, tools) -> AgentRun:
        messages: list[dict] = [
            {"role": "system", "content": V18_SYSTEM_PROMPT},
            {"role": "user", "content": task.user_prompt},
        ]
        occurrences: dict[str, int] = {}
        for index, name in enumerate(task.metadata["required_tool_sequence_v18"]):
            occurrences[name] = occurrences.get(name, 0) + 1
            arguments = _arguments(task, name, occurrences[name])
            execution = tools.execute(name, arguments)
            call_id = f"call_{index}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(execution.output, ensure_ascii=False),
                    },
                ]
            )
        calls = [item.to_dict() for item in tools.executions]
        final = _final_response(task, calls)
        messages.append({"role": "assistant", "content": final})
        return AgentRun(tuple(messages), final, "model_finished")


class FakeOpenAICompatibleAgent(DeterministicV18Agent):
    def __init__(self, *_args, **_kwargs) -> None:
        pass


class V18EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="v18-evaluation-")
        cls.dataset = Path(cls.temp.name) / "dataset"
        build_v18_train_dev_dataset(
            cls.dataset,
            train_per_scenario=1,
            dev_per_scenario=1,
            seed=1901,
        )
        cls.manifest = cls.dataset / "tasks.jsonl"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_manifest_selection_is_dev_only_and_scenario_balanced(self) -> None:
        paths, plan = select_v18_dev_tasks(self.manifest, max_tasks_per_scenario=1)
        self.assertEqual(len(paths), len(V18_SCENARIOS))
        self.assertEqual(plan["selected_tasks"], len(V18_SCENARIOS))
        self.assertEqual(set(plan["scenario_counts"]), set(V18_SCENARIOS))
        self.assertTrue(all(TaskSpec.from_file(path).split == "dev" for path in paths))

    def test_manifest_selection_can_target_named_scenarios(self) -> None:
        names = ("contradictory_order", "refund_quantity_exceeds")
        paths, plan = select_v18_dev_tasks(self.manifest, scenarios=names)
        self.assertEqual(len(paths), len(names))
        self.assertEqual(set(plan["scenario_counts"]), set(names))
        self.assertEqual(plan["scenario_filter"], sorted(names))
        with self.assertRaisesRegex(ValueError, "unknown v18 dev scenarios"):
            select_v18_dev_tasks(self.manifest, scenarios=("not_a_real_scenario",))

    def test_prompt_exposes_complete_terminal_decision_vocabulary(self) -> None:
        self.assertNotIn("NO_ACTION", V18_TERMINAL_DECISIONS)
        for decision in V18_TERMINAL_DECISIONS:
            self.assertIn(decision, V18_SYSTEM_PROMPT)
        for required in (
            "ORDER_CONFIRMATION_REQUIRED",
            "ORDER_LOOKUP_COMPLETED",
            "QUANTITY_REJECTED",
            "ITEM_ORDER_MISMATCH",
        ):
            self.assertIn(required, V18_TERMINAL_DECISIONS)

    def test_manifest_with_test_row_is_rejected_before_evaluation(self) -> None:
        rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines()]
        rows[0]["split"] = "test"
        bad_manifest = Path(self.temp.name) / "test-contaminated.jsonl"
        bad_manifest.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "test rows are forbidden"):
            select_v18_dev_tasks(bad_manifest)

    def test_preflight_rejects_missing_dev_sqlite_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v18-missing-db-") as directory:
            dataset = Path(directory) / "dataset"
            shutil.copytree(self.dataset, dataset)
            rows = [json.loads(line) for line in (dataset / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
            row = next(item for item in rows if item["split"] == "dev")
            (dataset / row["database"]).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing v18 dev SQLite"):
                select_v18_dev_tasks(dataset / "tasks.jsonl", max_tasks_per_scenario=1)

    def test_preflight_rejects_corrupted_dev_sqlite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v18-corrupt-db-") as directory:
            dataset = Path(directory) / "dataset"
            shutil.copytree(self.dataset, dataset)
            rows = [json.loads(line) for line in (dataset / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
            row = next(item for item in rows if item["split"] == "dev")
            with (dataset / row["database"]).open("ab") as handle:
                handle.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "frozen_dev_payload_hash_mismatch"):
                select_v18_dev_tasks(dataset / "tasks.jsonl", max_tasks_per_scenario=1)

    def test_reference_controller_exercises_real_v18_evaluator(self) -> None:
        paths, _ = select_v18_dev_tasks(self.manifest, max_tasks_per_scenario=1)
        original_hashes = {path: sha256_file(TaskSpec.from_file(path).database_path) for path in paths}
        rows = [
            run_task_v18(
                TaskSpec.from_file(path),
                DeterministicV18Agent(),
                trial=0,
                stage="reference-test-double",
                model="not-a-learned-model",
            )
            for path in paths
        ]
        summary = summarize_v18_rollouts(
            rows,
            expected_tasks=len(paths),
            trials_per_task=1,
        )
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["overall"]["strict_pass_rate"], 1.0)
        self.assertEqual(summary["overall"]["tool_format_stable_rate"], 1.0)
        self.assertEqual(summary["overall"]["hard_safety_failure_rollouts"], 0)
        self.assertEqual(summary["overall"]["worker_errors"], 0)
        self.assertEqual(summary["overall"]["missing_critical_write_coverage"], [])
        self.assertEqual(set(summary["by_scenario"]), set(V18_SCENARIOS))
        for path, digest in original_hashes.items():
            self.assertEqual(sha256_file(TaskSpec.from_file(path).database_path), digest)

    def test_raw_tool_format_audit_rejects_malformed_arguments(self) -> None:
        result = audit_tool_call_format(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "bad",
                            "type": "function",
                            "function": {
                                "name": "verify_customer_identity",
                                "arguments": "{not-json",
                            },
                        }
                    ],
                }
            ]
        )
        self.assertFalse(result["passed"])
        self.assertIn("arguments_invalid_json", result["failures"][0])

    def test_cli_validate_only_never_contacts_a_model_server(self) -> None:
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_sft_v18.py"),
            "--manifest",
            str(self.manifest),
            "--output",
            str(Path(self.temp.name) / "unused-output"),
            "--stage",
            "final=sft-final",
            "--max-tasks-per-scenario",
            "1",
            "--validate-only",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["protocol"]["task_plan"]["selected_tasks"], len(V18_SCENARIOS))
        self.assertFalse(payload["protocol"]["test_rows_allowed"])

    def test_cli_persists_rows_and_resume_skips_completed_trials(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "evaluate_sft_v18_test_module",
            ROOT / "scripts/evaluate_sft_v18.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OpenAICompatibleAgent = FakeOpenAICompatibleAgent
        output = Path(self.temp.name) / "resume-output"
        arguments = [
            str(ROOT / "scripts/evaluate_sft_v18.py"),
            "--manifest",
            str(self.manifest),
            "--output",
            str(output),
            "--stage",
            "final=sft-final",
            "--max-tasks-per-scenario",
            "1",
            "--progress-every",
            "0",
        ]
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 0)
        rollout_path = output / "final/rollouts.jsonl"
        first_rows = rollout_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(first_rows), len(V18_SCENARIOS))

        class MustNotRun:
            def __init__(self, *_args, **_kwargs) -> None:
                raise AssertionError("resume attempted to rerun a completed rollout")

        module.OpenAICompatibleAgent = MustNotRun
        with patch.object(sys, "argv", [*arguments, "--resume"]), redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 0)
        self.assertEqual(rollout_path.read_text(encoding="utf-8").splitlines(), first_rows)


if __name__ == "__main__":
    unittest.main()
