"""Audit-only rescoring of a completed dev comparison; never generate new replies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_customer_service.data import load_jsonl, write_jsonl
from ecommerce_customer_service.diagnostics import analyze_bad_cases
from ecommerce_customer_service.evaluation import compare_rollouts, evaluate_rollouts
from ecommerce_customer_service.local_hf import select_dev_tasks
from ecommerce_customer_service.rescoring import rescore_rollouts
from ecommerce_customer_service.verifier import VERIFIER_VERSION

SOURCE_FILES = (
    "protocol.json", "model_info.json", "summary.json",
    "base/report.json", "base/rollouts.jsonl",
    "sft/report.json", "sft/rollouts.jsonl",
)
DERIVED_FIELDS = {"schema_version", "verifier", "verifier_version", "reward", "reward_components", "rescore_metadata"}


def non_intent_failures(values: set[str]) -> set[str]:
    return {item for item in values if not item.startswith("missing_intent:")}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def source_hashes(source: Path) -> dict[str, str]:
    return {name: sha256_file(source / name) for name in SOURCE_FILES}


def run(source: Path, manifest: Path, output: Path) -> dict[str, Any]:
    source, manifest, output = source.resolve(), manifest.resolve(), output.resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output must be separate, non-nested directories")
    if output.exists():
        raise FileExistsError(f"Output already exists; no overwrite: {output}")
    hashes_before = source_hashes(source)
    original = read_json(source / "summary.json")
    protocol = read_json(source / "protocol.json")
    if original.get("complete") is not True:
        raise ValueError("Source evaluation is not complete")
    if original.get("protocol") != protocol:
        raise ValueError("Source protocol does not match the completed summary")
    if protocol.get("split") != "dev" or protocol.get("selection") != "full_dev":
        raise ValueError("This audit entry point accepts only completed full-dev evaluations")
    if sha256_file(manifest) != protocol.get("manifest_sha256"):
        raise ValueError("Manifest hash differs from the generation protocol")
    selected = select_dev_tasks(manifest, per_template=None)
    expected_ids = {task.task_id for _, task in selected}
    if expected_ids != set(protocol["task_ids"]) or len(expected_ids) != protocol["tasks"]:
        raise ValueError("Dev task IDs/count differ from the generation protocol")
    source_rows = {stage: load_jsonl(source / stage / "rollouts.jsonl") for stage in ("base", "sft")}
    expected_keys = {(task_id, trial) for task_id in expected_ids for trial in range(protocol["trials_per_task"])}
    for stage, rows in source_rows.items():
        keys = [(row["task_id"], row["trial"]) for row in rows]
        if len(keys) != len(expected_keys) or set(keys) != expected_keys:
            raise ValueError(f"{stage}: missing/duplicate/unexpected task-trial records")
        if any(row.get("split") != "dev" or row.get("error") is not None
               or row.get("stopped_reason") == "worker_error"
               or row.get("verifier_version") != protocol["verifier_version"] for row in rows):
            raise ValueError(f"{stage}: invalid split, worker error or source verifier version")
        if evaluate_rollouts(rows) != original["stages"][stage]["evaluation"]:
            raise ValueError(f"{stage}: saved report differs from the raw trajectories")

    # All input checks precede creation. No operation targets the original directory.
    output.mkdir(parents=True, exist_ok=False)
    audit_protocol = {
        "method": "post_hoc_communication_calibration",
        "model_rerun": False,
        "source_directory": str(source),
        "source_verifier_version": protocol["verifier_version"],
        "target_verifier_version": VERIFIER_VERSION,
        "source_file_sha256": hashes_before,
        "source_generation_protocol": protocol,
        "state_checks": "retained from original isolated execution; no post-state database available",
        "trajectory_payload": "unchanged: prompts, responses, tool calls/results, seeds and generation diagnostics",
        "required_tools_and_policy": "recomputed and asserted identical to source",
        "scoring_source_sha256": {
            name: sha256_file(ROOT / name) for name in (
                "src/ecommerce_customer_service/communication.py",
                "src/ecommerce_customer_service/verifier.py",
                "src/ecommerce_customer_service/rescoring.py",
                "scripts/rescore_local_hf.py",
            )
        },
    }
    write_json(output / "protocol.json", audit_protocol)
    changes: list[dict[str, Any]] = []
    derived: dict[str, list[dict[str, Any]]] = {}
    stages: dict[str, Any] = {}
    for stage in ("base", "sft"):
        stage_dir = output / stage
        stage_dir.mkdir()
        metadata = rescore_rollouts(source / stage / "rollouts.jsonl", manifest, stage_dir / "rollouts.jsonl")
        rows = load_jsonl(stage_dir / "rollouts.jsonl")
        for old, new in zip(source_rows[stage], rows):
            if {key: value for key, value in old.items() if key not in DERIVED_FIELDS} != {
                key: value for key, value in new.items() if key not in DERIVED_FIELDS
            }:
                raise RuntimeError(f"Trajectory payload changed: {stage}/{old['task_id']}")
            for field in ("state_passed", "policy_passed", "required_tools_passed"):
                if old["verifier"][field] != new["verifier"][field]:
                    raise RuntimeError(f"Non-communication component changed: {stage}/{old['task_id']}/{field}")
            before = set(old["verifier"]["failures"])
            after = set(new["verifier"]["failures"])
            if non_intent_failures(before) != non_intent_failures(after):
                raise RuntimeError(f"Non-intent failure changed: {stage}/{old['task_id']}")
            if before != after or old["reward"] != new["reward"]:
                changes.append({
                    "stage": stage, "task_id": old["task_id"], "scenario": old["scenario"],
                    "reward_before": old["reward"], "reward_after": new["reward"],
                    "failures_before": old["verifier"]["failures"],
                    "failures_after": new["verifier"]["failures"],
                    "removed_failures": sorted(before - after), "added_failures": sorted(after - before),
                    "final_response": old["final_response"], "trajectory_payload_unchanged": True,
                })
        report = {
            "evaluation": evaluate_rollouts(rows),
            "badcases": analyze_bad_cases(rows),
            "verifier_component_rates": {
                field: sum(bool(row["verifier"][field]) for row in rows) / len(rows)
                for field in ("state_passed", "communication_passed", "policy_passed", "required_tools_passed")
            },
            "rescore": metadata,
            "source_generation_diagnostics": {
                key: original["stages"][stage].get(key) for key in
                ("invalid_tool_call_rollouts", "max_steps_rollouts", "truncated_generations", "source_databases_unchanged", "elapsed_seconds")
            },
        }
        write_json(stage_dir / "report.json", report)
        stages[stage], derived[stage] = report, rows
    write_jsonl(output / "changes.jsonl", changes)
    hashes_after = source_hashes(source)
    if hashes_after != hashes_before:
        raise RuntimeError("Original evaluation files changed during the audit")
    summary = {
        "complete": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": audit_protocol["method"], "model_rerun": False,
        "source_files_unchanged": True, "source_file_sha256": hashes_after,
        "source_verifier_version": protocol["verifier_version"], "verifier_version": VERIFIER_VERSION,
        "stages": stages, "paired_comparison": compare_rollouts(derived["base"], derived["sft"]),
        "changed_rows": len(changes),
        "interpretation": (
            "Same saved dev trajectories, recalibrated after inspecting dev failures. "
            "Not new model learning, not an independent performance gain, held-out test or RL gate."
        ),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.source, args.manifest, args.output)
    print(json.dumps({
        "complete": summary["complete"], "source_files_unchanged": summary["source_files_unchanged"],
        "verifier_version": summary["verifier_version"], "changed_rows": summary["changed_rows"],
        "stages": {stage: report["rescore"] for stage, report in summary["stages"].items()},
        "paired_comparison": summary["paired_comparison"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
