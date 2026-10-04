"""Audit-only v18 rescoring of saved trajectories without model inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_customer_service.evaluation_v18 import summarize_v18_rollouts
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.tasks import load_task_paths
from ecommerce_customer_service.verifier_v18 import (
    V18_VERIFIER_VERSION,
    score_v18,
    verify_task_v18,
)


DERIVED_FIELDS = {"reward", "reward_components", "verification", "verifier_version", "rescore_metadata"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _rescore_row(source: dict[str, Any], task: TaskSpec) -> dict[str, Any]:
    """Recompute trajectory checks while retaining the ephemeral post-state result."""

    calls = list(source.get("tool_executions") or [])
    candidate = verify_task_v18(task, str(source.get("final_response") or ""), calls)
    old = dict(source["verification"])
    state_failures = [
        str(value) for value in old.get("failures") or () if str(value).startswith("state_")
    ]
    recomputed_failures = [
        str(value) for value in candidate.failures if not str(value).startswith("state_")
    ]
    if source.get("error") and "worker_error" not in recomputed_failures:
        recomputed_failures.append("worker_error")
    failures = tuple(dict.fromkeys((*state_failures, *recomputed_failures)))
    state_passed = bool(old["state_passed"])
    verification = replace(
        candidate,
        state_passed=state_passed,
        failures=failures,
        passed=(
            state_passed
            and candidate.communication_passed
            and candidate.policy_passed
            and candidate.required_tools_passed
            and candidate.tool_sequence_passed
            and candidate.terminal_passed
            and candidate.safety_gate_passed
            and not source.get("error")
        ),
    )
    reward = score_v18(
        task,
        verification,
        calls,
        stopped_reason="error" if source.get("error") else str(source.get("stopped_reason") or "completed"),
    )
    row = deepcopy(source)
    row["reward"] = reward.strict_binary_v18
    row["reward_components"] = reward.to_dict()
    row["verification"] = verification.to_dict()
    row["verifier_version"] = V18_VERIFIER_VERSION
    row["rescore_metadata"] = {
        "method": "retain_ephemeral_state_recompute_saved_trajectory",
        "model_rerun": False,
        "source_verifier_version": source.get("verifier_version"),
        "source_reward": source.get("reward"),
    }
    return row


def run(source: Path, manifest: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    manifest = manifest.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    source_hash = _sha256(source)
    source_rows = _read_jsonl(source)
    tasks = {
        task.task_id: task
        for path in load_task_paths(manifest)
        for task in (TaskSpec.from_file(path),)
    }
    if set(str(row.get("task_id")) for row in source_rows) - set(tasks):
        raise ValueError("source contains task IDs absent from the supplied manifest")

    rescored = [_rescore_row(row, tasks[str(row["task_id"])]) for row in source_rows]
    for old, new in zip(source_rows, rescored):
        old_payload = {key: value for key, value in old.items() if key not in DERIVED_FIELDS}
        new_payload = {key: value for key, value in new.items() if key not in DERIVED_FIELDS}
        if old_payload != new_payload:
            raise RuntimeError(f"trajectory payload changed:{old['task_id']}")

    changes = []
    for old, new in zip(source_rows, rescored):
        if (
            _canonical_json(old["verification"]) != _canonical_json(new["verification"])
            or _canonical_json(old["reward_components"]) != _canonical_json(new["reward_components"])
        ):
            changes.append(
                {
                    "task_id": old["task_id"],
                    "scenario": old["scenario"],
                    "strict_before": old["reward"],
                    "strict_after": new["reward"],
                    "shaped_before": old["reward_components"]["train_shaped_v18"],
                    "shaped_after": new["reward_components"]["train_shaped_v18"],
                    "failures_before": old["verification"]["failures"],
                    "failures_after": new["verification"]["failures"],
                }
            )

    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "rollouts.jsonl", rescored)
    _write_jsonl(output / "changes.jsonl", changes)
    evaluation = summarize_v18_rollouts(
        rescored,
        expected_tasks=len({str(row["task_id"]) for row in rescored}),
        trials_per_task=max(int(row.get("trial") or 0) for row in rescored) + 1,
    )
    summary = {
        "complete": evaluation["complete"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "retain_ephemeral_state_recompute_saved_trajectory",
        "model_rerun": False,
        "source_file": str(source),
        "source_sha256": source_hash,
        "source_files_unchanged": _sha256(source) == source_hash,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "source_verifier_version": source_rows[0].get("verifier_version") if source_rows else None,
        "verifier_version": V18_VERIFIER_VERSION,
        "changed_rows": len(changes),
        "evaluation": evaluation,
        "interpretation": (
            "The model was not rerun. Saved responses and tool trajectories are unchanged; "
            "only derived verifier and reward fields were recalculated after dev calibration."
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.source, args.manifest, args.output)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "model_rerun": result["model_rerun"],
                "source_files_unchanged": result["source_files_unchanged"],
                "source_verifier_version": result["source_verifier_version"],
                "verifier_version": result["verifier_version"],
                "changed_rows": result["changed_rows"],
                "overall": result["evaluation"]["overall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
