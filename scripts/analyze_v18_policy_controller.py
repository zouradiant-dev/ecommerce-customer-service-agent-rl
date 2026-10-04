#!/usr/bin/env python3
"""Compare a policy-controller evaluation with saved baseline rollouts.

The comparison is keyed by task_id, so a calibration subset can be compared
against a larger baseline run without assuming file order.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def passed(row: dict[str, Any]) -> bool:
    return bool((row.get("verification") or {}).get("passed"))


def failures(row: dict[str, Any], key: str = "failures") -> list[str]:
    return list((row.get("verification") or {}).get(key) or [])


def raw_terminal(row: dict[str, Any]) -> str | None:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return message.get("local_raw_generation")
    return None


def intervention_flags(row: dict[str, Any]) -> list[str]:
    audit = row.get("policy_controller") or {}
    values: list[str] = []
    if audit.get("terminal_normalized"):
        values.append("terminal_normalized")
    if audit.get("controller_read"):
        values.append("controller_read")
    if audit.get("controller_evidence"):
        values.append("controller_evidence")
    if audit.get("controller_action"):
        values.append("controller_action")
    if audit.get("blocked_proposals"):
        values.append("blocked_proposal")
    return values or ["none"]


def compact_tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.get("name"),
            "ok": item.get("ok"),
            "is_write": item.get("is_write"),
            "policy_violation": item.get("policy_violation"),
            "arguments": item.get("arguments"),
        }
        for item in row.get("tool_executions") or []
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_rows = read_jsonl(args.candidate)
    baseline_rows = read_jsonl(args.baseline)
    baseline_by_task = {row["task_id"]: row for row in baseline_rows}
    missing = [row["task_id"] for row in candidate_rows if row["task_id"] not in baseline_by_task]
    if missing:
        raise ValueError(f"baseline is missing {len(missing)} candidate tasks")

    outcome_counts: Counter[str] = Counter()
    same_seed = 0
    intervention_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    scenarios: dict[str, Counter[str]] = defaultdict(Counter)
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    for candidate in candidate_rows:
        baseline = baseline_by_task[candidate["task_id"]]
        candidate_pass = passed(candidate)
        baseline_pass = passed(baseline)
        outcome = (
            "both_pass"
            if candidate_pass and baseline_pass
            else "improved"
            if candidate_pass
            else "regressed"
            if baseline_pass
            else "both_fail"
        )
        outcome_counts[outcome] += 1
        if (candidate.get("inference") or {}).get("seed") == (baseline.get("inference") or {}).get("seed"):
            same_seed += 1
        for flag in intervention_flags(candidate):
            intervention_outcomes[flag][outcome] += 1
        scenarios[str(candidate.get("scenario"))][outcome] += 1

        detail = {
            "task_id": candidate.get("task_id"),
            "scenario": candidate.get("scenario"),
            "same_seed": (candidate.get("inference") or {}).get("seed")
            == (baseline.get("inference") or {}).get("seed"),
            "candidate_failures": failures(candidate),
            "candidate_hard_failures": failures(candidate, "hard_safety_failures"),
            "baseline_failures": failures(baseline),
            "baseline_hard_failures": failures(baseline, "hard_safety_failures"),
            "candidate_final": candidate.get("final_response"),
            "candidate_raw_terminal": raw_terminal(candidate),
            "baseline_final": baseline.get("final_response"),
            "controller": candidate.get("policy_controller") or {},
            "candidate_tools": compact_tools(candidate),
            "baseline_tools": compact_tools(baseline),
        }
        if outcome == "regressed":
            regressions.append(detail)
        elif outcome == "improved":
            improvements.append(detail)

    report = {
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "common_tasks": len(candidate_rows),
        "same_seed_tasks": same_seed,
        "outcomes": dict(outcome_counts),
        "candidate_passes": sum(passed(row) for row in candidate_rows),
        "baseline_passes_on_same_tasks": sum(
            passed(baseline_by_task[row["task_id"]]) for row in candidate_rows
        ),
        "candidate_hard_failures": sum(bool(failures(row, "hard_safety_failures")) for row in candidate_rows),
        "baseline_hard_failures_on_same_tasks": sum(
            bool(failures(baseline_by_task[row["task_id"]], "hard_safety_failures"))
            for row in candidate_rows
        ),
        "intervention_outcomes": {
            key: dict(value) for key, value in sorted(intervention_outcomes.items())
        },
        "scenario_outcomes": {key: dict(value) for key, value in sorted(scenarios.items())},
        "regressions": regressions,
        "improvements": improvements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({key: report[key] for key in (
        "common_tasks",
        "same_seed_tasks",
        "outcomes",
        "candidate_passes",
        "baseline_passes_on_same_tasks",
        "candidate_hard_failures",
        "baseline_hard_failures_on_same_tasks",
        "intervention_outcomes",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
