"""Read-only pairing of dev runs; incomplete runs never receive a full pass rate."""
from pathlib import Path
import argparse
import json


def compare(baseline: Path, runs: dict[str, Path]) -> dict:
    base = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if not base["complete"]:
        raise ValueError("baseline_must_be_complete")
    if base.get("mode") != "model" or base.get("real_model_execution") is not True:
        raise ValueError("comparison_requires_real_model_mode")
    outcomes = lambda folder: {path.parent.name: json.loads(path.read_text(encoding="utf-8"))["passed"]
                              for path in folder.glob("cases/*/result.json")}
    before = outcomes(baseline)
    controls = ("mode", "real_model_execution", "evaluation_kind", "model", "case_sha256", "case_ids", "context_counter", "context_budget", "context_policy", "recent_turns")
    rows = []
    for label, folder in runs.items():
        result = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if any(base[key] != result[key] for key in controls):
            raise ValueError("nonpaired_run:" + label)
        after = outcomes(folder)
        if result["complete"] and set(before) != set(after):
            raise ValueError("case_results_missing:" + label)
        differences = [name for name, digest in base["source_sha256"].items()
                       if result["source_sha256"].get(name) != digest]
        rows.append({"profile": label, "complete": result["complete"], "passed": result["passed_cases"],
            "evaluated": result["evaluated_cases"], "expected": result["expected_cases"],
            "pass_rate": result["case_pass_rate"] if result["complete"] else None,
            "improved_cases": [name for name in before if not before[name] and after[name]] if result["complete"] else None,
            "regressed_cases": [name for name in before if before[name] and not after[name]] if result["complete"] else None,
            "model_calls": result["model_calls"], "tool_calls": result["tool_calls"],
            "prompt_sha256": result["system_prompt_sha256"], "tools_sha256": result["tools_sha256"],
            "source_differences": differences, "failures": result["failures"]})
    return {"evaluation_kind": "exploratory_paired_dev_comparison", "independent_test_claim": False,
            "model_training_performed": False, "rows": rows,
            "limitations": ["single_greedy_trial", "synthetic_dev_cases", "review_source_differences",
                            "incomplete_runs_not_ranked", "limited_phrase_verifier_requires_manual_review"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="profile=run_directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        name, path = value.split("=", 1)
        if name in runs:
            parser.error("duplicate run label")
        runs[name] = Path(path)
    report = compare(args.baseline, runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        target.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
