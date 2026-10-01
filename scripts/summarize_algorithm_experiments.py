from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.data import load_jsonl  # noqa: E402
from ecommerce_customer_service.diagnostics import analyze_bad_cases  # noqa: E402
from ecommerce_customer_service.evaluation import compare_rollouts, evaluate_rollouts  # noqa: E402


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must use NAME=ROLLOUTS.jsonl")
    return name.strip(), Path(raw_path).resolve()


def summarize(runs: list[tuple[str, Path]], baseline: str) -> dict[str, Any]:
    if len({name for name, _ in runs}) != len(runs):
        raise ValueError("duplicate_run_name")
    rows_by_name = {name: load_jsonl(path) for name, path in runs}
    if baseline not in rows_by_name:
        raise ValueError("baseline_run_missing")
    report: dict[str, Any] = {
        "baseline": baseline,
        "runs": {},
        "paired_comparisons": {},
    }
    for name, path in runs:
        rows = rows_by_name[name]
        report["runs"][name] = {
            "source": str(path),
            "evaluation": evaluate_rollouts(rows),
            "bad_cases": analyze_bad_cases(rows),
        }
        if name != baseline:
            report["paired_comparisons"][f"{baseline}_vs_{name}"] = compare_rollouts(
                rows_by_name[baseline],
                rows,
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize SFT/RL rollout files under one paired evaluation protocol"
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME=ROLLOUTS.jsonl")
    parser.add_argument("--baseline", default="sft_control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
