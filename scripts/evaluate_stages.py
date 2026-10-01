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
from ecommerce_customer_service.simulator import OpenAICompatibleAgent, collect_rollouts  # noqa: E402
from ecommerce_customer_service.tasks import load_task_paths  # noqa: E402


def parse_stage(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("stage must use NAME=MODEL, for example base=Qwen/Qwen2.5-7B-Instruct")
    name, model = value.split("=", 1)
    if not name.strip() or not model.strip():
        raise argparse.ArgumentTypeError("stage name and model must both be non-empty")
    return name.strip(), model.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Base/SFT/GRPO models with one paired protocol")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", action="append", type=parse_stage, required=True, help="repeat NAME=MODEL")
    parser.add_argument("--base-stage", default="base")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--runtime", choices=["native", "langgraph"], default="native")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--trials-per-task", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    stages = dict(args.stage)
    if len(stages) != len(args.stage):
        raise ValueError("stage names must be unique")
    if args.base_stage not in stages:
        raise ValueError(f"base stage {args.base_stage!r} is not present")
    agent_class = OpenAICompatibleAgent
    if args.runtime == "langgraph":
        try:
            from ecommerce_customer_service.langgraph_agent import LangGraphCustomerServiceAgent
        except ImportError as exc:
            raise SystemExit('Install the optional Agent runtime with: pip install -e ".[agent]"') from exc
        agent_class = LangGraphCustomerServiceAgent
    task_paths = load_task_paths(args.manifest, split=args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    rollout_rows: dict[str, list[dict[str, Any]]] = {}
    for name, model in stages.items():
        stage_dir = args.output / name
        rollout_path = stage_dir / "rollouts.jsonl"
        collection = collect_rollouts(
            task_paths,
            lambda task, trial, model=model: agent_class(
                model,
                api_base=args.api_base,
                api_key=args.api_key,
                temperature=args.temperature,
                max_steps=args.max_steps,
            ),
            rollout_path,
            trials_per_task=args.trials_per_task,
        )
        rows = load_jsonl(rollout_path)
        rollout_rows[name] = rows
        report = {
            "model": model,
            "collection": collection,
            "evaluation": evaluate_rollouts(rows),
            "badcases": analyze_bad_cases(rows),
        }
        reports[name] = report
        write_json(stage_dir / "report.json", report)

    base_rows = rollout_rows[args.base_stage]
    comparisons = {
        name: compare_rollouts(base_rows, rows)
        for name, rows in rollout_rows.items()
        if name != args.base_stage
    }
    summary = {
        "protocol": {
            "manifest": str(args.manifest.resolve()),
            "split": args.split,
            "trials_per_task": args.trials_per_task,
            "temperature": args.temperature,
            "max_steps": args.max_steps,
            "runtime": args.runtime,
            "base_stage": args.base_stage,
        },
        "stages": reports,
        "paired_comparisons_to_base": comparisons,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
