from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.sft_v18_supplement import build_v18_supplement  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train-only v18 corrective SFT supplement")
    parser.add_argument("--output", type=Path, default=ROOT / "data/v18-sft-supplement-r1")
    parser.add_argument("--original-train", type=Path, default=ROOT / "data/v18-sft-reference-r1/train.jsonl")
    parser.add_argument("--original-dev", type=Path, default=ROOT / "data/v18-sft-reference-r1/dev.jsonl")
    parser.add_argument("--focus-tasks-per-scenario", type=int, default=160)
    parser.add_argument("--replay-rows-per-scenario", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1818)
    args = parser.parse_args()
    summary = build_v18_supplement(
        args.output,
        original_train=args.original_train,
        original_dev=args.original_dev,
        focus_tasks_per_scenario=args.focus_tasks_per_scenario,
        replay_rows_per_scenario=args.replay_rows_per_scenario,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
