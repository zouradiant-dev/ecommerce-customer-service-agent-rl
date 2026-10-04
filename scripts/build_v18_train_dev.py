from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.tasks_v18 import build_v18_train_dev_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fresh v18 train/dev tasks without creating test-v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-scenario", type=int, default=100)
    parser.add_argument("--dev-per-scenario", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1807)
    args = parser.parse_args()
    summary = build_v18_train_dev_dataset(
        args.output,
        train_per_scenario=args.train_per_scenario,
        dev_per_scenario=args.dev_per_scenario,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
