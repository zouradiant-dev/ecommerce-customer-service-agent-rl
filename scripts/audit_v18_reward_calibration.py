from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.grpo_v18 import build_calibration_receipt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a signed-by-hash CPU calibration receipt for v18")
    parser.add_argument("--rollouts", type=Path, required=True, help="human-reviewed JSONL rollout rows")
    parser.add_argument("--protocol", type=Path, required=True, help="immutable sampling protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_calibration_receipt(ROOT, args.rollouts.resolve(), args.protocol.resolve())
    if args.output.exists():
        raise FileExistsError(f"new calibration receipt output required:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["reward_calibrated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
