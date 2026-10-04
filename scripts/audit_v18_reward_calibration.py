from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.v18_readiness import assess_v18_reward_calibration  # noqa: E402


BOUND_SOURCES = (
    ROOT / "src/ecommerce_customer_service/tools_v18.py",
    ROOT / "src/ecommerce_customer_service/terminal_v18.py",
    ROOT / "src/ecommerce_customer_service/verifier_v18.py",
    ROOT / "configs/v18-reward.json",
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a signed-by-hash CPU calibration receipt for v18")
    parser.add_argument("--rollouts", type=Path, required=True, help="human-reviewed JSONL rollout rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rollouts.read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = {path.relative_to(ROOT).as_posix(): digest(path) for path in BOUND_SOURCES}
    sources[args.rollouts.resolve().as_posix()] = digest(args.rollouts.resolve())
    receipt = assess_v18_reward_calibration(rows, source_sha256=sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["reward_calibrated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
