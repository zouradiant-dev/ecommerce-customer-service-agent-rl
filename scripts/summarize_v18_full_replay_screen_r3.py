from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the checkpoint regression screen as a compact table")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "outputs/v18-sft-full-replay-screen-r3/summary.json",
    )
    args = parser.parse_args()
    value = json.loads(args.summary.read_text(encoding="utf-8"))
    result = {}
    for stage, summary in value.get("stage_summaries", {}).items():
        result[stage] = {
            scenario: {
                "pass": int(item["strict_passes"]),
                "done": int(item["rollouts"]),
                "hard": int(item["hard_safety_failure_rollouts"]),
            }
            for scenario, item in summary.get("by_scenario", {}).items()
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
