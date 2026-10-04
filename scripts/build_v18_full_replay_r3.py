from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.data import load_jsonl  # noqa: E402
from ecommerce_customer_service.experiment_prompt_v18 import V18_SYSTEM_PROMPT  # noqa: E402
from ecommerce_customer_service.sft_v18_48gb import sha256_file  # noqa: E402
from ecommerce_customer_service.training import validate_training_row  # noqa: E402


def _copy(row: dict[str, Any], role: str) -> dict[str, Any]:
    copied = dict(row)
    copied["messages"] = [dict(message) for message in row.get("messages") or []]
    if not copied["messages"] or copied["messages"][0].get("role") != "system":
        raise ValueError(f"missing_system_prompt:{row.get('id')}")
    copied["messages"][0]["content"] = V18_SYSTEM_PROMPT
    copied["metadata"] = {**dict(row.get("metadata") or {}), "full_replay_role": role}
    return copied


def _prompts(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(message.get("content") or "")
        for row in rows
        for message in row.get("messages") or []
        if message.get("role") == "user"
    }


def build(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"full replay output must be empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = ROOT / "data/v18-sft-reference-r1/train.jsonl"
    supplement_path = ROOT / "data/v18-sft-supplement-r1/train.jsonl"
    dev_path = ROOT / "data/v18-sft-reference-r1/dev.jsonl"
    base_rows = [_copy(row, "base_replay") for row in load_jsonl(base_path)]
    focus_rows = [
        _copy(row, "corrective_focus")
        for row in load_jsonl(supplement_path)
        if (row.get("metadata") or {}).get("supplement_role") == "focus"
    ]
    dev_rows = load_jsonl(dev_path)
    rows = base_rows + focus_rows

    ids: set[str] = set()
    failures: list[str] = []
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in ids:
            failures.append(f"duplicate_or_missing_id:{row_id}")
        ids.add(row_id)
        metadata = dict(row.get("metadata") or {})
        if metadata.get("split") != "train" or metadata.get("test_material") is not False:
            failures.append(f"split_leak:{row_id}")
        failures.extend(f"{row_id}:{item}" for item in validate_training_row(row))
    if _prompts(rows) & _prompts(dev_rows):
        failures.append("dev_prompt_overlap")
    if {str(row.get("task_id")) for row in rows} & {str(row.get("task_id")) for row in dev_rows}:
        failures.append("dev_task_overlap")
    if failures:
        raise ValueError("invalid_full_replay_data:" + ";".join(failures[:20]))

    train_path = output_dir / "train.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "ecommerce_cs_v18_full_replay_summary_r3",
        "train_rows": len(rows),
        "base_replay_rows": len(base_rows),
        "corrective_focus_rows": len(focus_rows),
        "corrective_fraction": len(focus_rows) / len(rows),
        "role_rows": dict(sorted(Counter((row.get("metadata") or {})["full_replay_role"] for row in rows).items())),
        "scenario_rows": dict(sorted(Counter(str(row["scenario"]) for row in rows).items())),
        "dev_rows_in_train": 0,
        "test_rows_in_train": 0,
        "dev_prompt_overlap": 0,
        "dev_task_overlap": 0,
        "sources": {
            "base_train_sha256": sha256_file(base_path),
            "supplement_train_sha256": sha256_file(supplement_path),
            "frozen_dev_sha256": sha256_file(dev_path),
        },
        "train_sha256": sha256_file(train_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v18 full replay plus 8 percent corrective focus")
    parser.add_argument("--output", type=Path, default=ROOT / "data/v18-sft-full-replay-r3")
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
