from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
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


def _copy(row: dict[str, Any], *, role: str, id_suffix: str = "") -> dict[str, Any]:
    copied = dict(row)
    copied["messages"] = [dict(message) for message in row.get("messages") or []]
    if not copied["messages"] or copied["messages"][0].get("role") != "system":
        raise ValueError(f"missing_system_message:{row.get('id')}")
    copied["messages"][0]["content"] = V18_SYSTEM_PROMPT
    copied["metadata"] = {**dict(row.get("metadata") or {}), "rebalance_role": role}
    if id_suffix:
        copied["id"] = str(row["id"]) + id_suffix
    return copied


def _prompts(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(message.get("content") or "")
        for row in rows
        for message in row.get("messages") or []
        if message.get("role") == "user"
    }


def build(output_dir: Path, *, seed: int = 1820) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"rebalance output must be empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    supplement_path = ROOT / "data/v18-sft-supplement-r1/train.jsonl"
    original_train_path = ROOT / "data/v18-sft-reference-r1/train.jsonl"
    original_dev_path = ROOT / "data/v18-sft-reference-r1/dev.jsonl"
    supplement = load_jsonl(supplement_path)
    original_train = load_jsonl(original_train_path)
    original_dev = load_jsonl(original_dev_path)

    contradiction = [
        _copy(row, role="contradictory_focus")
        for row in supplement
        if row.get("scenario") == "contradictory_order"
        and (row.get("metadata") or {}).get("supplement_role") == "focus"
    ]
    cross_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in supplement:
        if (
            row.get("scenario") == "cross_order_item_mismatch"
            and (row.get("metadata") or {}).get("supplement_role") == "focus"
        ):
            cross_by_task[str(row["task_id"])].append(row)
    selected_cross_tasks = sorted(cross_by_task)[:40]
    cross = [
        _copy(row, role="cross_order_retention")
        for task_id in selected_cross_tasks
        for row in sorted(cross_by_task[task_id], key=lambda value: str(value["id"]))
    ]
    quantity = [
        _copy(row, role="quantity_counterbalance", id_suffix="_rebalance_r2")
        for row in original_train
        if row.get("scenario") == "refund_quantity_exceeds"
    ]
    replay = [
        _copy(row, role="general_replay")
        for row in supplement
        if (row.get("metadata") or {}).get("supplement_role") == "replay"
    ]
    rows = contradiction + cross + quantity + replay
    random.Random(seed).shuffle(rows)

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
    dev_prompt_overlap = _prompts(rows) & _prompts(original_dev)
    dev_task_ids = {str(row.get("task_id")) for row in original_dev}
    train_task_ids = {str(row.get("task_id")) for row in rows}
    if dev_prompt_overlap:
        failures.append(f"dev_prompt_overlap:{len(dev_prompt_overlap)}")
    if train_task_ids & dev_task_ids:
        failures.append(f"dev_task_overlap:{len(train_task_ids & dev_task_ids)}")
    if failures:
        raise ValueError("invalid_rebalance_data:" + ";".join(failures[:20]))

    train_path = output_dir / "train.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "ecommerce_cs_v18_rebalance_summary_v2",
        "seed": seed,
        "train_rows": len(rows),
        "role_rows": dict(sorted(Counter((row.get("metadata") or {})["rebalance_role"] for row in rows).items())),
        "scenario_rows": dict(sorted(Counter(str(row["scenario"]) for row in rows).items())),
        "dev_rows_in_train": 0,
        "test_rows_in_train": 0,
        "dev_prompt_overlap": 0,
        "dev_task_overlap": 0,
        "sources": {
            "supplement_train_sha256": sha256_file(supplement_path),
            "original_train_sha256": sha256_file(original_train_path),
            "frozen_dev_sha256": sha256_file(original_dev_path),
        },
        "train_sha256": sha256_file(train_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a short balanced v18 correction set")
    parser.add_argument("--output", type=Path, default=ROOT / "data/v18-sft-rebalance-r2")
    parser.add_argument("--seed", type=int, default=1820)
    args = parser.parse_args()
    print(json.dumps(build(args.output, seed=args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
