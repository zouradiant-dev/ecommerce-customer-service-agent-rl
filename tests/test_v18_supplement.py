from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.data import load_jsonl
from ecommerce_customer_service.experiment_prompt_v18 import V18_SYSTEM_PROMPT
from ecommerce_customer_service.sft_v18_supplement import (
    FOCUS_SCENARIOS,
    validate_supplement_training_file,
)
from ecommerce_customer_service.training import validate_training_row


def test_frozen_supplement_has_no_dev_or_test_leakage() -> None:
    data_dir = ROOT / "data/v18-sft-supplement-r1"
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    receipt = validate_supplement_training_file(
        data_dir / "train.jsonl",
        expected_rows=1880,
    )
    assert receipt["rows"] == 1880
    assert receipt["roles"] == {"focus": 1280, "replay": 600}
    assert all(receipt["scenarios"][scenario] > 20 for scenario in FOCUS_SCENARIOS)
    assert summary["dev_rows_in_train"] == 0
    assert summary["test_rows_in_train"] == 0
    assert summary["dev_prompt_overlap"] == 0
    assert summary["dev_task_overlap"] == 0


def test_replay_rows_use_the_current_complete_protocol_prompt() -> None:
    rows = load_jsonl(ROOT / "data/v18-sft-supplement-r1/train.jsonl")
    replay = [row for row in rows if (row.get("metadata") or {}).get("supplement_role") == "replay"]
    assert len(replay) == 600
    assert all(row["messages"][0] == {"role": "system", "content": V18_SYSTEM_PROMPT} for row in replay)


def test_trainer_rejects_init_adapter_plus_resume_checkpoint() -> None:
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_sft.py"),
            "--train", "unused.jsonl",
            "--output", "unused-output",
            "--init-adapter", "adapter",
            "--resume-from", "checkpoint",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode != 0
    assert "mutually exclusive" in process.stderr


def test_rebalance_data_is_balanced_and_leak_free() -> None:
    data_dir = ROOT / "data/v18-sft-rebalance-r2"
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    rows = load_jsonl(data_dir / "train.jsonl")
    assert len(rows) == 1680
    assert summary["role_rows"] == {
        "contradictory_focus": 480,
        "cross_order_retention": 200,
        "general_replay": 600,
        "quantity_counterbalance": 400,
    }
    assert summary["scenario_rows"]["contradictory_order"] == 500
    assert summary["scenario_rows"]["cross_order_item_mismatch"] == 220
    assert summary["scenario_rows"]["refund_quantity_exceeds"] == 420
    assert all(summary[key] == 0 for key in ("dev_rows_in_train", "test_rows_in_train", "dev_prompt_overlap", "dev_task_overlap"))
    assert all(row["messages"][0] == {"role": "system", "content": V18_SYSTEM_PROMPT} for row in rows)
    assert all(not validate_training_row(row) for row in rows)


def test_full_replay_r3_preserves_the_original_distribution_and_is_leak_free() -> None:
    data_dir = ROOT / "data/v18-sft-full-replay-r3"
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    rows = load_jsonl(data_dir / "train.jsonl")
    assert len(rows) == 16080
    assert summary["role_rows"] == {"base_replay": 14800, "corrective_focus": 1280}
    assert summary["corrective_fraction"] < 0.08
    assert all(
        summary[key] == 0
        for key in ("dev_rows_in_train", "test_rows_in_train", "dev_prompt_overlap", "dev_task_overlap")
    )
    assert all(row["messages"][0] == {"role": "system", "content": V18_SYSTEM_PROMPT} for row in rows)
    assert all(not validate_training_row(row) for row in rows)
