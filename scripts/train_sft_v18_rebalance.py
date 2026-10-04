from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.data import load_jsonl  # noqa: E402
from ecommerce_customer_service.sft_v18_48gb import inspect_48gb_hardware, sha256_file  # noqa: E402
from ecommerce_customer_service.training import validate_training_row  # noqa: E402


TRAIN = ROOT / "data/v18-sft-rebalance-r2/train.jsonl"
SUMMARY = ROOT / "data/v18-sft-rebalance-r2/summary.json"
DEV = ROOT / "data/v18-sft-reference-r1/dev.jsonl"
ADAPTER = ROOT / "outputs/v18-sft-full-newgpu-r1"
OUTPUT = ROOT / "outputs/v18-sft-rebalance-r2"
PREFLIGHT_OUTPUT = ROOT / "outputs/v18-sft-rebalance-preflight-r2"
EXPECTED = {
    "train": "6f7695dc33c000348e83c94366246a4f87d0810924defc89f93b20a8b2c970e3",
    "summary": "1b82eec49cdb7f8caf8a5da4aadb2e3de06d71f336fb7ba993b5be7452667f3a",
    "dev": "a1b246662715aaa8f72a8cdb234d218a19ff2d94fbe2164572d5eb8d4efae5d0",
    "adapter_model": "f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c",
    "adapter_config": "d3c421aa3bc1b4ae4cefe760a9db5dda54fa42a2a54fa491372233d2c3c7b094",
}
EXPECTED_ROLES = {
    "contradictory_focus": 480,
    "cross_order_retention": 200,
    "general_replay": 600,
    "quantity_counterbalance": 400,
}


def preflight(*, hardware: bool, device: int) -> dict[str, Any]:
    files = {
        "train": TRAIN,
        "summary": SUMMARY,
        "dev": DEV,
        "adapter_model": ADAPTER / "adapter_model.safetensors",
        "adapter_config": ADAPTER / "adapter_config.json",
    }
    for label, path in files.items():
        if not path.is_file():
            raise ValueError(f"missing_rebalance_input:{label}:{path}")
        actual = sha256_file(path)
        if actual != EXPECTED[label]:
            raise ValueError(f"rebalance_hash_mismatch:{label}:{actual}:{EXPECTED[label]}")
    rows = load_jsonl(TRAIN)
    if len(rows) != 1680:
        raise ValueError(f"rebalance_row_count:{len(rows)}")
    roles: Counter[str] = Counter()
    for row in rows:
        failures = validate_training_row(row)
        metadata = dict(row.get("metadata") or {})
        if failures or metadata.get("split") != "train" or metadata.get("test_material") is not False:
            raise ValueError(f"invalid_rebalance_row:{row.get('id')}:{','.join(failures)}")
        roles[str(metadata.get("rebalance_role"))] += 1
    if dict(roles) != EXPECTED_ROLES:
        raise ValueError(f"rebalance_role_counts:{dict(roles)}")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if any(int(summary.get(key, -1)) != 0 for key in ("dev_rows_in_train", "test_rows_in_train", "dev_prompt_overlap", "dev_task_overlap")):
        raise ValueError("rebalance_data_leakage")
    return {
        "schema_version": "ecommerce_cs_v18_rebalance_preflight_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "roles": dict(roles),
        "max_steps": 28,
        "dev_or_test_rows_in_train": 0,
        "adapter_model_sha256": EXPECTED["adapter_model"],
        "hardware": inspect_48gb_hardware(device, minimum_gib=22.0) if hardware else {"checked": False},
        "ready": True,
    }


def command(output: Path, device: int, resume: Path | None) -> list[str]:
    result = [
        sys.executable, str(ROOT / "scripts/train_sft.py"),
        "--train", str(TRAIN),
        "--dev", str(DEV),
        "--base-model", "Qwen/Qwen2.5-7B-Instruct",
        "--model-revision", "a09a35458c702b33eeacc393d103063234e8bc28",
        "--output", str(output),
        "--max-steps", "28",
        "--batch-size", "1",
        "--gradient-accumulation", "16",
        "--learning-rate", "1e-5",
        "--warmup-ratio", "0.05",
        "--max-sequence-length", "4096",
        "--eval-every", "14",
        "--save-every", "4",
        "--max-dev-examples", "30",
        "--seed", "1820",
        "--device", str(device),
        "--model-dtype", "bfloat16",
        "--trainable-params-fp32",
        "--defer-gradient-accumulation-scaling",
        "--gradient-checkpointing",
        "--no-gradient-checkpointing-reentrant",
        "--attn-implementation", "sdpa",
        "--load-in-4bit",
    ]
    result.extend(["--resume-from", str(resume)] if resume else ["--init-adapter", str(ADAPTER)])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 28-step balanced v18 correction")
    parser.add_argument("phase", choices=("preflight", "train"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.phase == "train" and args.data_only:
        raise SystemExit("--data-only is only valid for preflight")
    receipt = preflight(hardware=args.phase == "train" and not args.data_only, device=args.device)
    output = (args.output or (PREFLIGHT_OUTPUT if args.phase == "preflight" else OUTPUT)).resolve()
    if args.phase == "preflight":
        output.mkdir(parents=True, exist_ok=True)
        (output / "preflight-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    resume = args.resume_from.resolve() if args.resume_from else None
    if resume:
        if output not in resume.parents or not (resume / "training_state.pt").is_file():
            raise SystemExit("resume checkpoint must be valid and inside output")
    elif output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be absent or empty:{output}")
    output.mkdir(parents=True, exist_ok=True)
    receipt["command"] = command(output, args.device, resume)
    (output / "preflight-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    process = subprocess.Popen(receipt["command"], cwd=ROOT)
    (output / "trainer.pid").write_text(str(process.pid) + "\n", encoding="utf-8")

    def forward(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            process.send_signal(signum)
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        returncode = process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if returncode:
        raise SystemExit(returncode)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if int(metrics.get("optimizer_steps") or 0) != 28 or not all(
        isinstance(metrics.get(key), (int, float)) and math.isfinite(float(metrics[key]))
        for key in ("initial_dev_loss", "final_dev_loss")
    ):
        raise SystemExit("rebalance completion gate failed")
    print(json.dumps({"complete": True, "metrics": str(output / "metrics.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
