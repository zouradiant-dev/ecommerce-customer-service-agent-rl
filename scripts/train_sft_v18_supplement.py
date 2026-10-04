from __future__ import annotations

import argparse
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
from ecommerce_customer_service.sft_v18_supplement import validate_supplement_training_file  # noqa: E402
from ecommerce_customer_service.training import validate_training_row  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/sft-v18-supplement-r1.json"
DEFAULT_OUTPUTS = {
    "preflight": ROOT / "outputs/v18-sft-supplement-preflight-r1",
    "smoke": ROOT / "outputs/v18-sft-supplement-smoke-r1",
    "full": ROOT / "outputs/v18-sft-supplement-full-r1",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded v18 corrective SFT continuation")
    parser.add_argument("phase", choices=("preflight", "smoke", "full"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-output", type=Path, default=DEFAULT_OUTPUTS["smoke"])
    parser.add_argument("--resume-from", type=Path, help="checkpoint from the same supplement phase")
    parser.add_argument("--data-only", action="store_true")
    return parser


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _check_file(spec: dict[str, Any], label: str) -> Path:
    path = _resolve(str(spec["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"missing_frozen_input:{label}:{path}")
    actual = sha256_file(path)
    if actual != str(spec["sha256"]).lower():
        raise ValueError(f"frozen_input_hash_mismatch:{label}:{actual}:{spec['sha256']}")
    return path


def _validate_dev(path: Path, expected_rows: int) -> dict[str, Any]:
    rows = load_jsonl(path)
    if len(rows) != expected_rows:
        raise ValueError(f"dev_row_count_mismatch:{len(rows)}:{expected_rows}")
    for row in rows:
        failures = validate_training_row(row)
        metadata = dict(row.get("metadata") or {})
        if failures or metadata.get("split") != "dev" or metadata.get("test_material") is not False:
            raise ValueError(f"invalid_frozen_dev_row:{row.get('id')}:{','.join(failures)}")
    return {"rows": len(rows), "test_rows": 0}


def validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("profile_version") != "ecommerce_cs_sft_v18_supplement_r1":
        raise ValueError("unexpected_supplement_profile")
    revision = str(config["model"].get("revision") or "")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("model_revision_must_be_40_char_commit")
    data = config["data"]
    summary_path = _check_file(data["summary"], "summary")
    train_path = _check_file(data["train"], "train")
    dev_path = _check_file(data["dev"], "dev")
    train_receipt = validate_supplement_training_file(train_path, expected_rows=int(data["train"]["rows"]))
    dev_receipt = _validate_dev(dev_path, int(data["dev"]["rows"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("supplement_version") != "ecommerce_cs_v18_supplement_r1"
        or int(summary.get("dev_rows_in_train", -1)) != 0
        or int(summary.get("test_rows_in_train", -1)) != 0
        or int(summary.get("dev_prompt_overlap", -1)) != 0
        or int(summary.get("dev_task_overlap", -1)) != 0
    ):
        raise ValueError("supplement_provenance_invalid")

    adapter_dir = _resolve(str(config["initial_adapter"]["path"])).resolve()
    adapter_files = {
        "adapter_model.safetensors": config["initial_adapter"]["adapter_model_sha256"],
        "adapter_config.json": config["initial_adapter"]["adapter_config_sha256"],
    }
    for name, expected in adapter_files.items():
        path = adapter_dir / name
        if not path.is_file():
            raise ValueError(f"missing_initial_adapter_file:{path}")
        actual = sha256_file(path)
        if actual != str(expected).lower():
            raise ValueError(f"initial_adapter_hash_mismatch:{name}:{actual}:{expected}")

    effective_batch = int(config["training"]["effective_batch_size"])
    expected_steps = math.ceil(int(data["train"]["rows"]) / effective_batch)
    if int(config["phases"]["full"]["max_steps"]) != expected_steps:
        raise ValueError("full_step_count_not_one_nominal_epoch")
    receipt = {
        "schema_version": "ecommerce_cs_v18_supplement_preflight_v1",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "adapter_dir": str(adapter_dir),
        "adapter_model_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "train": train_receipt,
        "dev": dev_receipt,
        "full_max_steps": expected_steps,
        "dev_or_test_rows_in_train": 0,
        "data_ready": True,
    }
    return config, receipt


def _command(
    config: dict[str, Any], phase: str, output: Path, device: int, resume_from: Path | None
) -> list[str]:
    settings = config["training"]
    phase_settings = config["phases"][phase]
    data = config["data"]
    command = [
        sys.executable,
        str(ROOT / "scripts/train_sft.py"),
        "--train", str(_resolve(data["train"]["path"])),
        "--dev", str(_resolve(data["dev"]["path"])),
        "--base-model", str(config["model"]["id"]),
        "--model-revision", str(config["model"]["revision"]),
        "--output", str(output),
        "--max-steps", str(phase_settings["max_steps"]),
        "--batch-size", str(settings["micro_batch_size"]),
        "--gradient-accumulation", str(settings["gradient_accumulation_steps"]),
        "--learning-rate", str(settings["learning_rate"]),
        "--warmup-ratio", str(settings["warmup_ratio"]),
        "--max-sequence-length", str(settings["max_sequence_length"]),
        "--eval-every", str(phase_settings["eval_every"]),
        "--save-every", str(phase_settings["save_every"]),
        "--max-dev-examples", str(phase_settings["max_dev_examples"]),
        "--seed", "1818",
        "--device", str(device),
        "--model-dtype", str(settings["model_dtype"]),
        "--trainable-params-fp32",
        "--defer-gradient-accumulation-scaling",
        "--gradient-checkpointing",
        "--no-gradient-checkpointing-reentrant",
        "--attn-implementation", str(settings["attention_implementation"]),
        "--load-in-4bit",
    ]
    if int(phase_settings.get("diagnose_gradients_until") or 0):
        command.extend(["--diagnose-gradients-until", str(phase_settings["diagnose_gradients_until"])])
    if resume_from:
        command.extend(["--resume-from", str(resume_from)])
    else:
        command.extend(["--init-adapter", str(_resolve(config["initial_adapter"]["path"]))])
    return command


def _write_receipt(output: Path, receipt: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preflight-receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _validate_smoke(path: Path, config: dict[str, Any], config_hash: str) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    receipt_path = path / "preflight-receipt.json"
    if not metrics_path.is_file() or not receipt_path.is_file():
        raise ValueError("full run requires completed smoke output")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = {
        "config_hash": receipt.get("config_sha256") == config_hash,
        "steps": int(metrics.get("optimizer_steps") or 0) == int(config["phases"]["smoke"]["max_steps"]),
        "adapter_mode": metrics.get("adapter_initialization") == "existing_adapter",
        "train_rows": int(metrics.get("train_rows") or 0) == int(config["data"]["train"]["rows"]),
        "dev_source_rows": int(metrics.get("dev_source_rows") or 0) == int(config["data"]["dev"]["rows"]),
        "finite_losses": all(
            isinstance(metrics.get(key), (int, float)) and math.isfinite(float(metrics[key]))
            for key in ("initial_dev_loss", "final_dev_loss")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("supplement_smoke_gate_failed:" + ",".join(failed))
    return {"metrics_sha256": sha256_file(metrics_path), "checks": checks}


def main() -> int:
    args = _parser().parse_args()
    if args.data_only and args.phase != "preflight":
        raise SystemExit("--data-only is accepted only for preflight")
    if args.resume_from and args.phase == "preflight":
        raise SystemExit("preflight cannot resume")
    config, receipt = validate_config(args.config)
    receipt.update(
        {
            "phase": args.phase,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "trainer_sha256": sha256_file(ROOT / "scripts/train_sft.py"),
            "launcher_sha256": sha256_file(Path(__file__).resolve()),
        }
    )
    receipt["hardware"] = (
        {"checked": False}
        if args.data_only
        else inspect_48gb_hardware(args.device, minimum_gib=float(config["gpu"]["minimum_reported_memory_gib"]))
    )
    output = (args.output or DEFAULT_OUTPUTS[args.phase]).resolve()
    if args.phase == "preflight":
        path = _write_receipt(output, receipt)
        print(json.dumps({**receipt, "receipt_path": str(path)}, ensure_ascii=False, indent=2))
        return 0

    resume_from = args.resume_from.resolve() if args.resume_from else None
    if resume_from:
        if not (resume_from / "training_state.pt").is_file():
            raise SystemExit(f"resume checkpoint has no training_state.pt: {resume_from}")
        if output not in resume_from.parents:
            raise SystemExit("resume checkpoint must be inside the selected output directory")
    elif output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output}")
    if args.phase == "full":
        receipt["smoke_gate"] = _validate_smoke(args.smoke_output.resolve(), config, receipt["config_sha256"])
    command = _command(config, args.phase, output, args.device, resume_from)
    receipt["command"] = command
    receipt["resume_from"] = str(resume_from) if resume_from else None
    receipt_path = _write_receipt(output, receipt)
    print(json.dumps({"phase": args.phase, "receipt_path": str(receipt_path), "command": command}, ensure_ascii=False, indent=2))
    process = subprocess.Popen(command, cwd=ROOT)
    (output / "trainer.pid").write_text(str(process.pid) + "\n", encoding="utf-8")

    def forward_signal(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            process.send_signal(signum)
        raise SystemExit(128 + signum)

    previous = {signum: signal.signal(signum, forward_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        returncode = process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if returncode:
        raise SystemExit(returncode)
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit("supplement training completed without metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if int(metrics.get("optimizer_steps") or 0) != int(config["phases"][args.phase]["max_steps"]):
        raise SystemExit("supplement optimizer step count mismatch")
    print(json.dumps({"phase": args.phase, "complete": True, "metrics": str(metrics_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
