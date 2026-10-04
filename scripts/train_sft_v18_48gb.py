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

from ecommerce_customer_service.sft_v18_48gb import (  # noqa: E402
    inspect_48gb_hardware,
    sha256_file,
    validate_v18_sft_bundle,
)


DEFAULT_CONFIG = ROOT / "configs" / "sft-v18-48gb.json"
DEFAULT_SMOKE_OUTPUT = ROOT / "outputs" / "v18-sft-smoke-r1"
DEFAULT_FULL_OUTPUT = ROOT / "outputs" / "v18-sft-full-r1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded 48G single-GPU v18 SFT launcher")
    parser.add_argument("phase", choices=("preflight", "smoke", "full"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-output", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="validate frozen inputs without CUDA; accepted only for preflight",
    )
    return parser


def _write_receipt(output: Path, receipt: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preflight-receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _training_command(config: dict[str, Any], phase: str, output: Path, device: int) -> list[str]:
    settings = config["training"]
    phase_settings = config["phases"][phase]
    data = config["data"]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_sft.py"),
        "--train",
        str(ROOT / data["train"]["path"]),
        "--dev",
        str(ROOT / data["dev"]["path"]),
        "--base-model",
        str(config["model"]["id"]),
        "--model-revision",
        str(config["model"]["revision"]),
        "--output",
        str(output),
        "--max-steps",
        str(phase_settings["max_steps"]),
        "--batch-size",
        str(settings["micro_batch_size"]),
        "--gradient-accumulation",
        str(settings["gradient_accumulation_steps"]),
        "--learning-rate",
        str(settings["learning_rate"]),
        "--warmup-ratio",
        str(settings["warmup_ratio"]),
        "--max-sequence-length",
        str(settings["max_sequence_length"]),
        "--eval-every",
        str(phase_settings["eval_every"]),
        "--save-every",
        str(phase_settings["save_every"]),
        "--max-dev-examples",
        str(phase_settings["max_dev_examples"]),
        "--seed",
        "1807",
        "--device",
        str(device),
        "--model-dtype",
        str(settings.get("model_dtype", "bfloat16")),
        "--trainable-params-fp32"
        if settings.get("trainable_parameters_fp32", False)
        else "--no-trainable-params-fp32",
        "--defer-gradient-accumulation-scaling"
        if settings.get("defer_gradient_accumulation_scaling", False)
        else "--no-defer-gradient-accumulation-scaling",
        "--gradient-checkpointing",
        "--lora-r",
        str(settings["lora_rank"]),
        "--lora-alpha",
        str(settings["lora_alpha"]),
        "--lora-dropout",
        str(settings["lora_dropout"]),
        "--attn-implementation",
        str(settings.get("attention_implementation", "sdpa")),
    ]
    command.append(
        "--gradient-checkpointing-reentrant"
        if settings.get("gradient_checkpointing_reentrant", True)
        else "--no-gradient-checkpointing-reentrant"
    )
    command.append("--load-in-4bit" if config["model"].get("load_in_4bit") else "--no-load-in-4bit")
    diagnose_gradients_until = int(phase_settings.get("diagnose_gradients_until") or 0)
    if diagnose_gradients_until:
        command.extend(["--diagnose-gradients-until", str(diagnose_gradients_until)])
    return command


def _validate_smoke_gate(
    smoke_output: Path,
    config: dict[str, Any],
    current_config_sha256: str,
) -> dict[str, Any]:
    smoke_metrics = smoke_output / "metrics.json"
    smoke_receipt = smoke_output / "preflight-receipt.json"
    if not smoke_metrics.is_file() or not smoke_receipt.is_file():
        raise ValueError(f"full run requires smoke metrics and preflight receipt: {smoke_output}")
    metrics = json.loads(smoke_metrics.read_text(encoding="utf-8"))
    receipt = json.loads(smoke_receipt.read_text(encoding="utf-8"))
    expected_steps = int(config["phases"]["smoke"]["max_steps"])
    expected_train = int(config["data"]["train"]["rows"])
    expected_dev = int(config["data"]["dev"]["rows"])
    expected_dev_sample = int(config["phases"]["smoke"]["max_dev_examples"])
    checks = {
        "receipt_phase": receipt.get("phase") == "smoke",
        "hardware_ready": (receipt.get("hardware") or {}).get("hardware_ready") is True,
        "config_hash": receipt.get("config_sha256") == current_config_sha256,
        "data_ready": receipt.get("data_ready") is True and int(receipt.get("test_rows", -1)) == 0,
        "optimizer_steps": int(metrics.get("optimizer_steps") or 0) == expected_steps,
        "model": metrics.get("base_model") == config["model"]["id"],
        "revision": metrics.get("base_model_revision") == config["model"]["revision"],
        "train_rows": int(metrics.get("train_rows") or 0) == expected_train,
        "dev_source_rows": int(metrics.get("dev_source_rows") or 0) == expected_dev,
        "dev_rows": int(metrics.get("dev_rows") or 0) == expected_dev_sample,
        "no_skips": int(metrics.get("train_skipped") or 0) == 0 and int(metrics.get("dev_skipped") or 0) == 0,
        "finite_dev_loss": all(
            isinstance(metrics.get(key), (int, float)) and math.isfinite(float(metrics[key]))
            for key in ("initial_dev_loss", "final_dev_loss")
        ),
        "finite_memory": isinstance(metrics.get("max_memory_gib"), (int, float))
        and math.isfinite(float(metrics["max_memory_gib"]))
        and float(metrics["max_memory_gib"]) > 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("smoke_gate_failed:" + ",".join(failed))
    return {
        "metrics_path": str(smoke_metrics),
        "metrics_sha256": sha256_file(smoke_metrics),
        "receipt_path": str(smoke_receipt),
        "receipt_sha256": sha256_file(smoke_receipt),
        "optimizer_steps": expected_steps,
        "initial_dev_loss": metrics["initial_dev_loss"],
        "final_dev_loss": metrics["final_dev_loss"],
        "max_memory_gib": metrics["max_memory_gib"],
        "checks": checks,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.data_only and args.phase != "preflight":
        raise SystemExit("--data-only is permitted only with the preflight phase")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt = validate_v18_sft_bundle(config_path, project_root=ROOT)
    receipt["phase"] = args.phase
    receipt["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["launcher_sha256"] = sha256_file(Path(__file__).resolve())
    receipt["trainer_sha256"] = sha256_file(ROOT / "scripts" / "train_sft.py")
    receipt["hardware"] = {"checked": False} if args.data_only else inspect_48gb_hardware(
        args.device,
        minimum_gib=float(config["gpu"]["minimum_reported_memory_gib"]),
    )

    if args.phase == "preflight":
        output = (args.output or ROOT / "outputs" / "v18-sft-preflight-r1").resolve()
        path = _write_receipt(output, receipt)
        print(json.dumps({**receipt, "receipt_path": str(path)}, ensure_ascii=False, indent=2))
        return 0

    output = (args.output or (DEFAULT_SMOKE_OUTPUT if args.phase == "smoke" else DEFAULT_FULL_OUTPUT)).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output}")
    if args.phase == "full":
        try:
            receipt["smoke_gate"] = _validate_smoke_gate(
                args.smoke_output.resolve(),
                config,
                receipt["config_sha256"],
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    command = _training_command(config, args.phase, output, args.device)
    receipt["command"] = command
    receipt_path = _write_receipt(output, receipt)
    print(json.dumps({"phase": args.phase, "receipt_path": str(receipt_path), "command": command}, ensure_ascii=False, indent=2))
    process = subprocess.Popen(command, cwd=ROOT)
    (output / "trainer.pid").write_text(str(process.pid) + "\n", encoding="utf-8")

    def forward_signal(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            process.send_signal(signum)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise SystemExit(128 + signum)

    previous_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        returncode = process.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if returncode:
        raise SystemExit(returncode)
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit("training completed without metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if int(metrics.get("optimizer_steps") or 0) != int(config["phases"][args.phase]["max_steps"]):
        raise SystemExit("training metrics do not match configured optimizer steps")
    print(json.dumps({"phase": args.phase, "completed": True, "metrics": str(metrics_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
