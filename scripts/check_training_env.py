from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


REQUIRED = ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "tokenizers")


def inspect_import(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "name": name,
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - the report must include binary/DLL import failures
        return {"name": name, "ok": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the CUDA/QLoRA environment before renting GPU time")
    parser.add_argument("--output", type=Path, default=Path("reports/training-env.json"))
    parser.add_argument("--allow-cpu", action="store_true", help="do not fail only because CUDA is absent")
    args = parser.parse_args()

    packages = [inspect_import(name) for name in REQUIRED]
    torch_info: dict[str, Any] = {
        "available": False,
        "cuda_available": False,
        "cuda_version": None,
        "devices": [],
        "bf16_supported": False,
    }
    try:
        import torch

        torch_info.update(
            {
                "available": True,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
                "devices": [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2),
                        "capability": list(torch.cuda.get_device_capability(index)),
                    }
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception as exc:  # noqa: BLE001
        torch_info["error"] = f"{type(exc).__name__}: {exc}"

    pip = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    package_ok = all(item["ok"] for item in packages)
    cuda_ok = bool(torch_info["cuda_available"]) or args.allow_cpu
    status_ok = package_ok and cuda_ok and pip.returncode == 0 and platform.architecture()[0] == "64bit"
    report = {
        "status_ok": status_ok,
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "packages": packages,
        "torch": torch_info,
        "pip_check": {"ok": pip.returncode == 0, "output": pip.stdout.strip()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
