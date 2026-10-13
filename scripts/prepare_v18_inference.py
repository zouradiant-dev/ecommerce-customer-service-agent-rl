from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.serving.artifacts import extract_verified_archive, verify_adapter, verify_local_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify/extract the pinned v18 adapter; no install, network or GPU")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "models" / "v18-sft-verified")
    parser.add_argument("--base-model-dir", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "v18-inference-preflight.json")
    args = parser.parse_args()
    report = (extract_verified_archive(args.archive, args.adapter_dir) if args.archive
              else verify_adapter(args.adapter_dir))
    if args.base_model_dir:
        verify_local_base(args.base_model_dir, args.adapter_dir)
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report.update(base_directory_checked=bool(args.base_model_dir), installed_packages=versions,
                  required_libraries_installed=all(versions[name] for name in ("torch", "transformers", "peft", "accelerate", "safetensors")),
                  environment_ready=False, note="GPU load/generation not checked by this offline preparation")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
