"""Build and byte-audit the source-only v18 GRPO deployment patch."""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_customer_service.grpo_v18 import (  # noqa: E402
    CALIBRATION_RUNTIME_SOURCES,
    TRAINING_RUNTIME_SOURCES,
)
from ecommerce_customer_service.local_hf import sha256_file  # noqa: E402


EXTRA_FILES = (
    ".gitattributes",
    "README.md",
    "requirements-online-grpo.txt",
    "scripts/audit_grpo_smoke_result.py",
    "scripts/audit_v18_reward_calibration.py",
    "scripts/package_grpo_v18.py",
    "scripts/run_grpo_v18.sh",
    "scripts/run_v18_grpo_calibration.sh",
    "docs/server-experiments.md",
    "docs/v18-project-issues-solutions-and-handoff.md",
    "tests/test_grpo_v18.py",
    "tests/test_v18.py",
    "tests/test_evaluation_v18.py",
)


def build_package(output: Path) -> dict:
    relative_files = sorted(set(CALIBRATION_RUNTIME_SOURCES + TRAINING_RUNTIME_SOURCES + EXTRA_FILES))
    payload = {relative: ROOT / relative for relative in relative_files}
    missing = [relative for relative, path in payload.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing package sources:" + ",".join(missing))
    entries = [
        {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for relative, path in payload.items()
    ]
    manifest = {
        "schema_version": "ecommerce_cs_v18_grpo_patch_v1",
        "source_only": True,
        "contains_model_weights": False,
        "contains_task_data": False,
        "qualification_steps": 10,
        "experiment_steps": 100,
        "formal_rl_gate_passed": False,
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative, path in payload.items():
            archive.add(path, arcname=relative, recursive=False)
        info = tarfile.TarInfo("V18-GRPO-PATCH-MANIFEST.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        if names != set(payload) | {"V18-GRPO-PATCH-MANIFEST.json"}:
            raise RuntimeError("v18 GRPO package member set mismatch")
        for entry in entries:
            handle = archive.extractfile(entry["path"])
            if handle is None or sha256(handle.read()).hexdigest() != entry["sha256"]:
                raise RuntimeError("v18 GRPO package payload hash mismatch:" + entry["path"])
    return {
        "archive": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "files": len(entries),
        "payload_verified": True,
        "formal_rl_gate_passed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "v18-grpo-training-interface-r1.tar.gz",
    )
    args = parser.parse_args()
    print(json.dumps(build_package(args.output.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
