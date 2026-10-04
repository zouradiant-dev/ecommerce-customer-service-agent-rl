from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.evaluation_v18 import select_v18_dev_tasks, sha256_file  # noqa: E402
from ecommerce_customer_service.models import TaskSpec  # noqa: E402


CODE_FILES = (
    "configs/sft-v18-supplement-r1.json",
    "scripts/train_sft.py",
    "scripts/train_sft_v18_supplement.py",
    "scripts/run_v18_supplement.sh",
    "scripts/run_v18_sft_eval_supplement.sh",
    "scripts/evaluate_sft_v18.py",
    "src/ecommerce_customer_service/sft_v18_supplement.py",
    "src/ecommerce_customer_service/sft_v18_48gb.py",
    "src/ecommerce_customer_service/experiment_prompt_v18.py",
    "src/ecommerce_customer_service/evaluation_v18.py",
    "src/ecommerce_customer_service/communication.py",
    "src/ecommerce_customer_service/communication_v17.py",
    "src/ecommerce_customer_service/verifier_v17.py",
    "src/ecommerce_customer_service/verifier_v18.py",
    "tests/test_v18_supplement.py",
    "docs/v18-sft-full-dev-analysis-2026-10-04.md",
)


def build_package(output: Path) -> dict[str, object]:
    payload: dict[str, Path] = {relative: ROOT / relative for relative in CODE_FILES}
    supplement_dir = ROOT / "data/v18-sft-supplement-r1"
    for source in supplement_dir.rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts:
            payload[source.relative_to(ROOT).as_posix()] = source
    frozen_dev = ROOT / "data/v18-sft-reference-r1/dev.jsonl"
    payload[frozen_dev.relative_to(ROOT).as_posix()] = frozen_dev

    manifest = ROOT / "data/v18-train-dev-r1/tasks.jsonl"
    task_paths, plan = select_v18_dev_tasks(manifest)
    payload[manifest.relative_to(ROOT).as_posix()] = manifest
    for task_path in task_paths:
        task = TaskSpec.from_file(task_path)
        for source in (task_path, task.database_path):
            payload[source.relative_to(ROOT).as_posix()] = source

    missing = [relative for relative, path in payload.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("package inputs missing:" + ",".join(missing))
    entries = [
        {"path": relative, "bytes": source.stat().st_size, "sha256": sha256_file(source)}
        for relative, source in sorted(payload.items())
    ]
    receipt = {
        "schema_version": "ecommerce_cs_v18_supplement_package_v1",
        "purpose": "corrective_sft_from_existing_adapter_then_targeted_dev_evaluation",
        "train_rows": 1880,
        "focus_rows": 1280,
        "replay_rows": 600,
        "dev_or_test_rows_in_train": 0,
        "full_optimizer_steps": 118,
        "dev_tasks_included": len(task_paths),
        "dev_manifest_sha256": plan["manifest_sha256"],
        "required_initial_adapter": {
            "path": "outputs/v18-sft-full-newgpu-r1",
            "adapter_model_sha256": "f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c",
            "adapter_config_sha256": "d3c421aa3bc1b4ae4cefe760a9db5dda54fa42a2a54fa491372233d2c3c7b094",
        },
        "files": entries,
    }
    receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative, source in sorted(payload.items()):
            archive.add(source, arcname=relative, recursive=False)
        info = tarfile.TarInfo("V18-SUPPLEMENT-PACKAGE-MANIFEST.json")
        info.size = len(receipt_bytes)
        archive.addfile(info, io.BytesIO(receipt_bytes))

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        if "V18-SUPPLEMENT-PACKAGE-MANIFEST.json" not in names:
            raise RuntimeError("package manifest missing")
        for entry in entries:
            handle = archive.extractfile(entry["path"])
            if handle is None or sha256(handle.read()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"archive payload hash mismatch:{entry['path']}")
    return {
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "files": len(entries),
        "dev_tasks": len(task_paths),
        "archive_payload_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "v18-sft-supplement-r1.tar.gz")
    args = parser.parse_args()
    print(json.dumps(build_package(args.output.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
