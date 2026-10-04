"""Build an evaluation patch with every frozen dev task and SQLite snapshot."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_customer_service.evaluation_v18 import select_v18_dev_tasks, sha256_file
from ecommerce_customer_service.models import TaskSpec

CODE_FILES = (
    "src/ecommerce_customer_service/communication.py",
    "src/ecommerce_customer_service/communication_v17.py",
    "src/ecommerce_customer_service/evaluation_v18.py",
    "src/ecommerce_customer_service/experiment_prompt_v18.py",
    "src/ecommerce_customer_service/local_hf.py",
    "src/ecommerce_customer_service/verifier_v18.py",
    "src/ecommerce_customer_service/verifier_v17.py",
    "scripts/evaluate_sft_v18.py",
    "scripts/rescore_sft_v18.py",
    "scripts/run_v18_sft_eval_calibration.sh",
    "scripts/run_v18_sft_eval_smoke.sh",
    "scripts/build_v18_eval_package.py",
    "tests/test_evaluation_v18.py",
    "docs/v18-sft-evaluation.md",
)


def build_package(output: Path) -> dict:
    manifest = ROOT / "data/v18-train-dev-r1/tasks.jsonl"
    task_paths, plan = select_v18_dev_tasks(manifest)
    payload = {relative: ROOT / relative for relative in CODE_FILES}
    payload[manifest.relative_to(ROOT).as_posix()] = manifest
    for path in task_paths:
        task = TaskSpec.from_file(path)
        for source in (path, task.database_path):
            payload[source.relative_to(ROOT).as_posix()] = source
    entries = [
        {"path": relative, "bytes": source.stat().st_size, "sha256": sha256_file(source)}
        for relative, source in sorted(payload.items())
    ]
    receipt = {
        "schema_version": "ecommerce_cs_v18_eval_package_v3",
        "dev_tasks": len(task_paths),
        "dev_task_json_files": len(task_paths),
        "dev_sqlite_files": len(task_paths),
        "scenario_counts": dict(Counter(TaskSpec.from_file(path).scenario for path in task_paths)),
        "contains_train_task_databases": False,
        "contains_test": False,
        "manifest_sha256": plan["manifest_sha256"],
        "files": entries,
    }
    receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative, source in sorted(payload.items()):
            archive.add(source, arcname=relative, recursive=False)
        info = tarfile.TarInfo("V18-EVAL-PACKAGE-MANIFEST.json")
        info.size = len(receipt_bytes)
        archive.addfile(info, io.BytesIO(receipt_bytes))
    # Audit archive bytes, rather than trusting paths from the build machine.
    with tarfile.open(output, "r:gz") as archive:
        for entry in entries:
            handle = archive.extractfile(entry["path"])
            if handle is None or sha256(handle.read()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"archive_payload_hash_mismatch:{entry['path']}")
    return {
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "dev_tasks": len(task_paths),
        "scenarios": len(receipt["scenario_counts"]),
        "task_json_and_sqlite_files": len(task_paths) * 2,
        "archive_payload_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "packages/v18-sft-evaluator-r3.tar.gz")
    args = parser.parse_args()
    print(json.dumps(build_package(args.output.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
