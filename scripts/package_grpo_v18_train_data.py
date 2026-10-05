"""Build and byte-audit the frozen train-only data package required by v18 GRPO."""
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

from ecommerce_customer_service.grpo_v18 import (  # noqa: E402
    EXPECTED_MANIFEST_SHA256,
    select_v18_train_tasks,
)
from ecommerce_customer_service.local_hf import sha256_file  # noqa: E402


def build_package(output: Path) -> dict[str, object]:
    manifest = ROOT / "data/v18-train-dev-r1/tasks.jsonl"
    selected, plan = select_v18_train_tasks(manifest, tasks_per_scenario=100)
    if len(selected) != 3000 or len(plan["scenarios"]) != 30:
        raise ValueError("expected frozen v18 train selection of 30 scenarios x 100 tasks")
    if plan["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("unexpected frozen v18 manifest")

    payload: dict[str, Path] = {
        manifest.relative_to(ROOT).as_posix(): manifest,
    }
    train_rows = ROOT / "data/v18-train-dev-r1/train.jsonl"
    if train_rows.is_file():
        payload[train_rows.relative_to(ROOT).as_posix()] = train_rows
    for task_path, task in selected:
        for source in (task_path, task.database_path):
            payload[source.relative_to(ROOT).as_posix()] = source
    if len(payload) not in (6001, 6002):
        raise ValueError(f"unexpected v18 train package file count:{len(payload)}")

    entries = [
        {"path": relative, "bytes": source.stat().st_size, "sha256": sha256_file(source)}
        for relative, source in sorted(payload.items())
    ]
    receipt = {
        "schema_version": "ecommerce_cs_v18_grpo_train_data_v1",
        "split": "train",
        "tasks": len(selected),
        "scenarios": len(plan["scenarios"]),
        "scenario_counts": plan["scenario_counts"],
        "manifest_sha256": plan["manifest_sha256"],
        "contains_dev_task_payloads": False,
        "contains_test": False,
        "files": entries,
    }
    receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative, source in sorted(payload.items()):
            archive.add(source, arcname=relative, recursive=False)
        info = tarfile.TarInfo("V18-GRPO-TRAIN-DATA-MANIFEST.json")
        info.size = len(receipt_bytes)
        archive.addfile(info, io.BytesIO(receipt_bytes))

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        if names != set(payload) | {"V18-GRPO-TRAIN-DATA-MANIFEST.json"}:
            raise RuntimeError("v18 GRPO train data package member set mismatch")
        for entry in entries:
            handle = archive.extractfile(entry["path"])
            if handle is None or sha256(handle.read()).hexdigest() != entry["sha256"]:
                raise RuntimeError("v18 GRPO train data payload hash mismatch:" + entry["path"])
    return {
        "archive": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "tasks": len(selected),
        "files": len(entries),
        "payload_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "v18-grpo-train-data-r1.tar.gz",
    )
    args = parser.parse_args()
    print(json.dumps(build_package(args.output.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
