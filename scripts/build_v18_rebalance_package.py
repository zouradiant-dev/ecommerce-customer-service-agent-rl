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

from ecommerce_customer_service.sft_v18_48gb import sha256_file  # noqa: E402


FILES = (
    "data/v18-sft-rebalance-r2/train.jsonl",
    "data/v18-sft-rebalance-r2/summary.json",
    "scripts/train_sft.py",
    "scripts/train_sft_v18_rebalance.py",
    "scripts/run_v18_rebalance.sh",
    "scripts/run_v18_rebalance_screen.sh",
    "scripts/evaluate_sft_v18.py",
    "tests/test_v18_supplement.py",
)


def build(output: Path) -> dict[str, object]:
    payload = {name: ROOT / name for name in FILES}
    missing = [name for name, path in payload.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("rebalance package inputs missing:" + ",".join(missing))
    entries = [
        {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(payload.items())
    ]
    receipt = {
        "schema_version": "ecommerce_cs_v18_rebalance_package_v2",
        "train_rows": 1680,
        "max_optimizer_steps": 28,
        "checkpoint_interval": 4,
        "dev_or_test_rows_in_train": 0,
        "required_initial_adapter_model_sha256": "f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c",
        "files": entries,
    }
    receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name, path in sorted(payload.items()):
            archive.add(path, arcname=name, recursive=False)
        info = tarfile.TarInfo("V18-REBALANCE-PACKAGE-MANIFEST.json")
        info.size = len(receipt_bytes)
        archive.addfile(info, io.BytesIO(receipt_bytes))
    with tarfile.open(output, "r:gz") as archive:
        for entry in entries:
            handle = archive.extractfile(entry["path"])
            if handle is None or sha256(handle.read()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"archive payload hash mismatch:{entry['path']}")
    return {
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "files": len(entries),
        "archive_payload_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the v18 28-step rebalance correction")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "v18-sft-rebalance-r2.tar.gz")
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
