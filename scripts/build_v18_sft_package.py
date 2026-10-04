from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "packages" / "v18-sft-48gb-r1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file(relative: Path, output: Path) -> None:
    source = ROOT / relative
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _payload_files(config: dict) -> Iterable[Path]:
    yield Path("configs/sft-v18-48gb.json")
    yield Path("requirements-training.txt")
    yield Path("pyproject.toml")
    yield Path("scripts/train_sft.py")
    yield Path("scripts/train_sft_v18_48gb.py")
    for spec in (config.get("data") or {}).values():
        yield Path(spec["path"])
    for path in sorted((ROOT / "src" / "ecommerce_customer_service").glob("*.py")):
        yield path.relative_to(ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen portable v18 48G SFT package")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to replace existing package directory: {output}")
    output.mkdir(parents=True)
    config = json.loads((ROOT / "configs" / "sft-v18-48gb.json").read_text(encoding="utf-8"))
    copied: set[Path] = set()
    for relative in _payload_files(config):
        normalized = Path(relative.as_posix())
        if normalized in copied:
            continue
        copied.add(normalized)
        _copy_file(normalized, output)

    instructions = """# v18 48G SFT execution package

This package contains frozen v18 reference SFT train/dev rows. They were produced
by a deterministic verified controller and are not learned-model results. It
contains no test-v2 material, task SQLite copies, or reference rollout archive.

Run on a Linux CUDA server from this directory:

```bash
python -m pip install -r requirements-training.txt
python scripts/train_sft_v18_48gb.py preflight
python scripts/train_sft_v18_48gb.py smoke
# Inspect outputs/v18-sft-smoke-r1/metrics.json before continuing.
python scripts/train_sft_v18_48gb.py full
```

Install a CUDA-compatible PyTorch build before the requirements file. The
launcher requires at least 45 GiB reported GPU memory, BF16 support, exact frozen
hashes, and a completed 20-step smoke run before the 925-step full run. The base
model and LoRA parameters use FP32 after BF16 backward proved non-finite on
multiple otherwise valid micro-batches.
Micro-batch losses are backpropagated unscaled and the FP32 accumulated gradient
is averaged immediately before clipping and the optimizer update.
"""
    (output / "PACKAGE-CONTENTS.md").write_text(instructions, encoding="utf-8")

    entries = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "ecommerce_cs_v18_sft_package_v1",
        "profile_version": config["profile_version"],
        "contains_test_v2": False,
        "contains_task_databases": False,
        "contains_reference_rollouts": False,
        "files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    manifest_path = output / "PACKAGE-MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {"directory": str(output), "manifest": str(manifest_path), "files": len(entries) + 1}
    if not args.no_archive:
        archive = output.with_suffix(".tar.gz")
        if archive.exists():
            raise SystemExit(f"refusing to replace existing archive: {archive}")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(output, arcname=output.name)
        result["archive"] = str(archive)
        result["archive_bytes"] = archive.stat().st_size
        result["archive_sha256"] = sha256_file(archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
