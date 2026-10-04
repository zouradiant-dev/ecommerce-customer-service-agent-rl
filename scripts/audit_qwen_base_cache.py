"""Read-only CPU audit of the pinned Qwen snapshot; never loads a model on GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TARGET = "model.layers.26.mlp.up_proj.weight"
SOURCE = (
    "https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct/revision/"
    + REVISION + "?blobs=true"
)
# Official LFS metadata retrieved for this exact revision on 2026-10-03.
EXPECTED = {
    "model-00001-of-00004.safetensors": (
        "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
        3945441440,
    ),
    "model-00002-of-00004.safetensors": (
        "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
        3864726352,
    ),
    "model-00003-of-00004.safetensors": (
        "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
        3864726424,
    ),
    "model-00004-of-00004.safetensors": (
        "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
        3556377672,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def check_shard(path: Path, expected_hash: str, expected_size: int) -> dict:
    if not path.is_file():
        return {"file": path.name, "exists": False, "matches_official": False}
    digest = sha256_file(path)
    size = path.stat().st_size
    return {
        "file": path.name,
        "resolved_path": str(path.resolve()),
        "bytes": size,
        "sha256": digest,
        "expected_sha256": expected_hash,
        "matches_official": digest == expected_hash and size == expected_size,
    }


def inspect_disk_tensor(path: Path) -> dict:
    import torch
    from safetensors import safe_open

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(TARGET)
        finite = torch.isfinite(tensor)
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        finite_values = tensor[finite].float()
        return {
            "tensor": TARGET,
            "device": str(tensor.device),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "all_finite": nan_count + inf_count == 0,
            "max_finite_abs": (
                float(finite_values.abs().max().item()) if finite_values.numel() else None
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    hub = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))
    parser.add_argument(
        "--snapshot", type=Path,
        default=hub / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / REVISION,
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/base-cache-audit.json"))
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    target_shard = index["weight_map"][TARGET]
    if target_shard not in EXPECTED:
        raise ValueError(f"unexpected target shard: {target_shard}")

    report = {"revision": REVISION, "source": SOURCE, "snapshot": str(snapshot)}
    print(json.dumps({**report, "target_shard": target_shard}), flush=True)
    try:
        report["disk_tensor"] = inspect_disk_tensor(snapshot / target_shard)
    except Exception as exc:
        report["disk_tensor_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps({key: value for key, value in report.items() if key.startswith("disk_tensor")}), flush=True)

    shards = []
    names = [target_shard] + [name for name in EXPECTED if name != target_shard]
    for name in names:
        print(f"Hashing {name} ...", flush=True)
        expected_hash, expected_size = EXPECTED[name]
        entry = check_shard(snapshot / name, expected_hash, expected_size)
        shards.append(entry)
        print(json.dumps(entry), flush=True)
    report["shards"] = shards
    report["bad_shards"] = [entry["file"] for entry in shards if not entry["matches_official"]]
    report["all_shards_match_official"] = not report["bad_shards"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_shards_match_official": report["all_shards_match_official"],
        "bad_shards": report["bad_shards"],
        "receipt": str(args.output.resolve()),
    }), flush=True)
    return 0 if report["all_shards_match_official"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
