"""Offline, byte-level verification of the saved v18 inference bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_ID = "ecommerce-v18-sft"
ARCHIVE_SHA256 = "24695aace5336e68fc7c947797339c2c19f423862f2360186872f95dfc299d6f"
FILE_HASHES = {
    "adapter_model.safetensors": "f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c",
    "adapter_config.json": "d3c421aa3bc1b4ae4cefe760a9db5dda54fa42a2a54fa491372233d2c3c7b094",
    "tokenizer_config.json": "7e88129d9769a0b14b1587a7d5e829fe93ac0e1511636471fdfc0811951418e6",
    "tokenizer.json": "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "added_tokens.json": "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_adapter(directory: Path) -> dict:
    directory = directory.resolve(strict=True)
    hashes = {}
    for name, expected in FILE_HASHES.items():
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing_or_linked_artifact:{name}")
        hashes[name] = file_sha256(path)
        if hashes[name] != expected:
            raise ValueError(f"artifact_hash_mismatch:{name}")
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    if (config.get("base_model_name_or_path") != BASE_MODEL or config.get("peft_type") != "LORA"
            or config.get("task_type") != "CAUSAL_LM" or config.get("r") != 16
            or config.get("lora_alpha") != 32):
        raise ValueError("adapter_configuration_mismatch")
    return {"schema_version": "v18_inference_artifact_v1", "adapter_verified": True,
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "model_id": MODEL_ID,
            "files_sha256": hashes, "base_weights_in_bundle": False,
            "gpu_inference_verified": False, "online_quality_verified": False}


def extract_verified_archive(archive: Path, destination: Path) -> dict:
    # Only copy explicitly pinned regular files; never execute/extract pickle or tar links.
    if file_sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("inference_archive_hash_mismatch")
    if destination.exists():
        raise FileExistsError("destination_exists; verify existing directory instead")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v18-verify-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "adapter"
        stage.mkdir()
        seen = set()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                    raise ValueError("unsafe_archive_path")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("archive_links_or_special_files_not_allowed")
                if len(relative.parts) != 2 or relative.parts[0] != "v18-sft-full-newgpu-r1":
                    raise ValueError("unexpected_archive_layout")
                name = relative.name
                if name not in FILE_HASHES:
                    continue
                if name in seen or member.size > 200_000_000:
                    raise ValueError("duplicate_or_oversized_artifact")
                seen.add(name)
                with bundle.extractfile(member) as source, (stage / name).open("xb") as target:
                    shutil.copyfileobj(source, target)
        report = verify_adapter(stage)
        # rename fails if a competing preparation created the target; no replacement.
        stage.rename(destination)
    return {**report, "archive_sha256": ARCHIVE_SHA256}


def _tokenizer_signature(directory: Path) -> dict:
    """Compare tokenizer behavior, not version-dependent JSON formatting."""
    tokenizer = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    model = tokenizer.get("model") or {}
    vocab = model.get("vocab")
    if (model.get("type") != "BPE" or not isinstance(vocab, dict) or not vocab
            or any(not isinstance(token, str) or type(index) is not int or not 0 <= index < 152064
                   for token, index in vocab.items()) or len(set(vocab.values())) != len(vocab)):
        raise ValueError("base_tokenizer_mismatch:invalid_vocabulary")
    merges = []
    for value in model.get("merges", []):
        pair = value.split(" ") if isinstance(value, str) else value
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(token, str) for token in pair):
            raise ValueError("base_tokenizer_mismatch:invalid_merges")
        merges.append(pair)
    added = []
    defaults = {"single_word": False, "lstrip": False, "rstrip": False, "normalized": True, "special": False}
    for token in tokenizer.get("added_tokens", []):
        if (not isinstance(token, dict) or not isinstance(token.get("content"), str)
                or type(token.get("id")) is not int or not 0 <= token["id"] < 152064):
            raise ValueError("base_tokenizer_mismatch:invalid_added_tokens")
        added.append({"id": token["id"], "content": token["content"],
                      **{key: token.get(key, default) for key, default in defaults.items()}})
    if len({token["id"] for token in added}) != len(added) or len({token["content"] for token in added}) != len(added):
        raise ValueError("base_tokenizer_mismatch:duplicate_added_tokens")
    if json.loads((directory / "vocab.json").read_text(encoding="utf-8")) != vocab:
        raise ValueError("base_tokenizer_mismatch:inconsistent_vocab_file")
    merge_lines = (directory / "merges.txt").read_text(encoding="utf-8").splitlines()
    if merge_lines and merge_lines[0].startswith("#version:"):
        merge_lines = merge_lines[1:]
    # '#' is a real vocabulary token; '# #' is a merge rule, not a comment.
    file_merges = [line.split(" ") for line in merge_lines if line.strip()]
    if file_merges != merges:
        raise ValueError("base_tokenizer_mismatch:inconsistent_merges_file")
    model_defaults = {"unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None,
                      "dropout": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False}
    return {"vocab": vocab, "merges": merges, "added_tokens": sorted(added, key=lambda token: token["id"]),
            "model_options": {key: model.get(key, default) for key, default in model_defaults.items()},
            **{key: tokenizer.get(key) for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder")}}


def verify_tokenizer_compatibility(base: Path, adapter: Path) -> None:
    expected, actual = _tokenizer_signature(adapter), _tokenizer_signature(base)
    for field in expected:
        if actual[field] != expected[field]:
            raise ValueError(f"base_tokenizer_mismatch:{field}")


def verify_local_base(directory: Path, adapter: Path) -> None:
    """Require the pinned HF snapshot and matching vocabulary, without downloading."""
    directory = directory.resolve(strict=True)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if directory.name != BASE_REVISION and config.get("_commit_hash") != BASE_REVISION:
        raise ValueError("base_revision_unverified; use the pinned Hugging Face snapshot directory")
    if config.get("model_type") != "qwen2" or config.get("vocab_size") != 152064:
        raise ValueError("base_configuration_mismatch")
    # Training's save_pretrained can change formatting and string/pair merge
    # representation. The actual inference tokenizer remains the SHA-pinned adapter.
    verify_tokenizer_compatibility(directory, adapter)
    if not (directory / "model.safetensors").is_file():
        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index.get("weight_map", {}).values())
        if not shards or any(Path(name).name != name or not (directory / name).is_file() for name in shards):
            raise ValueError("missing_or_invalid_base_weight_shards")
    # Snapshot path/config identifies a revision; it is not a cryptographic hash of every base shard.
