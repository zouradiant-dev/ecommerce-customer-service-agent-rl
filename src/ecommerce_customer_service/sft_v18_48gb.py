from __future__ import annotations

from collections import Counter
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
from typing import Any

from .tasks_v18 import V18_SCENARIOS
from .training import validate_training_row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _check_frozen_file(root: Path, label: str, spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(spec["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"missing_frozen_input:{label}:{path}")
    actual = sha256_file(path)
    expected = str(spec["sha256"]).lower()
    if actual != expected:
        raise ValueError(f"frozen_input_hash_mismatch:{label}:{actual}:{expected}")
    return path, {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def _validate_split(path: Path, split: str, expected_rows: int) -> dict[str, Any]:
    count = 0
    scenarios: Counter[str] = Counter()
    tasks: set[str] = set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_json:{split}:{line_number}") from exc
            row_id = str(row.get("id") or "")
            metadata = dict(row.get("metadata") or {})
            failures = validate_training_row(row)
            if failures:
                raise ValueError(f"invalid_sft_row:{split}:{line_number}:{','.join(failures)}")
            if row.get("format_version") != "ecommerce_cs_agent_sft_v18":
                raise ValueError(f"wrong_format_version:{split}:{line_number}")
            if metadata.get("split") != split or metadata.get("test_material") is not False:
                raise ValueError(f"split_or_test_leak:{split}:{line_number}")
            if metadata.get("learned_model_result") is not False:
                raise ValueError(f"reference_row_mislabeled:{split}:{line_number}")
            if metadata.get("source") != "deterministic_reference_v18_r1":
                raise ValueError(f"unexpected_reference_source:{split}:{line_number}")
            task_id = str(row.get("task_id") or "")
            scenario = str(row.get("scenario") or "")
            if not task_id.startswith(f"v18_{split}_") or scenario not in V18_SCENARIOS:
                raise ValueError(f"namespace_or_scenario_mismatch:{split}:{line_number}")
            if not row_id or row_id in ids:
                raise ValueError(f"missing_or_duplicate_row_id:{split}:{line_number}:{row_id}")
            ids.add(row_id)
            tasks.add(task_id)
            scenarios[scenario] += 1
            count += 1
    if count != expected_rows:
        raise ValueError(f"row_count_mismatch:{split}:{count}:{expected_rows}")
    if set(scenarios) != set(V18_SCENARIOS):
        raise ValueError(f"scenario_coverage_mismatch:{split}")
    return {
        "rows": count,
        "tasks": len(tasks),
        "scenario_rows": dict(sorted(scenarios.items())),
        "test_rows": 0,
        "learned_model_rows": 0,
    }


def validate_v18_sft_bundle(config_path: Path, *, project_root: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = (project_root or config_path.parents[1]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("profile_version") != "ecommerce_cs_sft_v18_48gb_r1":
        raise ValueError("unexpected_profile_version")
    revision = str((config.get("model") or {}).get("revision") or "")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("model_revision_must_be_40_char_commit")

    frozen: dict[str, Any] = {}
    data = dict(config.get("data") or {})
    paths: dict[str, Path] = {}
    for label in (
        "reference_summary",
        "task_summary",
        "task_manifest",
        "train",
        "dev",
        "task_generator",
        "reference_generator",
    ):
        paths[label], frozen[label] = _check_frozen_file(root, label, dict(data[label]))

    splits = {
        split: _validate_split(paths[split], split, int(data[split]["rows"]))
        for split in ("train", "dev")
    }
    if {item for split in splits.values() for item in split["scenario_rows"]} != set(V18_SCENARIOS):
        raise ValueError("combined_scenario_coverage_mismatch")
    reference_summary = json.loads(paths["reference_summary"].read_text(encoding="utf-8"))
    task_summary = json.loads(paths["task_summary"].read_text(encoding="utf-8"))
    if not reference_summary.get("all_reference_strict_pass"):
        raise ValueError("reference_strict_pass_not_proven")
    if reference_summary.get("learned_model_result") is not False or int(reference_summary.get("test_tasks", -1)) != 0:
        raise ValueError("reference_summary_provenance_invalid")
    if task_summary.get("test_v2_created") is not False or int(task_summary.get("test_rows", -1)) != 0:
        raise ValueError("task_summary_test_leak")

    training = dict(config["training"])
    phases = dict(config["phases"])
    expected_steps = (splits["train"]["rows"] + int(training["effective_batch_size"]) - 1) // int(
        training["effective_batch_size"]
    )
    if int(phases["full"]["max_steps"]) != expected_steps:
        raise ValueError(f"full_step_count_not_one_nominal_epoch:{phases['full']['max_steps']}:{expected_steps}")
    return {
        "schema_version": "ecommerce_cs_v18_sft_preflight_v1",
        "profile_version": config["profile_version"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model": {"id": config["model"]["id"], "revision": revision},
        "frozen_inputs": frozen,
        "splits": splits,
        "full_max_steps": expected_steps,
        "test_rows": 0,
        "learned_model_rows": 0,
        "data_ready": True,
    }


def inspect_48gb_hardware(device: int, *, minimum_gib: float = 45.0) -> dict[str, Any]:
    required_packages = ("transformers", "accelerate", "peft", "bitsandbytes", "tokenizers", "huggingface-hub")
    versions: dict[str, str] = {}
    missing: list[str] = []
    for package in required_packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError("missing_training_dependencies:" + ",".join(missing))
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch_not_installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable")
    if device < 0 or device >= torch.cuda.device_count():
        raise RuntimeError(f"cuda_device_out_of_range:{device}:{torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(device)
    memory_gib = properties.total_memory / 1024**3
    if memory_gib < minimum_gib:
        raise RuntimeError(f"insufficient_gpu_memory:{memory_gib:.2f}:{minimum_gib:.2f}")
    with torch.cuda.device(device):
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    if not bf16_supported:
        raise RuntimeError("bf16_not_supported")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "packages": versions,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "device_name": properties.name,
        "memory_gib": round(memory_gib, 3),
        "bf16_supported": True,
        "hardware_ready": True,
    }
