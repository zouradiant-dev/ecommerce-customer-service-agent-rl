"""Hash-checked reward fix deployment with backups; never starts GPU jobs."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def checked_path(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
        raise ValueError("unsafe_bundle_path")
    target = root / relative
    target.resolve().relative_to(root)
    if any(path.is_symlink() for path in (target, *target.parents) if path != root and root in path.parents):
        raise ValueError("symlinked_bundle_or_target_path")
    return target


def install(package: Path, repo: Path):
    package, repo = package.resolve(), repo.resolve()
    manifest = json.loads((package / "bundle.json").read_text(encoding="utf-8"))
    if not (repo / "src/ecommerce_customer_service/tools.py").is_file():
        raise ValueError("target_is_not_ecommerce_repository")
    proc = Path("/proc")
    if proc.is_dir():
        jobs = ("train_sft.py", "train_grpo.py", "train_grpo_smoke_v12_r1.py",
                "evaluate_local_hf.py", "sample_train_readiness.py", "rescore_reward_v12.py")
        for directory in proc.iterdir():
            if not directory.name.isdigit():
                continue
            try:
                command = (directory / "cmdline").read_bytes().decode(errors="replace")
                cwd = (directory / "cwd").resolve()
                if (cwd == repo or str(repo) + "/scripts/" in command) and any(job in command for job in jobs):
                    raise RuntimeError(f"active_project_job:{directory.name}; wait before deploying")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    for relative, expected in manifest["frozen_files"].items():
        if digest(checked_path(repo, relative)) != expected:
            raise ValueError(f"business_tools_changed:{relative}; no installation performed")
    planned = []
    for relative, expected in manifest["payload_sha256"].items():
        source = checked_path(package / "payload", relative)
        target = checked_path(repo, relative)
        if digest(source) != expected:
            raise ValueError(f"payload_modified:{relative}")
        if target.exists():
            current = digest(target)
            if current == expected:
                continue
            if current not in manifest["previous_sha256"].get(relative, []):
                raise ValueError(f"unexpected_existing_file:{relative}; preserve and review it")
        planned.append((relative, source, target))
    if not planned:
        return {"installed": True, "already_installed": True, "changed_files": 0, "gpu_started": False}
    backup_parent = repo / "outputs/code-backups"
    backup_parent.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="reward-fix-v12-", dir=backup_parent))
    before = {}
    for relative, _, target in planned:
        before[relative] = digest(target) if target.exists() else None
        if target.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    (backup / "backup.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    for relative, source, target in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != manifest["payload_sha256"][relative]:
            raise RuntimeError(f"copy_verification_failed; originals retained at {backup}")
    return {"installed": True, "changed_files": len(planned), "backup": str(backup),
            "verifier_version": manifest["verifier_version"], "original_reports_unchanged": True,
            "gpu_started": False, "formal_rl_gate_passed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.package, args.repo), ensure_ascii=False, indent=2))
