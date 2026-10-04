"""Install a checked source-only bundle with recoverable backups; no GPU/API/pip."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def install(package: Path, repo: Path):
    package, repo = package.resolve(), repo.resolve()
    manifest = json.loads((package / "bundle.json").read_text(encoding="utf-8"))
    if not (repo / "src" / "ecommerce_customer_service" / "simulator.py").is_file():
        raise ValueError("target is not the ecommerce repository")
    # Do not change imported source underneath an active project job.
    proc = Path("/proc")
    if proc.is_dir():
        for directory in proc.iterdir():
            if not directory.name.isdigit():
                continue
            try:
                command = (directory / "cmdline").read_bytes().decode(errors="replace")
                cwd = (directory / "cwd").resolve()
                if cwd == repo and any(name in command for name in (
                    "scripts/evaluate_local_hf.py", "scripts/train_sft.py", "scripts/train_grpo.py",
                    "scripts/sample_train_readiness.py")):
                    raise RuntimeError(f"active project job PID {directory.name}; finish it before deploying")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    for relative, expected in manifest["frozen_files"].items():
        path = (repo / relative).resolve()
        path.relative_to(repo)
        if digest(path) != expected:
            raise ValueError(f"frozen v11 code differs; no installation performed:{relative}")
    planned = []
    for relative, expected in manifest["payload_sha256"].items():
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("unsafe bundle path")
        source, target = package / "payload" / relative, repo / relative
        source.resolve().relative_to(package / "payload")
        target.resolve().relative_to(repo)
        if source.is_symlink() or target.is_symlink() or digest(source) != expected:
            raise ValueError(f"payload missing, symlinked or modified:{relative}")
        if target.exists():
            current = digest(target)
            if current == expected:
                continue
            if current != manifest["previous_sha256"].get(relative):
                raise ValueError(f"unexpected existing file; preserve it and review before installation:{relative}")
        planned.append((relative, source, target))
    if not planned:
        return {"installed": True, "already_installed": True, "changed_files": 0}
    backup_parent = repo / "outputs" / "code-backups"
    backup_parent.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="train-readiness-v12-", dir=backup_parent))
    before = {}
    for relative, _, target in planned:
        if target.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
            before[relative] = digest(saved)
        else:
            before[relative] = None
    (backup / "backup.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    for relative, source, target in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != manifest["payload_sha256"][relative]:
            raise RuntimeError(f"copy verification failed; originals retained at {backup}")
    return {"installed": True, "backup": str(backup), "changed_files": len(planned),
            "frozen_v11_unchanged": True, "gpu_started": False, "packages_installed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.package, args.repo), ensure_ascii=False, indent=2))
