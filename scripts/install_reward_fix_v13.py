"""Deploy explicit-opt-in v13 with the existing hash/backup-protected installer.

Never changes training entry points, default reward version, or experiment files.
"""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

PAYLOAD = {
    "src/ecommerce_customer_service/verifier.py",
    "src/ecommerce_customer_service/rescoring.py",
    "src/ecommerce_customer_service/communication_facts_v13.py",
    "scripts/rescore_reward_v13.py",
    "tests/test_reward_v13.py",
    "tests/test_rescore_reward_v13.py",
    "docs/reward-fix-v13.md",
}


def install(package: Path, repo: Path):
    package, repo = package.resolve(), repo.resolve()
    manifest = json.loads((package / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("verifier_version") != "deterministic_semantic_v13" or set(manifest["payload_sha256"]) != PAYLOAD:
        raise ValueError("unexpected_v13_bundle_contract")
    if manifest.get("training_default_version") != "deterministic_semantic_v12":
        raise ValueError("unexpected_training_default_change")
    common = package / "install_reward_fix_v12.py"
    if common.is_symlink() or sha256(common.read_bytes()).hexdigest() != manifest["bootstrap_sha256"][common.name]:
        raise ValueError("shared_installer_modified")
    proc = Path("/proc")
    if proc.is_dir():
        for directory in proc.iterdir():
            if not directory.name.isdigit():
                continue
            try:
                command = (directory / "cmdline").read_bytes().decode(errors="replace")
                cwd = (directory / "cwd").resolve()
                if "rescore_reward_v13.py" in command and (cwd == repo or str(repo) + "/scripts/" in command):
                    raise RuntimeError(f"active_project_job:{directory.name}; wait before deploying")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    spec = importlib.util.spec_from_file_location("v13_shared_hash_checked_installer", common)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    result = installer.install(package, repo)
    result.update(verifier_version="deterministic_semantic_v13", training_default_version="deterministic_semantic_v12",
                  training_entrypoints_changed=False, gpu_started=False, formal_rl_gate_passed=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.package, args.repo), ensure_ascii=False, indent=2))
