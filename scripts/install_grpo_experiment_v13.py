"""Hash-checked deployment, with backups. Does not start training or edit reports/weights."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

PAYLOAD = {
    "src/ecommerce_customer_service/online_env.py",
    "src/ecommerce_customer_service/simulator.py",
    "src/ecommerce_customer_service/grpo_experiment_v13.py",
    "src/ecommerce_customer_service/grpo_memory_v13.py",
    "scripts/train_grpo_experiment_v13.py",
    "scripts/run_grpo_experiment_v13.sh",
    "scripts/prepare_grpo_v13_review.py",
    "scripts/evaluate_grpo_v13.py",
    "scripts/audit_grpo_smoke_result.py",
    "scripts/install_grpo_experiment_v13.py",
    "scripts/package_grpo_experiment_v13.py",
    "scripts/install_reward_fix_v12.py",
    "tests/test_grpo_experiment_v13.py",
    "tests/test_grpo_experiment_v13_deploy.py",
    "tests/test_grpo_experiment_v13_memory.py",
    "docs/grpo-experiment-v13.md",
}


def install(package, repo):
    package, repo = package.resolve(), repo.resolve()
    manifest = json.loads((package / "bundle.json").read_text(encoding="utf-8"))
    if (set(manifest["payload_sha256"]) != PAYLOAD or manifest.get("kind") != "bounded_grpo_experiment_v13"
            or manifest.get("legacy_default_version") != "deterministic_semantic_v12"
            or manifest.get("gpu_started") is not False or manifest.get("formal_rl_gate_passed") is not False):
        raise ValueError("unexpected_bounded_v13_bundle_contract")
    if set(manifest["bootstrap_sha256"]) != {"install_grpo_experiment_v13.py", "install_reward_fix_v12.py"}:
        raise ValueError("missing_bootstrap_hashes")
    for name, expected in manifest["bootstrap_sha256"].items():
        if name not in {"install_grpo_experiment_v13.py", "install_reward_fix_v12.py"}:
            raise ValueError("unexpected_bootstrap")
        path = package / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("bootstrap_modified:" + name)
    for relative, expected in manifest["evidence_sha256"].items():
        if relative.startswith("/") or ".." in Path(relative).parts or "\\" in relative or ":" in relative:
            raise ValueError("unsafe_evidence_path")
        path = package / "evidence" / relative
        if not path.resolve().is_relative_to(package / "evidence") or path.is_symlink():
            raise ValueError("unsafe_evidence_path")
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("evidence_modified:" + relative)
    review = package / "review-confirmed/review.json"
    if review.is_symlink() or sha256(review.read_bytes()).hexdigest() != manifest["review_sha256"]:
        raise ValueError("human_review_modified")
    # The caller's command is the one installation process, not a training job.
    proc = Path("/proc")
    if proc.is_dir():
        for directory in proc.iterdir():
            if not directory.name.isdigit():
                continue
            try:
                command = (directory / "cmdline").read_bytes().decode(errors="replace")
                cwd = (directory / "cwd").resolve()
                if ((cwd == repo or str(repo) + "/scripts/" in command)
                        and any(name in command for name in ("train_grpo_experiment_v13.py", "evaluate_grpo_v13.py", "rescore_reward_v13.py"))):
                    raise RuntimeError("active_project_job:" + directory.name)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    spec = importlib.util.spec_from_file_location("experiment_v13_shared_installer", package / "install_reward_fix_v12.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    result = common.install(package, repo)
    result.update(legacy_default_version="deterministic_semantic_v12", new_explicit_entrypoint=True,
                  reports_and_weights_unchanged=True, gpu_started=False, formal_rl_gate_passed=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.package, args.repo), ensure_ascii=False, indent=2))
