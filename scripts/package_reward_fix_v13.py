"""Build a small, verified code-only patch from the preserved v12 package baseline."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
from install_reward_fix_v13 import PAYLOAD


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def build(baseline: Path, output: Path):
    baseline, output = baseline.resolve(), output.resolve()
    archive = output.with_name(output.name + ".tar.gz")
    if output.exists() or archive.exists() or output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError("new_non_nested_bundle_and_archive_required")
    previous = json.loads((baseline / "bundle.json").read_text(encoding="utf-8"))
    if previous["verifier_version"] != "deterministic_semantic_v12":
        raise ValueError("frozen_v12_baseline_required")
    for relative, expected in previous["payload_sha256"].items():
        if digest(baseline / "payload" / relative) != expected:
            raise ValueError("baseline_modified:" + relative)
    frozen_names = ("src/ecommerce_customer_service/communication.py", "src/ecommerce_customer_service/communication_facts.py",
                    "src/ecommerce_customer_service/online_env.py", "src/ecommerce_customer_service/tools.py")
    frozen = {}
    for relative in frozen_names:
        expected = previous["payload_sha256"].get(relative) or previous["frozen_files"][relative]
        if digest(ROOT / relative) != expected:
            raise ValueError("frozen_dependency_changed:" + relative)
        frozen[relative] = expected
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in sorted(PAYLOAD):
        target = output / "payload" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        hashes[relative] = digest(target)
    bootstrap = {}
    for name in ("install_reward_fix_v12.py", "install_reward_fix_v13.py"):
        shutil.copy2(ROOT / "scripts" / name, output / name)
        bootstrap[name] = digest(output / name)
    shutil.copy2(ROOT / "docs/reward-fix-v13.md", output / "README.md")
    manifest = {"version": output.name, "verifier_version": "deterministic_semantic_v13",
                "training_default_version": "deterministic_semantic_v12", "payload_sha256": hashes,
                "previous_sha256": {name: [previous["payload_sha256"][name]] for name in PAYLOAD
                                    if name in previous["payload_sha256"]},
                "frozen_files": frozen, "bootstrap_sha256": bootstrap,
                "original_v12_bundle_sha256": digest(baseline / "bundle.json"),
                "code_only": True, "reports_or_weights_included": False,
                "training_entrypoints_changed": False, "formal_rl_gate_passed": False, "gpu_started": False}
    (output / "bundle.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(archive, "x:gz") as handle:
        handle.add(output, arcname=output.name)
    checksum = digest(archive)
    archive.with_name(archive.name + ".sha256").write_bytes((checksum + "  " + archive.name + "\n").encode("ascii"))
    with tarfile.open(archive, "r:gz") as handle:
        for relative, expected in {**{"payload/" + name: value for name, value in hashes.items()}, **bootstrap}.items():
            if sha256(handle.extractfile(output.name + "/" + relative).read()).hexdigest() != expected:
                raise RuntimeError("archive_payload_verification_failed")
    return {"archive": str(archive), "sha256": checksum, "payload_files": len(hashes),
            "archive_payload_verified": True, "code_only": True, "gpu_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.baseline_package, args.output), ensure_ascii=False, indent=2))
