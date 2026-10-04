"""Package the reviewed, bounded v13 experiment, never launch a GPU or install dependencies."""
from copy import deepcopy
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.grpo_experiment_v13 import read_rows, validate_review
from install_grpo_experiment_v13 import PAYLOAD


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def build(baseline, evidence, review_directory, output, *, previous_release=None):
    baseline, evidence, review_directory, output = [path.resolve() for path in (baseline, evidence, review_directory, output)]
    archive = output.with_name(output.name + ".tar.gz")
    if output.exists() or archive.exists() or any(output == path or path in output.parents for path in (baseline, evidence, review_directory)):
        raise ValueError("new_non_nested_bundle_required")
    previous = json.loads((baseline / "bundle.json").read_text(encoding="utf-8"))
    if previous.get("verifier_version") != "deterministic_semantic_v12":
        raise ValueError("preserved_v12_baseline_required")
    for name, expected in previous["payload_sha256"].items():
        if digest(baseline / "payload" / name) != expected:
            raise ValueError("baseline_modified:" + name)
    release_hashes = {}
    release_manifest_hash = None
    if previous_release is not None:
        previous_release = previous_release.resolve()
        if output == previous_release or previous_release in output.parents:
            raise ValueError("new_non_nested_bundle_required")
        release = json.loads((previous_release / "bundle.json").read_text(encoding="utf-8"))
        if (release.get("kind") != "bounded_grpo_experiment_v13"
                or release.get("verifier_version") != "deterministic_semantic_v13"
                or release.get("legacy_default_version") != "deterministic_semantic_v12"
                or release.get("gpu_started") is not False or release.get("formal_rl_gate_passed") is not False
                or not set(release["payload_sha256"]).issubset(PAYLOAD)
                or set(release["bootstrap_sha256"]) != {"install_grpo_experiment_v13.py", "install_reward_fix_v12.py"}):
            raise ValueError("unexpected_previous_release_contract")
        for name, expected in release["payload_sha256"].items():
            if digest(previous_release / "payload" / name) != expected:
                raise ValueError("previous_release_modified:" + name)
            release_hashes[name] = expected
        for name, expected in release["bootstrap_sha256"].items():
            if digest(previous_release / name) != expected:
                raise ValueError("previous_release_bootstrap_modified:" + name)
            # A prior manual deployment may have copied a bootstrap helper into
            # scripts/. Only its verified exact prior bytes may be replaced.
            release_hashes.setdefault("scripts/" + name, expected)
        release_manifest_hash = digest(previous_release / "bundle.json")
    validate_review(review_directory / "review.json", read_rows(evidence / "sft/rollouts.jsonl"), evidence / "sft/rollouts.jsonl")
    # Require the already-deployed scoring patch. This package changes routing,
    # not semantics, legacy defaults, or the old smoke's step cap.
    frozen = {}
    for name in ("src/ecommerce_customer_service/verifier.py", "src/ecommerce_customer_service/communication.py",
                 "src/ecommerce_customer_service/communication_facts.py", "src/ecommerce_customer_service/communication_facts_v13.py",
                 "src/ecommerce_customer_service/tools.py", "src/ecommerce_customer_service/rewards.py",
                 "scripts/train_grpo_smoke_v12_r1.py", "src/ecommerce_customer_service/experiment_prompt.py",
                 "src/ecommerce_customer_service/grpo_readiness.py", "src/ecommerce_customer_service/models.py",
                 "src/ecommerce_customer_service/policies.py", "src/ecommerce_customer_service/training.py"):
        frozen[name] = digest(ROOT / name)
    output.mkdir(parents=True, exist_ok=False)
    hashes, prior = {}, {}
    for relative in sorted(PAYLOAD):
        target = output / "payload" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        hashes[relative] = digest(target)
        if relative in previous["payload_sha256"]:
            prior[relative] = [previous["payload_sha256"][relative]]
        # The original simulator was preserved as a generation-source hash.
        if relative == "src/ecommerce_customer_service/simulator.py":
            protocol = json.loads((evidence / "protocol.json").read_text(encoding="utf-8"))
            prior[relative] = [protocol["generation_source_sha256"][relative]]
        if relative in release_hashes:
            prior[relative] = sorted(set(prior.get(relative, []) + [release_hashes[relative]]))
    bootstrap = {}
    for name in ("install_reward_fix_v12.py", "install_grpo_experiment_v13.py"):
        shutil.copy2(ROOT / "scripts" / name, output / name)
        bootstrap[name] = digest(output / name)
    shutil.copytree(evidence, output / "evidence")
    shutil.copytree(review_directory, output / "review-confirmed")
    evidence_hashes = {path.relative_to(output / "evidence").as_posix(): digest(path)
                       for path in (output / "evidence").rglob("*") if path.is_file()}
    shutil.copy2(ROOT / "docs/grpo-experiment-v13.md", output / "README.md")
    manifest = {"kind": "bounded_grpo_experiment_v13", "version": output.name,
        "verifier_version": "deterministic_semantic_v13", "legacy_default_version": "deterministic_semantic_v12",
        "payload_sha256": hashes, "previous_sha256": prior, "frozen_files": frozen,
        "bootstrap_sha256": bootstrap, "evidence_sha256": evidence_hashes,
        "previous_release_manifest_sha256": release_manifest_hash,
        "review_sha256": digest(output / "review-confirmed/review.json"),
        "weights_included": False, "original_reports_overwritten": False,
        "qualification_steps": 10, "experiment_steps": 100, "train_templates": 5, "experiment_tasks": 50,
        "gpu_started": False, "formal_rl_gate_passed": False}
    (output / "bundle.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(archive, "x:gz") as handle:
        handle.add(output, arcname=output.name)
    checksum = digest(archive)
    archive.with_name(archive.name + ".sha256").write_bytes((checksum + "  " + archive.name + "\n").encode("ascii"))
    with tarfile.open(archive, "r:gz") as handle:
        for path in output.rglob("*"):
            if path.is_file():
                member = handle.extractfile(output.name + "/" + path.relative_to(output).as_posix())
                if sha256(member.read()).hexdigest() != digest(path):
                    raise RuntimeError("archive_payload_verification_failed")
    return {"archive": str(archive), "sha256": checksum, "bytes": archive.stat().st_size,
            "archive_payload_verified": True, "human_signoff_included": True, "gpu_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--review-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-release", type=Path,
                        help="Verified preserved v13 release whose exact installed hashes may be upgraded.")
    args = parser.parse_args()
    print(json.dumps(build(args.baseline, args.evidence, args.review_directory, args.output,
                           previous_release=args.previous_release), ensure_ascii=False, indent=2))
