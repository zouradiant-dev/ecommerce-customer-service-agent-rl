"""Package only reviewed source changes, binding originals to yesterday's backup."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = (
    "src/ecommerce_customer_service/communication.py", "src/ecommerce_customer_service/communication_facts.py",
    "src/ecommerce_customer_service/verifier.py", "src/ecommerce_customer_service/online_env.py",
    "src/ecommerce_customer_service/rescoring.py", "scripts/train_grpo.py", "scripts/train_grpo_smoke_v12_r1.py",
    "scripts/sample_train_readiness.py", "scripts/evaluate_local_hf.py", "scripts/rescore_reward_v12.py",
    "scripts/install_reward_fix_v12.py",
    "tests/test_reward_v12.py", "tests/test_reward_fix_deploy.py", "tests/test_pipeline.py",
    "tests/test_local_hf.py", "tests/test_evaluation_v9.py", "tests/test_evaluation_v11.py",
    "docs/reward-fix-v12.md",
)


def digest(path):
    checksum = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verified_server_baseline(archive, verification, relatives):
    """Accept only exact files from the separately verified server backup."""
    evidence = json.loads(verification.read_text(encoding="utf-8"))
    if not evidence.get("experiment_backup_verified") or not evidence.get("all_regular_files_read"):
        raise ValueError("server_baseline_not_verified")
    actual = digest(archive)
    if actual != evidence.get("sha256_actual") or actual != evidence.get("sha256_expected"):
        raise ValueError("server_baseline_archive_hash_mismatch")
    previous = {}
    with tarfile.open(archive, "r:") as baseline:
        for relative in sorted(relatives):
            name = "ecommerce-customer-service-agent-rl/" + relative
            recorded = evidence["archive_file_manifest"].get(name)
            if recorded is None:
                continue
            member = baseline.getmember(name)
            if not member.isfile():
                raise ValueError("server_baseline_not_regular_file:" + relative)
            raw = baseline.extractfile(member).read()
            checksum = sha256(raw).hexdigest()
            if len(raw) != recorded["bytes"] or checksum != recorded["sha256"]:
                raise ValueError("server_baseline_file_hash_mismatch:" + relative)
            previous[relative] = sorted({checksum, sha256(raw.replace(b"\r\n", b"\n")).hexdigest()})
    return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, required=True)
    parser.add_argument("--server-baseline-tar", type=Path)
    parser.add_argument("--server-baseline-verification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.server_baseline_tar) != bool(args.server_baseline_verification):
        parser.error("server baseline archive and verification must be supplied together")
    directory = args.output.resolve()
    archive = directory.with_name(directory.name + ".tar.gz")
    if directory.exists() or archive.exists():
        raise FileExistsError("new_bundle_and_archive_required")
    previous = {}
    prefix = "workspace/ecommerce-customer-service-agent-rl/"
    frozen = {"src/ecommerce_customer_service/tools.py": digest(ROOT / "src/ecommerce_customer_service/tools.py")}
    with zipfile.ZipFile(args.baseline_zip) as baseline:
        for name, expected in frozen.items():
            if sha256(baseline.read(prefix + name)).hexdigest() != expected:
                raise ValueError("business_tools_changed_since_backup")
        for relative in PAYLOAD:
            member = prefix + relative
            if member in baseline.namelist():
                raw = baseline.read(member)
                previous[relative] = sorted({sha256(raw).hexdigest(), sha256(raw.replace(b"\r\n", b"\n")).hexdigest()})
    server_previous = {}
    if args.server_baseline_tar:
        server_previous = verified_server_baseline(
            args.server_baseline_tar, args.server_baseline_verification,
            set(PAYLOAD) | set(frozen),
        )
        for relative, expected in frozen.items():
            if server_previous.get(relative) != [expected]:
                raise ValueError("server_business_tools_changed_since_backup:" + relative)
        for relative in PAYLOAD:
            if relative in server_previous:
                previous[relative] = sorted(set(previous.get(relative, [])) | set(server_previous[relative]))
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in PAYLOAD:
        target = directory / "payload" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        hashes[relative] = digest(target)
    shutil.copy2(ROOT / "scripts/install_reward_fix_v12.py", directory / "install_reward_fix_v12.py")
    shutil.copy2(ROOT / "docs/reward-fix-v12.md", directory / "README.md")
    manifest = {"version": directory.name, "verifier_version": "deterministic_semantic_v12",
        "payload_sha256": hashes, "previous_sha256": previous, "frozen_files": frozen,
        "original_baseline": str(args.baseline_zip.resolve()),
        "server_baseline": str(args.server_baseline_tar.resolve()) if args.server_baseline_tar else None,
        "server_previous_sha256": server_previous,
        "formal_rl_gate_passed": False, "gpu_training_started": False}
    (directory / "bundle.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(archive, "x:gz") as handle:
        handle.add(directory, arcname=directory.name)
    checksum = digest(archive)
    # Write bytes so Windows does not translate LF into CRLF in Linux checksum lists.
    archive.with_name(archive.name + ".sha256").write_bytes(
        (checksum + "  " + archive.name + "\n").encode("ascii")
    )
    with tarfile.open(archive, "r:gz") as handle:
        for name, expected in hashes.items():
            if sha256(handle.extractfile(directory.name + "/payload/" + name).read()).hexdigest() != expected:
                raise RuntimeError("archive_payload_verification_failed")
    print(json.dumps({"archive": str(archive), "sha256": checksum, "payload_files": len(hashes),
                      "archive_payload_verified": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
