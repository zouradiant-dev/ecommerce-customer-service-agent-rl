"""Build an immutable, source-only server bundle; no data/model/report payloads."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = (
    "src/ecommerce_customer_service/simulator.py", "src/ecommerce_customer_service/local_hf.py",
    "src/ecommerce_customer_service/experiment_prompt.py", "src/ecommerce_customer_service/grpo_readiness.py",
    "src/ecommerce_customer_service/online_env.py", "scripts/evaluate_local_hf.py",
    "scripts/sample_train_readiness.py", "scripts/train_grpo.py", "scripts/run_train_readiness_v12.sh",
    "requirements-online-grpo.txt", "docs/grpo-readiness-v12.md", "tests/test_grpo_readiness.py",
)
PREVIOUS = {
    "src/ecommerce_customer_service/simulator.py": "aef3e61faa82685044cce0d8bc4c2dd822727b1f52fb44acee7298b63b56294e",
    "src/ecommerce_customer_service/local_hf.py": "7c6f80724fb97ae424555a5fe1d064c88ce46c6deea9f4feab4de9a23e8d80d1",
    "scripts/evaluate_local_hf.py": "05363dca6fe437183c6ab4a08f8d343ad8217311fa04d1090ddd5a82b45216b5",
    "scripts/train_grpo.py": "f8fd45256f7b607068cf35828371aee874d2d097ca418170e9879131ab8b270a",
    "src/ecommerce_customer_service/online_env.py": "ad9c28a578ec2adaec4d364a6ddfb88cf653bf2fe7042a2d7b00ece1723e9306",
    "requirements-online-grpo.txt": "8d43aa402b5b6dac7c67c4801a399b95dfe6f69113be6cb9d1efdc77532644c1",
}
FROZEN = {
    "src/ecommerce_customer_service/communication.py": "bee8621901d4ac2fa765a7eee89188824553bed3a8a3b578dddf68f8f3f19848",
    "src/ecommerce_customer_service/verifier.py": "7a015f9335c501b0d2fb893306f5b67c3277323756f09f4fac73f60b372eb0ae",
    "src/ecommerce_customer_service/tools.py": "e914d546aef7487ef97eeaa74dd1dcc50e477e1c4f2856353a224a5a7014e291",
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new bundle directory")
    args = parser.parse_args()
    directory = args.output.resolve()
    archive = directory.with_name(directory.name + ".tar.gz")
    if directory.exists() or archive.exists():
        raise FileExistsError("bundle and archive must be new")
    for relative, expected in FROZEN.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"frozen reward/tool code changed:{relative}")
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in PAYLOAD:
        target = directory / "payload" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        hashes[relative] = digest(target)
    shutil.copy2(ROOT / "scripts" / "install_train_readiness_v12.py", directory / "install_train_readiness_v12.py")
    (directory / "bundle.json").write_text(json.dumps({"version": directory.name, "payload_sha256": hashes,
        "previous_sha256": PREVIOUS, "frozen_files": FROZEN}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(archive, "x:gz") as tar:
        tar.add(directory, arcname=directory.name)
    print(json.dumps({"archive": str(archive), "sha256": digest(archive), "payload_files": len(PAYLOAD)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
