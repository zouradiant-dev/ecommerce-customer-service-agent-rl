from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.sft_v18_48gb import sha256_file  # noqa: E402


FILES = (
    "src/ecommerce_customer_service/experiment_prompt_v18.py",
    "src/ecommerce_customer_service/evaluation_v18.py",
    "src/ecommerce_customer_service/policy_controller_v18.py",
    "scripts/evaluate_sft_v18.py",
    "scripts/run_v18_policy_controller_screen_r6.sh",
    "scripts/run_v18_policy_controller_calibration_r6.sh",
    "scripts/summarize_v18_full_replay_screen_r3.py",
    "tests/test_v18.py",
)


def build(output: Path) -> dict[str, object]:
    payload = {name: ROOT / name for name in FILES}
    missing = [name for name, path in payload.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("policy controller package inputs missing:" + ",".join(missing))
    entries = [
        {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(payload.items())
    ]
    receipt = {
        "schema_version": "ecommerce_cs_v18_policy_controller_package_r6",
        "learned_adapter_changed": False,
        "training_required": False,
        "controller_uses_frozen_terminal_answer": False,
        "controller_inputs": ["user_messages", "visible_tool_surface", "audited_tool_results"],
        "screen_rollouts": 70,
        "calibration_rollouts": 150,
        "files": entries,
    }
    receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name, path in sorted(payload.items()):
            archive.add(path, arcname=name, recursive=False)
        info = tarfile.TarInfo("V18-POLICY-CONTROLLER-R6-PACKAGE-MANIFEST.json")
        info.size = len(receipt_bytes)
        archive.addfile(info, io.BytesIO(receipt_bytes))
    with tarfile.open(output, "r:gz") as archive:
        for entry in entries:
            handle = archive.extractfile(entry["path"])
            if handle is None or sha256(handle.read()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"archive payload hash mismatch:{entry['path']}")
    return {
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "files": len(entries),
        "archive_payload_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Package v18 evidence-only policy controller")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "v18-policy-controller-r6.tar.gz")
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
