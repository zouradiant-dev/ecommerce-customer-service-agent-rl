from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.serving.connection_check import check_endpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify served v18 identity; optional real tool-call probe (no writes)")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--generate", action="store_true", help="Run one actual model generation and require list_orders({})")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "v18-endpoint-check.json")
    args = parser.parse_args()
    try:
        report = check_endpoint(args.api_base, os.environ.get("MODEL_API_KEY", ""), generate=args.generate)
        code = 0
    except Exception as exc:
        report = {"service_identity_checked": False, "generation_checked": False,
                  "online_quality_verified": False, "error_type": type(exc).__name__}
        # Do not print URL/credentials or uncontrolled server text from exceptions.
        code = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
