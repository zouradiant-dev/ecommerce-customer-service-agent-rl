from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.webapp import serve_demo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the dependency-free Agent RL audit demo")
    parser.add_argument("--output", type=Path, default=Path("outputs/web-demo"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    serve_demo(
        args.output,
        host=args.host,
        port=args.port,
        replicas=args.replicas,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
