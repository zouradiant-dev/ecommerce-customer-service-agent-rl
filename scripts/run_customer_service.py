from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_customer_service.serving.api import OnlineServer, load_auth
from ecommerce_customer_service.serving.backend import HttpBusinessBackend, MockBusinessBackend
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.connection_check import check_endpoint
from ecommerce_customer_service.serving.context_builder import ContextBuilder, load_token_counter
from ecommerce_customer_service.serving.runtime import exclusive_runtime
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.storage import Store


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent customer service with confirmed, idempotent cancellation")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs" / "customer-service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--auth-file", type=Path)
    parser.add_argument("--mode", choices=["manual", "model"], default="manual")
    parser.add_argument("--model", help="Actual served model/LoRA adapter name; no automatic training or adapter loading")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--verify-v18-endpoint", action="store_true", help="Require pinned v18 identity and a real read-only tool probe before serving")
    parser.add_argument("--backend", choices=["mock", "http"], default="mock")
    parser.add_argument("--business-base")
    parser.add_argument("--tokenizer-path", help="Local tokenizer directory for exact chat-template token accounting; never downloads")
    parser.add_argument("--context-budget", type=int, default=8192, help="Input token budget, or conservative estimate without tokenizer")
    parser.add_argument("--recent-turns", type=int, default=6)
    parser.add_argument("--context-policy", choices=["structured", "full-history"], default="structured")
    args = parser.parse_args()
    if args.mode == "model" and not args.model:
        parser.error("--mode model requires --model")
    if args.verify_v18_endpoint:
        if args.mode != "model" or args.model != "ecommerce-v18-sft":
            parser.error("--verify-v18-endpoint requires --mode model --model ecommerce-v18-sft")
        check_endpoint(args.api_base, os.environ.get("MODEL_API_KEY", ""), generate=True)
    if args.backend == "http" and (not args.business_base or not args.auth_file or not os.environ.get("BUSINESS_API_KEY")):
        parser.error("HTTP backend requires --business-base, --auth-file and BUSINESS_API_KEY")
    with exclusive_runtime(args.data_dir):
        return serve(args)


def serve(args: argparse.Namespace) -> int:
    auth_path = args.auth_file or args.data_dir / "demo-auth.json"
    auth = load_auth(auth_path, create_demo=args.backend == "mock" and args.auth_file is None)
    backend = (MockBusinessBackend(args.data_dir / "business.sqlite") if args.backend == "mock" else
               HttpBusinessBackend(args.business_base, os.environ["BUSINESS_API_KEY"]))
    model = (OpenAIChatModel(args.api_base, args.model, os.environ.get("MODEL_API_KEY", ""))
             if args.mode == "model" else None)
    counter = load_token_counter(args.tokenizer_path) if args.tokenizer_path else None
    context = ContextBuilder(counter=counter, max_input_tokens=args.context_budget, recent_turns=args.recent_turns, policy=args.context_policy)
    service = CustomerService(Store(args.data_dir / "sessions.sqlite"), backend, model, context_builder=context)
    recovered = service.recover()
    demo_accounts = ({"alice": "customer_alice", "bob": "customer_bob"}
                     if args.backend == "mock" and args.auth_file is None
                     and args.host in {"127.0.0.1", "localhost"} else None)
    server = OnlineServer((args.host, args.port), service, auth, demo_accounts=demo_accounts)
    print(f"Customer service: http://{args.host}:{server.server_port}")
    print(f"Mode={args.mode}; backend={args.backend}; auth file={auth_path.resolve()}")
    print(f"Recovery={recovered}; data directory={args.data_dir.resolve()}")
    print("Local prototype: one service process per data directory. Enterprise identity/TLS/monitoring are not included.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
