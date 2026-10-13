from pathlib import Path
import argparse
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.serving.context_builder import ContextBuilder, load_token_counter
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.online_evaluation import run_suite
from ecommerce_customer_service.serving.policy import PROFILES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-turn dev regression: fixtures for engineering, real endpoint for model evaluation")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "online-agent-cases-v1.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="A new output directory; existing runs are never overwritten")
    parser.add_argument("--mode", choices=["smoke", "model"], default="smoke")
    parser.add_argument("--model")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--context-budget", type=int, default=8192)
    parser.add_argument("--recent-turns", type=int, default=6)
    parser.add_argument("--context-policy", choices=["structured", "full-history"], default="structured")
    parser.add_argument("--policy-profile", choices=PROFILES, default="baseline_v1")
    args = parser.parse_args()
    if args.mode == "model" and not args.model:
        parser.error("--mode model requires --model")
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    counter = load_token_counter(args.tokenizer_path) if args.tokenizer_path else None
    context = ContextBuilder(counter=counter, max_input_tokens=args.context_budget, recent_turns=args.recent_turns, policy=args.context_policy)
    model = OpenAIChatModel(args.api_base, args.model, os.environ.get("MODEL_API_KEY", "")) if args.mode == "model" else None
    result = run_suite(cases, args.output, mode=args.mode, model=model, context=context, policy_profile=args.policy_profile)
    print(json.dumps({key: result[key] for key in ("evaluation_kind", "complete", "passed_cases", "evaluated_cases", "case_pass_rate")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["complete"] and result["passed_cases"] == result["evaluated_cases"] else 1)
