"""Sequential actual-model dev experiments; no model training or gold feedback."""
from pathlib import Path
import argparse
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.serving.connection_check import check_endpoint
from ecommerce_customer_service.serving.context_builder import ContextBuilder, load_token_counter
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.online_evaluation import run_suite
from ecommerce_customer_service.serving.policy import FACTORIAL_PROFILES, PROFILES


def main():
    parser = argparse.ArgumentParser(description="Paired dev prompt/description ablations against one real v18 endpoint")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "online-agent-cases-v1.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(FACTORIAL_PROFILES))
    args = parser.parse_args()
    if len(set(args.profiles)) != len(args.profiles) or args.profiles[0] != "baseline_v1":
        parser.error("profiles must be unique and begin with baseline_v1")
    key = os.environ.get("MODEL_API_KEY", "")
    identity = check_endpoint(args.api_base, key)
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or any(case.get("split") != "dev" for case in cases):
        parser.error("only nonempty dev cases are accepted")
    args.output.mkdir(parents=True, exist_ok=False)
    context = ContextBuilder(counter=load_token_counter(args.tokenizer_path), max_input_tokens=8192, recent_turns=6)
    model = OpenAIChatModel(args.api_base, identity["model_id"], key)
    (args.output / "endpoint-before.json").write_text(json.dumps(identity, indent=2)+"\n", encoding="utf-8")
    summaries = []
    for profile in args.profiles:
        print("START", profile, flush=True)
        result = run_suite(cases, args.output / profile, mode="model", model=model, context=context, policy_profile=profile)
        summaries.append(result)
        print("RESULT", profile, result["passed_cases"], "/", result["evaluated_cases"], "complete", result["complete"], flush=True)
        if not result["complete"] or any("model_or_turn_failed" in key for key in result["failures"]):
            print("STOP: model/infrastructure failure; do not rank incomplete results", flush=True)
            return 1
    after = check_endpoint(args.api_base, key)
    keys = ("model_id", "base_revision", "adapter_sha256", "tokenizer_sha256", "precision")
    if any(identity[key] != after[key] for key in keys):
        raise ValueError("endpoint_changed_during_ablation")
    (args.output / "endpoint-after.json").write_text(json.dumps(after, indent=2)+"\n", encoding="utf-8")
    outcomes = {}
    for profile in args.profiles:
        outcomes[profile] = {path.parent.name: json.loads(path.read_text(encoding="utf-8"))["passed"]
                            for path in (args.output / profile / "cases").glob("*/result.json")}
    baseline = outcomes[args.profiles[0]]
    comparisons = []
    for result in summaries:
        current = outcomes[result["policy_profile"]]
        comparisons.append({"profile": result["policy_profile"], "passed": result["passed_cases"],
            "cases": result["evaluated_cases"], "pass_rate": result["case_pass_rate"],
            "improved_cases": [case for case in baseline if not baseline[case] and current[case]],
            "regressed_cases": [case for case in baseline if baseline[case] and not current[case]],
            "model_calls": result["model_calls"], "tool_calls": result["tool_calls"]})
    report = {"evaluation_kind": "exploratory_paired_dev_ablation", "model_unchanged": True,
              "independent_test_claim": False, "model_training_performed": False,
              "single_greedy_trial": True, "comparisons": comparisons}
    (args.output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
