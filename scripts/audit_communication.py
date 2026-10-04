"""CPU-only preparation, opt-in paid judging, and shadow reports. No model reruns."""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.communication_audit import (
    canonical, prepare, sha256_file, summarize, verify_job, write_new,
)
from ecommerce_customer_service.data import load_jsonl


def select_requests(rows, limit):
    """Stable round-robin across rubric families, not only old failures."""
    if limit <= 0:
        raise ValueError("limit_must_be_positive")
    groups = defaultdict(list)
    for row in rows:
        key = tuple(r["id"] for r in row["payload"]["requirements"])
        groups[key].append(row)
    queues = [sorted(group, key=lambda r: r["request_id"]) for _, group in sorted(groups.items())]
    selected = []
    for i in range(max((len(group) for group in queues), default=0)):
        for group in queues:
            if i < len(group):
                selected.append(group[i])
                if len(selected) == limit:
                    return selected
    return selected


def reserve_yuan(row, input_price, output_price, max_tokens):
    # Byte count + framing allowance intentionally overestimates usual BPE input
    # tokens. This is a client estimate, NOT the provider's billing hard cap.
    input_bound = len(canonical(row["messages"]).encode("utf-8")) + 512
    return (Decimal(input_bound) * input_price + Decimal(max_tokens) * output_price) / Decimal(1_000_000)


class NoCredentialRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def invoke(api_base, api_key, model, row, max_tokens):
    body = {"model": model, "messages": row["messages"], "temperature": 0,
            "max_tokens": max_tokens, "stream": False, "enable_thinking": False,
            "response_format": {"type": "json_object"}}
    request = Request(api_base.rstrip("/") + "/chat/completions", data=canonical(body).encode("utf-8"),
                      headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST")
    # No automatic retries: an ambiguous timeout may already have been billed.
    with build_opener(NoCredentialRedirect()).open(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def run_api(job, output, *, execute, limit, budget, input_price, output_price, model,
            max_tokens=2048, api_base=None, api_key=None, call=invoke):
    protocol = verify_job(job)
    if any(not x.is_finite() or x <= 0 for x in (budget, input_price, output_price)) or not 256 <= max_tokens <= 8192:
        raise ValueError("invalid_budget_price_or_token_limit")
    selected = select_requests(load_jsonl(job / "requests.jsonl"), limit)
    # Fixed prices describe the <=32K input tier. Refuse larger byte bounds
    # rather than accidentally under-budgeting a higher pricing tier.
    if any(len(canonical(row["messages"]).encode("utf-8")) + 512 > 32_000 for row in selected):
        raise ValueError("input_byte_bound_exceeds_supported_32k_price_tier")
    costs = [reserve_yuan(row, input_price, output_price, max_tokens) for row in selected]
    affordable, reserved = [], Decimal(0)
    for row, cost in zip(selected, costs):
        if reserved + cost > budget:
            break
        affordable.append(row)
        reserved += cost
    plan = {"dry_run": not execute, "shadow_only": True, "model": model,
            "selected_requests": len(selected), "affordable_requests": len(affordable),
            "client_budget_yuan": str(budget), "conservative_reserved_yuan": str(reserved),
            "input_price_per_million": str(input_price), "output_price_per_million": str(output_price),
            "max_tokens": max_tokens, "job_version": protocol["version"],
            "billing_warning": "Client-side estimate, not a provider hard cap. Confirm model/region prices; no automatic retries."}
    if not execute:
        return plan
    if not affordable:
        raise ValueError("budget_cannot_cover_one_request")
    if not api_key:
        raise ValueError("missing_api_key_in_environment")
    parsed = urlsplit(api_base or "")
    if (parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".aliyuncs.com")
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("expected_explicit_https_bailian_api_base")
    job, output = job.resolve(), output.resolve()
    if output.exists() or job == output or output.is_relative_to(job) or job.is_relative_to(output):
        raise ValueError("new_separate_output_directory_required")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "run-protocol.json", {
        **plan, "api_base": api_base, "job_protocol_sha256": sha256_file(job / "protocol.json"),
        "input_data_leaves_machine": True,
        "price_authority": "Operator-provided prices must match the chosen model and region.",
    })
    grades, failure, actual_cost, unknown_billing = [], None, Decimal(0), 0
    with (output / "grades.jsonl").open("x", encoding="utf-8") as handle:
        for number, row in enumerate(affordable, 1):
            print(canonical({"event": "judge_start", "index": number, "planned": len(affordable)}), flush=True)
            usage_received = False
            try:
                response = call(api_base, api_key, model, row, max_tokens)
                usage = response.get("usage") or {}
                prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
                if any(type(n) is not int or n < 0 for n in (prompt_tokens, completion_tokens)):
                    raise ValueError("missing_or_invalid_billing_usage")
                usage_received = True
                actual_cost += (Decimal(prompt_tokens) * input_price + Decimal(completion_tokens) * output_price) / Decimal(1_000_000)
                choice = response["choices"][0]
                raw_content = choice["message"]["content"]
                try:
                    judgment = json.loads(raw_content) if choice.get("finish_reason") == "stop" else None
                except (json.JSONDecodeError, TypeError):
                    judgment = None
                if response.get("model") != model:
                    judgment = None
                grade = {"request_id": row["request_id"], "judgment": judgment, "usage": usage,
                         "finish_reason": choice.get("finish_reason"), "reported_model": response.get("model"),
                         "provider_request_id": response.get("id"), "raw_judge_content": raw_content}
                handle.write(canonical(grade) + "\n")
                handle.flush()
                grades.append(grade)
                print(canonical({"event": "judge_done", "completed": len(grades), "actual_cost_yuan": str(actual_cost)}), flush=True)
                if response.get("model") != model:
                    failure = "reported_judge_model_differs_from_frozen_model_stop"
                    break
                if actual_cost > budget or (prompt_tokens * input_price + completion_tokens * output_price) / Decimal(1_000_000) > reserve_yuan(row, input_price, output_price, max_tokens):
                    failure = "provider_usage_exceeded_client_estimate_stop"
                    break
            except Exception as exc:
                # Do not print provider bodies, URLs or credentials. Preserve partial output.
                failure = "api_request_stopped:" + type(exc).__name__
                unknown_billing += int(not usage_received)
                break
    report = summarize(job, grades)
    write_new(output / "shadow-report.json", report)
    summary = {**plan, "completed_requests": len(grades), "known_usage_cost_yuan": str(actual_cost),
               "unknown_billing_requests": unknown_billing, "budget_limited": len(affordable) < len(selected),
               "complete_selected_subset": failure is None and len(grades) == len(selected),
               "failure": failure, "status_counts": report["status_counts"],
               "benchmark_rewards_unchanged": True, "calibrated": False}
    write_new(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--rollouts", type=Path, action="append", required=True)
    prep.add_argument("--rubric", type=Path, default=ROOT / "configs/communication-rubric-v1.json")
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--job", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execute", action="store_true", help="Explicitly authorize paid API calls and transfer of prepared data")
    run.add_argument("--limit", type=int, default=80)
    run.add_argument("--budget-yuan", type=Decimal, default=Decimal("5"))
    run.add_argument("--input-price", type=Decimal, default=Decimal("2.5"))
    run.add_argument("--output-price", type=Decimal, default=Decimal("10"))
    run.add_argument("--model", default="qwen3-max-2026-01-23")
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument("--api-base", default=os.environ.get("BAILIAN_API_BASE"))
    imp = commands.add_parser("import")
    imp.add_argument("--job", type=Path, required=True)
    imp.add_argument("--grades", type=Path, required=True)
    imp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.manifest, args.rollouts, args.rubric, args.output)
        result = {k: v for k, v in result.items() if k not in {"input_sha256", "output_sha256"}}
    elif args.command == "run":
        result = run_api(args.job, args.output, execute=args.execute, limit=args.limit,
                         budget=args.budget_yuan, input_price=args.input_price, output_price=args.output_price,
                         model=args.model, max_tokens=args.max_tokens, api_base=args.api_base,
                         api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))
    else:
        if args.output.exists() or args.output.resolve().is_relative_to(args.job.resolve()):
            raise ValueError("new_output_outside_job_required")
        result = summarize(args.job, load_jsonl(args.grades))
        write_new(args.output, result)
        result = {k: v for k, v in result.items() if k != "audits"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("failure"):
        raise SystemExit(2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
