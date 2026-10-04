"""SFT-only train pilot in the existing CUDA environment; never starts training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_local_hf import collect_stage, load_backend, write_json
from ecommerce_customer_service.experiment_prompt import EXPERIMENT_SYSTEM_PROMPT, PROMPT_VERSION
from ecommerce_customer_service.grpo_readiness import DEFAULT_TEMPLATES, sampling_checks, select_train_tasks
from ecommerce_customer_service.local_hf import sha256_file, validate_lora_adapter
from ecommerce_customer_service.simulator import USER_REPLY_PROTOCOL_VERSION
from ecommerce_customer_service.verifier import VERIFIER_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--templates", nargs="+", default=list(DEFAULT_TEMPLATES))
    parser.add_argument("--tasks-per-template", type=int, default=1)
    parser.add_argument("--trials-per-task", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    import math
    if args.trials_per_task < 2 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("positive sampling temperature and at least two trials required")
    if min(args.tasks_per_template, args.max_steps, args.max_new_tokens, args.max_input_tokens) < 1 or args.device < 0:
        parser.error("invalid count or device")
    return args


def main():
    args = parse_args()
    args.manifest, args.adapter, args.output = args.manifest.resolve(), args.adapter.resolve(), args.output.resolve()
    validate_lora_adapter(args.adapter, args.base_model)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (args.adapter / name).is_file():
            raise FileNotFoundError(args.adapter / name)
    selected = select_train_tasks(args.manifest, templates=tuple(args.templates), per_template=args.tasks_per_template)
    args.database_hashes = {task.task_id: sha256_file(task.database_path) for _, task in selected}
    args.system_prompt = EXPERIMENT_SYSTEM_PROMPT
    protocol = {
        "kind": "train_readiness_diagnostic", "split": "train", "stage": "sft_only",
        "tasks": len(selected), "task_ids": [task.task_id for _, task in selected],
        "template_ids": args.templates, "trials_per_task": args.trials_per_task,
        "rollouts": len(selected) * args.trials_per_task,
        "base_model": args.base_model, "adapter": str(args.adapter),
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "source_database_sha256": args.database_hashes,
        "prompt_version": PROMPT_VERSION, "system_prompt": args.system_prompt,
        "prompt_sha256": sha256(args.system_prompt.encode()).hexdigest(),
        "tool_surface": "native_dynamic", "scripted_dialogue": False,
        "verifier_version": VERIFIER_VERSION, "user_reply_protocol_version": USER_REPLY_PROTOCOL_VERSION,
        "source_sha256": {relative: sha256_file(ROOT / relative) for relative in (
            "src/ecommerce_customer_service/simulator.py", "src/ecommerce_customer_service/local_hf.py",
            "src/ecommerce_customer_service/experiment_prompt.py", "src/ecommerce_customer_service/grpo_readiness.py",
            "src/ecommerce_customer_service/communication.py", "src/ecommerce_customer_service/verifier.py",
            "src/ecommerce_customer_service/communication_facts.py",
            "src/ecommerce_customer_service/tools.py", "scripts/sample_train_readiness.py", "scripts/evaluate_local_hf.py")},
        "temperature": args.temperature, "sampling_top_p": 1.0, "sampling_top_k": 0,
        "seed": args.seed, "max_steps": args.max_steps, "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens, "prompt_truncation": False,
        "correction_retries": 0, "json_repair": False, "cached_only": not args.allow_download,
        "formal_rl_gate_passed": False,
    }
    print(json.dumps({"event": "TRAIN_PREFLIGHT_OK", "protocol": protocol}, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    model, completer, info = load_backend(args)
    write_json(args.output / "model_info.json", info)
    started = time.monotonic()
    rows, report = collect_stage("sft", selected, completer, args)
    checks = sampling_checks(rows, trials_per_task=args.trials_per_task, expected_task_ids=protocol["task_ids"])
    write_json(args.output / "readiness.json", checks)
    review = {
        "purpose": "Review actual tool facts, state verification, policy and final response for BOTH rewards; never rubber-stamp.",
        "reviewer": None, "reviewed_at_utc": None,
        "rollouts_sha256": sha256_file(args.output / "sft" / "rollouts.jsonl"),
        "rows": [{"task_id": row["task_id"], "trial": row["trial"], "recorded_reward": row["reward"],
                  "status": "pending", "expected_reward": None, "evidence": ""} for row in rows],
    }
    write_json(args.output / "reward_review.json", review)
    with (args.output / "reward_review.md").open("x", encoding="utf-8") as handle:
        handle.write("# Train 奖励核查（待审核，不是新评分）\n\n请同时审核判 1 和判 0。工具结果、状态核查、政策与最终回答必须一致；疑似误判不自动进入 GRPO。完整轨迹见 sft/rollouts.jsonl。\n\n")
        for index, row in enumerate(rows, 1):
            handle.write(f"## {index}. {row['task_id']} / trial {row['trial']} / reward {row['reward']}\n\n")
            handle.write(json.dumps({"verifier": row["verifier"], "tool_executions": row["tool_executions"]}, ensure_ascii=False, indent=2))
            handle.write("\n\n最终回答：\n\n" + row["final_response"] + "\n\n")
    summary = {"complete": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "protocol": protocol, "model_info": info, "report": report, "sampling_checks": checks,
               "elapsed_seconds": time.monotonic() - started,
               "interpretation": "Train-only diagnostic, pending reward review; not a formal RL gate or held-out performance."}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"event": "TRAIN_READINESS_COMPLETE", "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
