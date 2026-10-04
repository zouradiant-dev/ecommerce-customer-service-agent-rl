from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecommerce_customer_service.diagnostics import analyze_bad_cases
from ecommerce_customer_service.evaluation import compare_rollouts, evaluate_rollouts
from ecommerce_customer_service.local_hf import (
    LocalTransformersAgent, TransformersCompleter, paired_seed, select_dev_tasks,
    sha256_file, validate_lora_adapter,
)
from ecommerce_customer_service.simulator import USER_REPLY_PROTOCOL_VERSION, run_task
from ecommerce_customer_service.verifier import VERIFIER_VERSION


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline 4-bit Base/SFT paired dev-task evaluation (no vLLM/API)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing outputs are never overwritten")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tasks-per-template", type=int, default=1)
    selection.add_argument("--full-dev", action="store_true")
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-download", action="store_true", help="opt in to Hugging Face downloads; otherwise cached only")
    parser.add_argument("--preflight-only", action="store_true", help="validate files/selection without GPU or creating outputs")
    args = parser.parse_args()
    if min(args.trials_per_task, args.max_steps, args.max_new_tokens, args.max_input_tokens, args.tasks_per_template) < 1:
        parser.error("counts and token budgets must be positive")
    if args.device < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("device and temperature must be non-negative")
    if args.temperature == 0 and args.trials_per_task != 1:
        parser.error("greedy decoding only supports one trial; use positive temperature for multiple trials")
    return args


def collect_stage(stage: str, selected: list[Any], completer: Any, args: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = args.output / stage
    directory.mkdir()
    rows = []
    total = len(selected) * args.trials_per_task
    started = time.monotonic()
    with (directory / "rollouts.jsonl").open("x", encoding="utf-8") as handle:
        for _, task in selected:
            for trial in range(args.trials_per_task):
                rollout_seed = paired_seed(args.seed, task.task_id, trial)
                completer.seed(rollout_seed)
                print(json.dumps({"event": "task_start", "stage": stage, "task_id": task.task_id,
                                  "trial": trial, "index": len(rows) + 1, "total": total}, ensure_ascii=False), flush=True)
                row = run_task(task, LocalTransformersAgent(
                    completer, max_steps=args.max_steps,
                    system_prompt=getattr(args, "system_prompt", None),
                ), trial=trial).to_dict()
                row["local_evaluation"] = {"stage": stage, "seed": rollout_seed}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
                print(json.dumps({"event": "task_done", "stage": stage, "task_id": task.task_id,
                                  "trial": trial, "completed": len(rows), "total": total,
                                  "reward": row["reward"], "stopped_reason": row["stopped_reason"],
                                  "failures": row["verifier"]["failures"], "elapsed_seconds": time.monotonic() - started,
                                  "error": row["error"]}, ensure_ascii=False), flush=True)
                if row["stopped_reason"] == "worker_error":
                    raise RuntimeError(f"Infrastructure error; evaluation stopped, partial rows preserved at {directory}: {row['error']}")
    # run_task should never alter a source snapshot; audit after EACH stage.
    for _, task in selected:
        if sha256_file(task.database_path) != args.database_hashes[task.task_id]:
            raise RuntimeError(f"source_database_changed:{task.task_id}")
    messages = [message for row in rows for message in row["messages"]]
    report = {
        "evaluation": evaluate_rollouts(rows), "badcases": analyze_bad_cases(rows),
        "invalid_tool_call_rollouts": sum(row["stopped_reason"] == "invalid_tool_call" for row in rows),
        "max_steps_rollouts": sum(row["stopped_reason"] == "max_steps" for row in rows),
        "truncated_generations": sum(message.get("local_finish_reason") == "length" for message in messages),
        "verifier_component_rates": {
            key: sum(bool(row["verifier"][key]) for row in rows) / len(rows)
            for key in ("state_passed", "communication_passed", "policy_passed", "required_tools_passed")
        },
        "source_databases_unchanged": True, "elapsed_seconds": time.monotonic() - started,
    }
    write_json(directory / "report.json", report)
    return rows, report


def load_backend(args: Any) -> tuple[Any, Any, Any]:
    # Keep the known training environment; no package installation is performed.
    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError("A valid CUDA device is required")
    torch.cuda.set_device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=local_only)
    saved_tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    if tokenizer.get_vocab() != saved_tokenizer.get_vocab() or tokenizer.chat_template != saved_tokenizer.chat_template:
        raise ValueError("base_and_adapter_tokenizers_differ; paired protocol cannot use different prompts")
    # Tool-call markers must survive skip_special_tokens=True for this parser.
    for marker in ("<tool_call>", "</tool_call>"):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is None or token_id in tokenizer.all_special_ids:
            raise ValueError(f"unsupported_tool_call_token:{marker}")
    print(json.dumps({"event": "loading_model", "base_model": args.base_model, "adapter": str(args.adapter),
                      "quantization": "nf4_double_quant", "dtype": str(dtype), "cached_only": local_only}), flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, local_files_only=local_only, low_cpu_mem_usage=True,
        torch_dtype=dtype, attn_implementation="sdpa", device_map={"": args.device},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
    )
    if args.max_input_tokens + args.max_new_tokens > base.config.max_position_embeddings:
        raise ValueError("requested token budgets exceed the model's context window")
    saved_config = json.loads((args.adapter / "config.json").read_text(encoding="utf-8"))
    if saved_config.get("vocab_size") != base.config.vocab_size:
        raise ValueError("adapter_vocab_size_differs_from_base")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    # Fresh config avoids inheriting sampling-only settings from the model repo.
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else 1.0,
        top_p=1.0, top_k=0 if args.temperature > 0 else 50, num_beams=1, use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    completer = TransformersCompleter(model, tokenizer, torch, generation, device=f"cuda:{args.device}",
                                     max_input_tokens=args.max_input_tokens)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")}
    info = {"versions": versions, "gpu": torch.cuda.get_device_name(args.device),
            "quantization": "nf4_double_quant", "dtype": str(dtype), "base_revision": getattr(base.config, "_commit_hash", None),
            "tokenizer_sha256": sha256_file(args.adapter / "tokenizer.json"),
            "chat_template_sha256": sha256(tokenizer.chat_template.encode()).hexdigest()}
    return model, completer, info


def main() -> int:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.adapter = args.adapter.resolve()
    args.output = args.output.resolve()
    config = validate_lora_adapter(args.adapter, args.base_model)
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (args.adapter / filename).is_file():
            raise FileNotFoundError(args.adapter / filename)
    selected = select_dev_tasks(args.manifest, per_template=None if args.full_dev else args.tasks_per_template)
    args.database_hashes = {task.task_id: sha256_file(task.database_path) for _, task in selected}
    protocol = {
        "split": "dev", "selection": "full_dev" if args.full_dev else "first_ids_per_template",
        "pilot": not args.full_dev, "tasks_per_template": None if args.full_dev else args.tasks_per_template,
        "tasks": len(selected), "task_ids": [task.task_id for _, task in selected],
        "trials_per_task": args.trials_per_task, "rollouts_per_stage": len(selected) * args.trials_per_task,
        "base_model": args.base_model, "adapter": str(args.adapter), "lora_r": config["r"],
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "source_database_sha256": args.database_hashes, "verifier_version": VERIFIER_VERSION,
        "user_reply_protocol_version": USER_REPLY_PROTOCOL_VERSION,
        "evaluator_source_sha256": {
            relative: sha256_file(ROOT / relative)
            for relative in (
                "src/ecommerce_customer_service/communication.py",
                "src/ecommerce_customer_service/communication_facts.py",
                "src/ecommerce_customer_service/verifier.py",
                "src/ecommerce_customer_service/simulator.py",
                "src/ecommerce_customer_service/local_hf.py",
                "scripts/evaluate_local_hf.py",
            )
        },
        "temperature": args.temperature, "sampling_top_p": 1.0,
        "sampling_top_k": 0 if args.temperature > 0 else 50,
        "seed": args.seed, "paired_seeds": True, "correction_retries": 0,
        "max_steps": args.max_steps, "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens, "prompt_truncation": False,
        "base_inference": "same_quantized_backbone_with_adapter_disabled",
        "json_repair": False, "cached_only": not args.allow_download,
    }
    print(json.dumps({"event": "preflight_ok", "protocol": protocol}, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return 0
    # Only allow a genuinely new output directory; even partial results are kept.
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    model, completer, model_info = load_backend(args)
    write_json(args.output / "model_info.json", model_info)
    started = time.monotonic()
    stage_rows = {}
    reports = {}
    for stage in ("base", "sft"):
        print(json.dumps({"event": "stage_start", "stage": stage}), flush=True)
        with model.disable_adapter() if stage == "base" else nullcontext():
            stage_rows[stage], reports[stage] = collect_stage(stage, selected, completer, args)
    summary = {
        "complete": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "model_info": model_info, "stages": reports,
        "paired_comparison": compare_rollouts(stage_rows["base"], stage_rows["sft"]),
        "elapsed_seconds": time.monotonic() - started,
        "interpretation": "Pilot is a harness check, not full-dev/test performance or an RL gate." if not args.full_dev
                          else "Dev-only checkpoint evidence; held-out test and RL gate remain separate.",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"event": "LOCAL_DEV_EVALUATION_COMPLETE", "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
