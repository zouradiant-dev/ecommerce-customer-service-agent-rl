"""Paired SFT/GRPO dev comparison using explicit v13; no test-set access or training."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.evaluation import compare_rollouts, evaluate_rollouts
from ecommerce_customer_service.experiment_prompt import EXPERIMENT_SYSTEM_PROMPT, PROMPT_VERSION
from ecommerce_customer_service.grpo_experiment_v13 import BASE_MODEL, BASE_REVISION, SCORING_FILES, source_hashes
from ecommerce_customer_service.local_hf import (
    LocalTransformersAgent, TransformersCompleter, paired_seed, select_dev_tasks, sha256_file, validate_lora_adapter,
)
from ecommerce_customer_service.simulator import USER_REPLY_PROTOCOL_VERSION, run_task
from ecommerce_customer_service.verifier import V13_VERIFIER_VERSION
from train_grpo_experiment_v13 import write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks-per-template", type=int, default=2)
    parser.add_argument("--full-dev", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def preflight(args):
    if args.tasks_per_template < 1:
        raise ValueError("positive_dev_selection_required")
    args.output, args.sft, args.manifest = args.output.resolve(), args.sft.resolve(), args.manifest.resolve()
    args.candidate = [path.resolve() for path in args.candidate]
    if args.output.exists() or any(args.output == path or path in args.output.parents for path in [args.sft, *args.candidate]):
        raise ValueError("new_evaluation_output_required")
    if len(set(args.candidate)) != len(args.candidate) or args.sft in args.candidate:
        raise ValueError("distinct_sft_and_candidate_adapters_required")
    for path in [args.sft, *args.candidate]:
        validate_lora_adapter(path, BASE_MODEL)
    selected = select_dev_tasks(args.manifest, per_template=None if args.full_dev else args.tasks_per_template)
    if any(task.split != "dev" for _, task in selected):
        raise ValueError("dev_only_comparison_required")
    hashes = {task.task_id: sha256_file(task.database_path) for _, task in selected}
    protocol = {"split": "dev", "tasks": len(selected), "full_dev": args.full_dev,
        "task_ids": [task.task_id for _, task in selected], "source_database_sha256": hashes,
        "manifest_sha256": sha256_file(args.manifest), "verifier_version": V13_VERIFIER_VERSION,
        "base_revision": BASE_REVISION, "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256(EXPERIMENT_SYSTEM_PROMPT.encode()).hexdigest(),
        "source_sha256": {**source_hashes(ROOT), "scripts/evaluate_grpo_v13.py": sha256_file(Path(__file__))},
        "tool_surface": "native_dynamic", "runtime_gap_from_static_training": True,
        "user_reply_protocol_version": USER_REPLY_PROTOCOL_VERSION,
        "temperature": 0.0, "trials_per_task": 1, "seed": 7, "max_new_tokens": 512,
        "max_steps": 20, "max_input_tokens": 8192, "correction_retries": 0, "json_repair": False,
        "adapters": {"sft": {"path": str(args.sft), "sha256": sha256_file(args.sft / "adapter_model.safetensors")},
            **{f"grpo_{index}": {"path": str(path), "sha256": sha256_file(path / "adapter_model.safetensors")}
               for index, path in enumerate(args.candidate, 1)}},
        "formal_rl_gate_passed": False, "held_out_test_run": False}
    return protocol, selected


def main():
    args = build_parser().parse_args()
    protocol, selected = preflight(args)
    print(json.dumps({"event": "V13_DEV_PREFLIGHT", "protocol": protocol}, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return 0
    os.environ.update(USE_TF="0", USE_TORCH="1", HF_HUB_OFFLINE="1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("single_bf16_gpu_required")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True, padding_side="left")
    saved = AutoTokenizer.from_pretrained(args.sft, local_files_only=True)
    if tokenizer.get_vocab() != saved.get_vocab() or tokenizer.chat_template != saved.chat_template:
        raise ValueError("sft_tokenizer_differs_from_frozen_base")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(base, args.sft, is_trainable=False, local_files_only=True)
    model.eval()
    generation = GenerationConfig(max_new_tokens=512, do_sample=False, temperature=1.0,
        top_p=1.0, top_k=50, num_beams=1, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    completer = TransformersCompleter(model, tokenizer, torch, generation, device="cuda:0", max_input_tokens=8192)
    reports, stage_rows = {}, {}
    started = time.monotonic()
    for stage, config in protocol["adapters"].items():
        if stage != "sft":
            model.load_adapter(config["path"], adapter_name="candidate", is_trainable=False, local_files_only=True)
        model.set_adapter("default" if stage == "sft" else "candidate")
        model.eval()
        directory = args.output / stage
        directory.mkdir()
        rows = []
        with (directory / "rollouts.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for _, task in selected:
                completer.seed(paired_seed(7, task.task_id, 0))
                row = run_task(task, LocalTransformersAgent(completer, max_steps=20,
                    system_prompt=EXPERIMENT_SYSTEM_PROMPT), verifier_version=V13_VERIFIER_VERSION).to_dict()
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                print(json.dumps({"event": "V13_DEV_TASK", "stage": stage, "task_id": task.task_id,
                    "reward": row["reward"], "failures": row["verifier"]["failures"]}, ensure_ascii=False), flush=True)
                if row["stopped_reason"] == "worker_error":
                    raise RuntimeError("evaluation_worker_error; partial originals retained")
                rows.append(row)
        for _, task in selected:
            if sha256_file(task.database_path) != protocol["source_database_sha256"][task.task_id]:
                raise RuntimeError("evaluation_mutated_source_database")
        stage_rows[stage] = rows
        reports[stage] = evaluate_rollouts(rows)
        write_json(directory / "report.json", reports[stage])
        if stage != "sft":
            model.set_adapter("default")
            model.delete_adapter("candidate")
            torch.cuda.empty_cache()
    for adapter in protocol["adapters"].values():
        if sha256_file(Path(adapter["path"]) / "adapter_model.safetensors") != adapter["sha256"]:
            raise RuntimeError("evaluation_adapter_changed")
    if source_hashes(ROOT) != {key: value for key, value in protocol["source_sha256"].items() if key != "scripts/evaluate_grpo_v13.py"}:
        raise RuntimeError("evaluation_sources_changed")
    summary = {"complete": True, "protocol": protocol, "stages": reports,
        "paired_comparisons": {stage: compare_rollouts(stage_rows["sft"], rows)
                               for stage, rows in stage_rows.items() if stage != "sft"},
        "elapsed_seconds": time.monotonic() - started, "source_databases_unchanged": True,
        "formal_rl_gate_passed": False, "held_out_test_run": False,
        "interpretation": "Dev checkpoint comparison only; no held-out claim or formal RL gate."}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"event": "V13_DEV_COMPLETE", "summary": summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
