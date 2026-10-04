"""Explicit, bounded v13 GRPO qualification (10) or exploratory experiment (100).

Neither phase is a formal RL gate. Both start from the immutable SFT adapter;
the qualification's updated adapter is NOT used as the experiment's reference.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.experiment_prompt import ONLINE_PROMPT_VERSION, ONLINE_SYSTEM_PROMPT
from ecommerce_customer_service.grpo_experiment_v13 import (
    BASE_MODEL, BASE_REVISION, fingerprint, read_rows, select_experiment_tasks,
    source_hashes, validate_evidence, validate_qualification,
)
from ecommerce_customer_service.local_hf import sha256_file, validate_lora_adapter
from ecommerce_customer_service.online_env import build_environment_factories, build_online_rows, strict_environment_reward
from ecommerce_customer_service.verifier import V13_VERIFIER_VERSION
from ecommerce_customer_service.grpo_memory_v13 import CudaMemoryAudit, MEMORY_POLICY, summarize_memory
from train_grpo_smoke_v12_r1 import check_runtime, make_policy_update_guard
from audit_grpo_smoke_result import tensor_comparison


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True, help="separate human-confirmed review; never overwrites original audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["qualification", "experiment"], required=True)
    parser.add_argument("--qualification", type=Path, help="successful, source-matched v13 10-step output; required for 100 steps")
    parser.add_argument("--experimental-only", action="store_true", help="acknowledge this is bounded exploration, not a passed formal RL gate")
    parser.add_argument("--max-hours", type=float, default=3.0, help="soft wall-time cap for this phase, checked at batch/optimizer boundaries")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--runtime-check", action="store_true")
    return parser


def validate_args(args):
    if not args.experimental_only:
        raise ValueError("explicit_experimental_only_required; formal RL gate remains false")
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 6:
        raise ValueError("max_hours_must_be_positive_and_at_most_six")
    if args.phase == "experiment" and not args.qualification:
        raise ValueError("v13_ten_step_qualification_required")
    if args.phase == "qualification" and args.qualification:
        raise ValueError("qualification_cannot_reuse_previous_training")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("single_gpu_native_hf_only")


def training_config():
    # Pin generation batch separately. Do NOT turn accumulation into an
    # unmeasured larger generation/KV batch in TRL 1.14.1.
    return dict(per_device_train_batch_size=1, gradient_accumulation_steps=8,
                generation_batch_size=8, num_generations=4, learning_rate=5e-6,
                temperature=0.8, beta=0.04, epsilon=0.2, loss_type="grpo",
                max_completion_length=2048, max_tool_calling_iterations=12,
                bf16=True, use_vllm=False, top_p=1.0, top_k=0, seed=7,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                mask_truncated_completions=True, remove_unused_columns=False,
                report_to="none", logging_steps=1, log_completions=True,
                save_total_limit=None)


def preflight(args):
    validate_args(args)
    for name in ("manifest", "adapter", "evidence", "review", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() or any(args.output == path or path in args.output.parents for path in (args.adapter, args.evidence)):
        raise ValueError("new_non_nested_output_required")
    adapter_config = validate_lora_adapter(args.adapter, BASE_MODEL)
    if adapter_config.get("target_parameters"):
        raise ValueError("target_parameters_not_supported_by_frozen_sft_copy")
    adapter_hash = sha256_file(args.adapter / "adapter_model.safetensors")
    evidence, tasks = validate_evidence(args.evidence, args.review, root=ROOT,
                                      manifest=args.manifest, adapter_sha256=adapter_hash)
    context = {"verifier_version": V13_VERIFIER_VERSION, "base_revision": BASE_REVISION,
               "adapter_sha256": adapter_hash, "manifest_sha256": sha256_file(args.manifest),
               "evidence": evidence, "config": training_config(), "source_sha256": source_hashes(ROOT),
               "experiment_task_files_sha256": {task.task_id: sha256_file(path) for path, task in tasks},
               "experiment_database_sha256": {task.task_id: sha256_file(task.database_path) for _, task in tasks},
               "memory_policy": MEMORY_POLICY, "online_prompt_version": ONLINE_PROMPT_VERSION,
               "online_prompt_sha256": sha256(ONLINE_SYSTEM_PROMPT.encode()).hexdigest()}
    selected = select_experiment_tasks(tasks, evidence["evidence_task_ids"], args.phase)
    rows, eligibility = build_online_rows([path for path, _ in selected], system_prompt=ONLINE_SYSTEM_PROMPT)
    if len(rows) != len(selected) or eligibility["excluded_scripted_dialogue_tasks"]:
        raise ValueError("selected_tasks_not_all_inline_train")
    info = {**context, "phase": args.phase, "max_steps": 10 if args.phase == "qualification" else 100,
            "save_steps": 10 if args.phase == "qualification" else 25,
            "max_hours": args.max_hours, "task_ids": [task.task_id for _, task in selected],
            "tasks": len(rows), "scope": "reviewed_train_template_families_only",
            "formal_rl_gate_passed": False, "held_out_improvement_evaluated": False,
            "fresh_sft_start_not_optimizer_resume": True,
            "context_fingerprint": fingerprint(context)}
    if args.phase == "experiment":
        qualification = args.qualification.resolve()
        info["qualification"] = validate_qualification(qualification, context=context,
            paths={task.task_id: path for path, task in tasks}, root=ROOT)
        ref = tensor_comparison(args.adapter / "adapter_model.safetensors", qualification / "ref/adapter_model.safetensors")
        policy = tensor_comparison(qualification / "ref/adapter_model.safetensors", qualification / "adapter_model.safetensors")
        if not ref["comparison_after_dtype_alignment_equal"] or policy["changed_tensors"] < 1:
            raise ValueError("qualification_reference_or_policy_tensor_check_failed")
        info["qualification"]["saved_reference_matches_initial_sft"] = True
        info["qualification"]["saved_policy_changed_tensors"] = policy["changed_tensors"]
    return info, rows, selected


class AuditedV13Reward:
    def __init__(self, path, deadline, memory=None):
        self.path, self.deadline = path, deadline
        self.memory = memory
        self.batch = self.no_signal_batches = 0
        self.path.open("x", encoding="utf-8").close()

    def __call__(self, completions, environments, **kwargs):
        if len(environments) != 8 or len({id(env) for env in environments}) != 8:
            raise RuntimeError("expected_eight_isolated_rollouts")
        if any(env._verifier_version != V13_VERIFIER_VERSION or env._task.split != "train" for env in environments):
            raise RuntimeError("online_reward_version_or_split_mismatch")
        rewards = strict_environment_reward(completions, environments, **kwargs)
        if len(rewards) != 8 or any(value not in (0.0, 1.0) for value in rewards):
            raise RuntimeError("invalid_strict_binary_reward_batch")
        mixed = 0
        for offset in (0, 4):
            if len({env._task.task_id for env in environments[offset:offset + 4]}) != 1:
                raise RuntimeError("relative_reward_group_contains_different_tasks")
            mixed += len(set(rewards[offset:offset + 4])) > 1
        self.batch += 1
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for completion, env, reward in zip(completions, environments, rewards):
                handle.write(json.dumps({"batch": self.batch, "task_id": env._task.task_id,
                    "split": "train", "reward": reward, "verifier_version": env._verifier_version,
                    "verifier": env._last_verification.to_dict(), "reward_components": env._last_breakdown.to_dict(),
                    "tool_executions": [call.to_dict() for call in env._tools.executions],
                    "completion": completion}, ensure_ascii=False, allow_nan=False) + "\n")
        self.no_signal_batches = 0 if mixed else self.no_signal_batches + 1
        print(json.dumps({"event": "V13_REWARD_BATCH", "batch": self.batch, "mixed_groups": mixed,
                          "no_signal_batches": self.no_signal_batches}), flush=True)
        if self.memory is not None:
            self.memory.sample("reward", step=self.batch)
        if self.no_signal_batches >= 3:
            raise RuntimeError("three_zero_variance_batches; stopped without automatic retry")
        if time.monotonic() >= self.deadline:
            raise RuntimeError("soft_wall_time_limit_reached_after_generation; partial checkpoints retained")
        return rewards


def run_training(args, info, rows, selected):
    versions = check_runtime()
    import torch
    from datasets import Dataset
    from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("single_visible_bf16_cuda_gpu_required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    memory = CudaMemoryAudit(torch.cuda, args.output / "memory_audit.jsonl")
    memory.begin("initialization")
    info["runtime"] = versions
    info["gpu"] = torch.cuda.get_device_name(0)
    write_json(args.output / "preflight.json", info)
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True, padding_side="left")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True)
    if tokenizer.get_vocab() != base_tokenizer.get_vocab() or tokenizer.chat_template != base_tokenizer.chat_template:
        raise ValueError("base_and_adapter_tokenizers_differ")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True, local_files_only=True)

    class FiniteTimeGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in (logs or {}).values()):
                raise RuntimeError("nonfinite_training_metric")

        def on_step_end(self, args, state, control, **kwargs):
            memory.sample("step_end", step=state.global_step)
            print(json.dumps({"event": "V13_PROGRESS", "phase": info["phase"],
                              "completed_steps": state.global_step, "total_steps": info["max_steps"]}), flush=True)
            if time.monotonic() >= deadline:
                control.should_training_stop = True
                control.should_save = True

    config = GRPOConfig(output_dir=str(args.output), max_steps=info["max_steps"],
                        save_steps=info["save_steps"], **training_config())
    factories = build_environment_factories([path for path, _ in selected], verifier_version=V13_VERIFIER_VERSION)
    reward = AuditedV13Reward(args.output / "online_rewards.jsonl", deadline, memory=memory)
    trainer = GRPOTrainer(model=model, args=config, train_dataset=Dataset.from_list(rows),
        reward_funcs=reward, environment_factory=factories, processing_class=tokenizer, callbacks=[FiniteTimeGuard()])
    if "ref" not in trainer.model.peft_config:
        raise RuntimeError("frozen_sft_reference_missing")
    trainer.model.set_adapter("default")
    for name, param in trainer.model.named_parameters():
        if ".ref." in name:
            default = trainer.model.get_parameter(name.replace(".ref.", ".default."))
            param.data = param.data.to(default.dtype)
            param.requires_grad_(False)
            if not torch.equal(param, default):
                raise RuntimeError("reference_not_identical_to_initial_sft")
        elif param.requires_grad and ".default." not in name:
            raise RuntimeError("unexpected_trainable_backbone_parameter")

    def reference_digest():
        digest = sha256()
        for name, param in sorted(trainer.model.named_parameters()):
            if ".ref." in name:
                digest.update(name.encode())
                digest.update(param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()

    initial_ref_hash = reference_digest()
    update_guard = make_policy_update_guard(TrainerCallback, torch, trainer.model, args.output / "gradient_audit.jsonl")
    trainer.add_callback(update_guard)
    write_json(args.output / "initialization.json", {"sft_adapter_sha256": info["adapter_sha256"],
        "reference_sha256": initial_ref_hash, "reference_identical_to_initial_policy": True,
        "base_revision": BASE_REVISION, "verifier_version": V13_VERIFIER_VERSION})
    print(json.dumps({"event": "V13_GRPO_START", "phase": args.phase,
                      "max_steps": info["max_steps"], "tasks": len(rows), "formal_rl_gate_passed": False}), flush=True)
    memory.end()
    memory.begin("training")
    trainer.train()
    memory.end()
    if trainer.state.global_step != info["max_steps"]:
        trainer.save_model(str(args.output / "interrupted"))
        raise RuntimeError("bounded_run_stopped_before_requested_steps; not a successful qualification")
    updates = update_guard.verify(info["max_steps"], trainer.state.global_step)
    if reference_digest() != initial_ref_hash:
        raise RuntimeError("sft_reference_changed")
    if (sha256_file(args.adapter / "adapter_model.safetensors") != info["adapter_sha256"]
            or source_hashes(ROOT) != info["source_sha256"]):
        raise RuntimeError("initial_sft_or_training_source_changed")
    for _, task in selected:
        if sha256_file(task.database_path) != info["experiment_database_sha256"][task.task_id]:
            raise RuntimeError("source_database_changed:" + task.task_id)
    memory.begin("saving")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    memory.end()
    memory.begin("reload")
    trained = get_peft_model_state_dict(trainer.model, adapter_name="default")
    trainer.model.load_adapter(str(args.output), adapter_name="reload_probe", is_trainable=False)
    reloaded = get_peft_model_state_dict(trainer.model, adapter_name="reload_probe")
    if set(trained) != set(reloaded) or any(not torch.equal(trained[key], reloaded[key]) for key in trained):
        raise RuntimeError("saved_adapter_reload_mismatch")
    trainer.model.delete_adapter("reload_probe")
    del trained, reloaded
    memory.end()
    if reward.batch != info["max_steps"] or len(read_rows(args.output / "online_rewards.jsonl")) != 8 * info["max_steps"]:
        raise RuntimeError("online_reward_batches_do_not_match_optimizer_steps")
    measurement = summarize_memory(memory.rows, info["max_steps"])
    write_json(args.output / "memory_report.json", measurement)
    print(json.dumps({"event": "V13_MEMORY_REPORT", **measurement}), flush=True)
    if not measurement["memory_safety_passed"]:
        raise RuntimeError("stage_memory_headroom_less_than_one_gib; see memory_report.json; do not start larger run")
    result = {"complete": True, "phase": args.phase, **updates,
        "verifier_version": V13_VERIFIER_VERSION, "adapter_save_reload_verified": True,
        "sft_reference_unchanged": True, "source_databases_unchanged": True,
        "memory_policy": MEMORY_POLICY, "peak_allocated_gib": measurement["peak_allocated_gib"],
        "peak_reserved_gib": measurement["peak_reserved_gib"], "memory_headroom_gib": measurement["memory_headroom_gib"],
        "elapsed_seconds": time.monotonic() - started, "online_reward_batches": reward.batch,
        "artifacts_sha256": {name: sha256_file(args.output / name) for name in (
            "preflight.json", "initialization.json", "online_rewards.jsonl", "gradient_audit.jsonl",
            "adapter_model.safetensors", "ref/adapter_model.safetensors", "memory_audit.jsonl", "memory_report.json")},
        "formal_rl_gate_passed": False, "held_out_improvement_evaluated": False,
        "interpretation": "Bounded train-only exploration; not calibrated held-out performance or a formal RL gate."}
    write_json(args.output / "complete.json", result)
    print(json.dumps({"event": "V13_GRPO_COMPLETE", **result}, ensure_ascii=False), flush=True)
    return 0


def main():
    args = build_parser().parse_args()
    info, rows, selected = preflight(args)
    print(json.dumps({"event": "V13_PREFLIGHT_OK", "protocol": info}, ensure_ascii=False, indent=2), flush=True)
    if args.runtime_check:
        versions = check_runtime()
        from trl import GRPOConfig
        from dataclasses import fields
        if "generation_batch_size" not in {field.name for field in fields(GRPOConfig)}:
            raise RuntimeError("explicit_generation_batch_size_API_missing")
        print(json.dumps({"event": "V13_RUNTIME_API_OK", "versions": versions, "model_loaded": False}), flush=True)
    if args.preflight_only or args.runtime_check:
        return 0
    try:
        return run_training(args, info, rows, selected)
    except Exception as exc:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            write_json(args.output / "failure.json", {"complete": False, "phase": args.phase,
                "error": type(exc).__name__ + ": " + str(exc), "automatic_retry": False,
                "memory_policy": MEMORY_POLICY,
                "memory_audit_sha256": sha256_file(args.output / "memory_audit.jsonl") if (args.output / "memory_audit.jsonl").is_file() else None,
                "memory_report_sha256": sha256_file(args.output / "memory_report.json") if (args.output / "memory_report.json").is_file() else None,
                "formal_rl_gate_passed": False, "partial_artifacts_preserved": True})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
