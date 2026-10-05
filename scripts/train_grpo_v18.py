"""Bounded v18 online GRPO: 10-step qualification then fresh 100-step run."""
from __future__ import annotations

import argparse
from dataclasses import fields
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_grpo_smoke_result import tensor_comparison  # noqa: E402
from ecommerce_customer_service.experiment_prompt_v18 import (  # noqa: E402
    V18_TRAINED_PROMPT_VERSION,
    V18_TRAINED_SYSTEM_PROMPT,
)
from ecommerce_customer_service.grpo_memory_v13 import (  # noqa: E402
    CudaMemoryAudit,
    MEMORY_POLICY,
    summarize_memory,
)
from ecommerce_customer_service.grpo_v18 import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    EXPECTED_SFT_ADAPTER_SHA256,
    TRAINING_RUNTIME_SOURCES,
    fingerprint,
    read_rows,
    select_v18_train_tasks,
    source_hashes,
    validate_calibration_evidence,
    validate_qualification,
)
from ecommerce_customer_service.local_hf import sha256_file, validate_lora_adapter  # noqa: E402
from ecommerce_customer_service.online_env import build_online_rows  # noqa: E402
from ecommerce_customer_service.online_env_v18 import (  # noqa: E402
    build_environment_factories_v18,
    environment_reward_v18,
)
from ecommerce_customer_service.tools_v18 import TOOLS_V18_VERSION  # noqa: E402
from ecommerce_customer_service.verifier_v18 import (  # noqa: E402
    TRAIN_REWARD_V18,
    V18_VERIFIER_VERSION,
)
from train_grpo_smoke_v12_r1 import check_runtime, make_policy_update_guard  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--calibration-rollouts", type=Path, required=True)
    parser.add_argument("--calibration-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("qualification", "experiment"), required=True)
    parser.add_argument("--qualification", type=Path, help="successful v18 10-step qualification; required for experiment")
    parser.add_argument("--experimental-only", action="store_true")
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--runtime-check", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.experimental_only:
        raise ValueError("explicit_experimental_only_required; formal RL gate remains false")
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 6:
        raise ValueError("max_hours_must_be_positive_and_at_most_six")
    if args.phase == "experiment" and not args.qualification:
        raise ValueError("v18_ten_step_qualification_required")
    if args.phase == "qualification" and args.qualification:
        raise ValueError("qualification_cannot_reuse_previous_training")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("single_gpu_native_hf_only")


def training_config() -> dict:
    # Keep the already-qualified 4090 generation geometry. Accumulation must not
    # silently turn into a larger KV-cache generation batch.
    return {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "generation_batch_size": 8,
        "num_generations": 4,
        "learning_rate": 5e-6,
        "temperature": 0.8,
        "beta": 0.04,
        "epsilon": 0.2,
        "loss_type": "grpo",
        "max_completion_length": 2048,
        "max_tool_calling_iterations": 12,
        "bf16": True,
        "use_vllm": False,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 1807,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "mask_truncated_completions": True,
        "remove_unused_columns": False,
        "report_to": "none",
        "logging_steps": 1,
        "log_completions": True,
        "save_total_limit": None,
    }


def preflight(args: argparse.Namespace) -> tuple[dict, list[dict], list[tuple[Path, object]]]:
    validate_args(args)
    for name in (
        "manifest",
        "adapter",
        "calibration_receipt",
        "calibration_rollouts",
        "calibration_protocol",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.qualification:
        args.qualification = args.qualification.resolve()
    protected = (
        args.adapter,
        args.calibration_receipt,
        args.calibration_rollouts,
        args.calibration_protocol,
    )
    if args.output.exists() or any(args.output == path or path in args.output.parents for path in protected):
        raise ValueError("new_non_nested_output_required")
    adapter_config = validate_lora_adapter(args.adapter, BASE_MODEL)
    if adapter_config.get("target_parameters"):
        raise ValueError("target_parameters_not_supported_by_frozen_sft_copy")
    adapter_hash = sha256_file(args.adapter / "adapter_model.safetensors")
    if adapter_hash != EXPECTED_SFT_ADAPTER_SHA256:
        raise ValueError("v18 GRPO must start from the frozen original SFT adapter")
    calibration = validate_calibration_evidence(
        root=ROOT,
        receipt_path=args.calibration_receipt,
        reviewed_rollouts=args.calibration_rollouts,
        protocol_path=args.calibration_protocol,
        manifest=args.manifest,
        adapter_sha256=adapter_hash,
    )
    selected, task_plan = select_v18_train_tasks(
        args.manifest,
        tasks_per_scenario=10,
    )
    if len(task_plan["scenarios"]) != 30 or len(selected) != 300:
        raise ValueError("v18 GRPO requires the frozen broad balanced 30x10 train selection")
    rows, eligibility = build_online_rows(
        [path for path, _ in selected],
        system_prompt=V18_TRAINED_SYSTEM_PROMPT,
    )
    if len(rows) != len(selected) or eligibility["excluded_scripted_dialogue_tasks"]:
        raise ValueError("selected v18 tasks are not all inline train tasks")
    base_context = {
        "schema_version": "ecommerce_cs_v18_grpo_preflight_v1",
        "method": "online_environment_grpo_v18",
        "trainer": "trl.GRPOTrainer",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter_sha256": adapter_hash,
        "manifest_sha256": task_plan["manifest_sha256"],
        "prompt_version": V18_TRAINED_PROMPT_VERSION,
        "prompt_sha256": sha256(V18_TRAINED_SYSTEM_PROMPT.encode()).hexdigest(),
        "verifier_version": V18_VERIFIER_VERSION,
        "tools_version": TOOLS_V18_VERSION,
        "reward_mode": TRAIN_REWARD_V18,
        "calibration": calibration,
        "task_plan": task_plan,
        "config": training_config(),
        "source_sha256": source_hashes(ROOT, TRAINING_RUNTIME_SOURCES),
        "memory_policy": MEMORY_POLICY,
        "fresh_sft_start_not_optimizer_resume": True,
        "formal_rl_gate_passed": False,
        "held_out_improvement_evaluated": False,
    }
    context = {**base_context, "context_fingerprint": fingerprint(base_context)}
    info = {
        **context,
        "phase": args.phase,
        "max_steps": 10 if args.phase == "qualification" else 100,
        "save_steps": 10 if args.phase == "qualification" else 25,
        "max_hours": args.max_hours,
    }
    if args.phase == "experiment":
        paths = {task.task_id: path for path, task in selected}
        info["qualification"] = validate_qualification(
            args.qualification,
            context=context,
            paths=paths,
            root=ROOT,
        )
        reference = tensor_comparison(
            args.adapter / "adapter_model.safetensors",
            args.qualification / "ref/adapter_model.safetensors",
        )
        policy = tensor_comparison(
            args.qualification / "ref/adapter_model.safetensors",
            args.qualification / "adapter_model.safetensors",
        )
        if not reference["comparison_after_dtype_alignment_equal"] or policy["changed_tensors"] < 1:
            raise ValueError("v18 qualification reference or policy tensor check failed")
    return info, rows, selected


class AuditedV18Reward:
    def __init__(self, path: Path, deadline: float, memory: CudaMemoryAudit | None = None) -> None:
        self.path = path
        self.deadline = deadline
        self.memory = memory
        self.batch = 0
        self.no_signal_batches = 0
        self.mixed_groups = 0
        self.total_groups = 0
        self.hard_safety_rollouts = 0
        self.path.open("x", encoding="utf-8").close()

    def __call__(self, completions, environments, **kwargs):
        if len(environments) != 8 or len({id(env) for env in environments}) != 8:
            raise RuntimeError("expected_eight_isolated_v18_rollouts")
        if any(
            env._reward_mode_v18 != TRAIN_REWARD_V18
            or env._task.split != "train"
            or not isinstance(env._task.metadata.get("terminal_v18"), dict)
            for env in environments
        ):
            raise RuntimeError("v18 online reward mode, split or metadata mismatch")
        rewards = environment_reward_v18(completions, environments, **kwargs)
        if len(rewards) != 8 or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not -1 <= value <= 1
            for value in rewards
        ):
            raise RuntimeError("invalid v18 shaped reward batch")
        mixed = 0
        for offset in (0, 4):
            group = environments[offset : offset + 4]
            if len({env._task.task_id for env in group}) != 1:
                raise RuntimeError("v18 relative reward group contains different tasks")
            mixed += len(set(rewards[offset : offset + 4])) > 1
        self.batch += 1
        self.mixed_groups += mixed
        self.total_groups += 2
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for completion, env, reward in zip(completions, environments, rewards):
                verification = env._last_v18_verification
                breakdown = env._last_v18_breakdown
                hard = bool(verification.hard_safety_failures)
                self.hard_safety_rollouts += int(hard)
                handle.write(
                    json.dumps(
                        {
                            "batch": self.batch,
                            "task_id": env._task.task_id,
                            "split": "train",
                            "scenario": env._task.scenario,
                            "reward": reward,
                            "reward_mode": TRAIN_REWARD_V18,
                            "verifier_version": V18_VERIFIER_VERSION,
                            "verification": verification.to_dict(),
                            "reward_components": breakdown.to_dict(),
                            "tool_executions": [call.to_dict() for call in env._tools.executions],
                            "completion": completion,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        self.no_signal_batches = 0 if mixed else self.no_signal_batches + 1
        print(
            json.dumps(
                {
                    "event": "V18_GRPO_REWARD_BATCH",
                    "batch": self.batch,
                    "mixed_groups": mixed,
                    "cumulative_mixed_group_rate": self.mixed_groups / self.total_groups,
                    "no_signal_batches": self.no_signal_batches,
                    "hard_safety_rollouts": self.hard_safety_rollouts,
                }
            ),
            flush=True,
        )
        if self.memory is not None:
            self.memory.sample("reward", step=self.batch)
        if self.no_signal_batches >= 3:
            raise RuntimeError("three_zero_variance_batches; stopped without automatic retry")
        if time.monotonic() >= self.deadline:
            raise RuntimeError("soft_wall_time_limit_reached_after_generation; partial checkpoints retained")
        return rewards


def run_training(
    args: argparse.Namespace,
    info: dict,
    rows: list[dict],
    selected: list[tuple[Path, object]],
) -> int:
    versions = check_runtime()
    import torch
    from datasets import Dataset
    from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    if "generation_batch_size" not in {field.name for field in fields(GRPOConfig)}:
        raise RuntimeError("explicit_generation_batch_size_API_missing")
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
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True, padding_side="left")
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=True,
    )
    if tokenizer.get_vocab() != base_tokenizer.get_vocab() or tokenizer.chat_template != base_tokenizer.chat_template:
        raise ValueError("base_and_adapter_tokenizers_differ")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(
        base,
        args.adapter,
        is_trainable=True,
        local_files_only=True,
    )

    class FiniteTimeGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(
                isinstance(value, (int, float)) and not math.isfinite(value)
                for value in (logs or {}).values()
            ):
                raise RuntimeError("nonfinite_v18_training_metric")

        def on_step_end(self, args, state, control, **kwargs):
            memory.sample("step_end", step=state.global_step)
            print(
                json.dumps(
                    {
                        "event": "V18_GRPO_PROGRESS",
                        "phase": info["phase"],
                        "completed_steps": state.global_step,
                        "total_steps": info["max_steps"],
                    }
                ),
                flush=True,
            )
            if time.monotonic() >= deadline:
                control.should_training_stop = True
                control.should_save = True

    config = GRPOConfig(
        output_dir=str(args.output),
        max_steps=info["max_steps"],
        save_steps=info["save_steps"],
        **training_config(),
    )
    factories = build_environment_factories_v18(
        [path for path, _ in selected],
        reward_mode=TRAIN_REWARD_V18,
    )
    reward = AuditedV18Reward(args.output / "online_rewards.jsonl", deadline, memory=memory)
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(rows),
        reward_funcs=reward,
        environment_factory=factories,
        processing_class=tokenizer,
        callbacks=[FiniteTimeGuard()],
    )
    if "ref" not in trainer.model.peft_config:
        raise RuntimeError("frozen_v18_sft_reference_missing")
    trainer.model.set_adapter("default")
    for name, parameter in trainer.model.named_parameters():
        if ".ref." in name:
            default = trainer.model.get_parameter(name.replace(".ref.", ".default."))
            parameter.data = parameter.data.to(default.dtype)
            parameter.requires_grad_(False)
            if not torch.equal(parameter, default):
                raise RuntimeError("reference_not_identical_to_initial_v18_sft")
        elif parameter.requires_grad and ".default." not in name:
            raise RuntimeError("unexpected_trainable_backbone_parameter")

    def reference_digest() -> str:
        digest = sha256()
        for name, parameter in sorted(trainer.model.named_parameters()):
            if ".ref." in name:
                digest.update(name.encode())
                digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()

    initial_ref_hash = reference_digest()
    update_guard = make_policy_update_guard(
        TrainerCallback,
        torch,
        trainer.model,
        args.output / "gradient_audit.jsonl",
    )
    trainer.add_callback(update_guard)
    write_json(
        args.output / "initialization.json",
        {
            "sft_adapter_sha256": info["adapter_sha256"],
            "reference_sha256": initial_ref_hash,
            "reference_identical_to_initial_policy": True,
            "base_revision": BASE_REVISION,
            "verifier_version": V18_VERIFIER_VERSION,
            "reward_mode": TRAIN_REWARD_V18,
        },
    )
    print(
        json.dumps(
            {
                "event": "V18_GRPO_START",
                "phase": info["phase"],
                "max_steps": info["max_steps"],
                "tasks": len(rows),
                "formal_rl_gate_passed": False,
            }
        ),
        flush=True,
    )
    memory.end()
    memory.begin("training")
    trainer.train()
    memory.end()
    if trainer.state.global_step != info["max_steps"]:
        trainer.save_model(str(args.output / "interrupted"))
        raise RuntimeError("bounded_v18_run_stopped_before_requested_steps")
    updates = update_guard.verify(info["max_steps"], trainer.state.global_step)
    mixed_group_rate = reward.mixed_groups / max(1, reward.total_groups)
    if mixed_group_rate < 0.20:
        raise RuntimeError("v18_online_mixed_group_rate_below_0.20")
    if reference_digest() != initial_ref_hash:
        raise RuntimeError("v18_sft_reference_changed")
    if (
        sha256_file(args.adapter / "adapter_model.safetensors") != info["adapter_sha256"]
        or source_hashes(ROOT, TRAINING_RUNTIME_SOURCES) != info["source_sha256"]
    ):
        raise RuntimeError("initial_v18_sft_or_training_source_changed")
    for _, task in selected:
        if sha256_file(task.database_path) != info["task_plan"]["database_sha256"][task.task_id]:
            raise RuntimeError("v18 source database changed:" + task.task_id)
    current_calibration = validate_calibration_evidence(
        root=ROOT,
        receipt_path=args.calibration_receipt,
        reviewed_rollouts=args.calibration_rollouts,
        protocol_path=args.calibration_protocol,
        manifest=args.manifest,
        adapter_sha256=info["adapter_sha256"],
    )
    if current_calibration != info["calibration"]:
        raise RuntimeError("v18 calibration evidence changed during training")
    memory.begin("saving")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    memory.end()
    memory.begin("reload")
    trained = get_peft_model_state_dict(trainer.model, adapter_name="default")
    trainer.model.load_adapter(str(args.output), adapter_name="reload_probe", is_trainable=False)
    reloaded = get_peft_model_state_dict(trainer.model, adapter_name="reload_probe")
    if set(trained) != set(reloaded) or any(not torch.equal(trained[key], reloaded[key]) for key in trained):
        raise RuntimeError("saved_v18_adapter_reload_mismatch")
    trainer.model.delete_adapter("reload_probe")
    del trained, reloaded
    memory.end()
    if reward.batch != info["max_steps"] or len(read_rows(args.output / "online_rewards.jsonl")) != 8 * info["max_steps"]:
        raise RuntimeError("v18 online reward batches do not match optimizer steps")
    measurement = summarize_memory(memory.rows, info["max_steps"])
    write_json(args.output / "memory_report.json", measurement)
    print(json.dumps({"event": "V18_GRPO_MEMORY_REPORT", **measurement}), flush=True)
    if not measurement["memory_safety_passed"]:
        raise RuntimeError("v18 stage memory headroom less than one GiB")
    artifact_names = (
        "preflight.json",
        "initialization.json",
        "online_rewards.jsonl",
        "gradient_audit.jsonl",
        "adapter_model.safetensors",
        "ref/adapter_model.safetensors",
        "memory_audit.jsonl",
        "memory_report.json",
    )
    result = {
        "complete": True,
        "phase": info["phase"],
        **updates,
        "verifier_version": V18_VERIFIER_VERSION,
        "reward_mode": TRAIN_REWARD_V18,
        "mixed_groups": reward.mixed_groups,
        "reward_groups": reward.total_groups,
        "mixed_group_rate": mixed_group_rate,
        "hard_safety_rollouts": reward.hard_safety_rollouts,
        "adapter_save_reload_verified": True,
        "sft_reference_unchanged": True,
        "source_databases_unchanged": True,
        "memory_policy": MEMORY_POLICY,
        "peak_allocated_gib": measurement["peak_allocated_gib"],
        "peak_reserved_gib": measurement["peak_reserved_gib"],
        "memory_headroom_gib": measurement["memory_headroom_gib"],
        "elapsed_seconds": time.monotonic() - started,
        "online_reward_batches": reward.batch,
        "artifacts_sha256": {
            name: sha256_file(args.output / name) for name in artifact_names
        },
        "formal_rl_gate_passed": False,
        "held_out_improvement_evaluated": False,
        "interpretation": "Bounded train-only v18 GRPO; held-out comparison is still required.",
    }
    write_json(args.output / "complete.json", result)
    print(json.dumps({"event": "V18_GRPO_COMPLETE", **result}, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    info, rows, selected = preflight(args)
    print(json.dumps({"event": "V18_GRPO_PREFLIGHT_OK", "protocol": info}, ensure_ascii=False, indent=2), flush=True)
    if args.runtime_check:
        versions = check_runtime()
        from trl import GRPOConfig

        if "generation_batch_size" not in {field.name for field in fields(GRPOConfig)}:
            raise RuntimeError("explicit_generation_batch_size_API_missing")
        print(json.dumps({"event": "V18_GRPO_RUNTIME_OK", "versions": versions, "model_loaded": False}), flush=True)
    if args.preflight_only or args.runtime_check:
        return 0
    try:
        return run_training(args, info, rows, selected)
    except Exception as exc:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            write_json(
                args.output / "failure.json",
                {
                    "complete": False,
                    "phase": args.phase,
                    "error": type(exc).__name__ + ": " + str(exc),
                    "automatic_retry": False,
                    "partial_artifacts_preserved": True,
                    "formal_rl_gate_passed": False,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
