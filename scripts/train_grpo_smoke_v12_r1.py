from __future__ import annotations

import argparse
from dataclasses import fields
from hashlib import sha256
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.online_env import (  # noqa: E402
    TOOL_OBSERVATION_VERSION,
    build_environment_factories,
    build_online_rows,
    strict_environment_reward,
)
from ecommerce_customer_service.tasks import load_task_paths  # noqa: E402
from ecommerce_customer_service.experiment_prompt import (
    EXPERIMENT_SYSTEM_PROMPT, ONLINE_PROMPT_VERSION, ONLINE_SYSTEM_PROMPT, PROMPT_VERSION,
)
from ecommerce_customer_service.grpo_readiness import validate_smoke_evidence
from ecommerce_customer_service.local_hf import sha256_file, validate_lora_adapter
from ecommerce_customer_service.verifier import VERIFIER_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Online multi-turn GRPO with one isolated ecommerce environment per rollout"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="cached base model matching the SFT adapter")
    parser.add_argument("--adapter", type=Path, required=True, help="SFT checkpoint-150 adapter; loaded trainable with a frozen SFT reference copy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--reward-mode", choices=["strict_binary", "audit_shaped"], default="strict_binary")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--max-tool-iterations", type=int, default=12)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--runtime-check", action="store_true", help="check installed APIs without loading model or writing outputs")
    parser.add_argument("--smoke-only", action="store_true", help="explicit opt-in to a maximum 20-step engineering run, not a formal RL gate")
    parser.add_argument("--readiness-dir", type=Path, help="complete train pilot with reviewed positive and negative rewards")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_args(args):
    if args.reward_mode != "strict_binary":
        raise ValueError("this readiness smoke uses strict_binary; shaped reward needs a separate reviewed experiment")
    if min(args.max_steps, args.batch_size, args.gradient_accumulation, args.max_completion_length,
           args.max_tool_iterations, args.save_steps) < 1 or args.num_generations < 2:
        raise ValueError("positive counts and at least two generations required")
    for name in ("learning_rate", "temperature", "epsilon"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"positive finite {name} required")
    if not math.isfinite(args.beta) or args.beta <= 0:
        raise ValueError("positive beta required to validate the frozen SFT reference in this smoke")
    if args.smoke_only and args.max_steps > 20:
        raise ValueError("engineering smoke is capped at 20 optimizer steps")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or args.use_vllm:
        raise ValueError("this checked smoke path is single-GPU native HF; no vLLM or distributed runtime yet")
    if (args.batch_size * args.gradient_accumulation) % args.num_generations:
        raise ValueError("generation batch must contain complete num_generations groups")


def check_runtime():
    if sys.version_info < (3, 11):
        raise RuntimeError("use a separate Python 3.11+ GRPO environment, keep the SFT environment intact")
    from packaging.version import Version
    from trl import GRPOConfig, GRPOTrainer
    required = {"environment_factory", "quantization_config", "peft_config"}
    if not required.issubset(inspect.signature(GRPOTrainer).parameters):
        raise RuntimeError("TRL lacks the checked stateful environment/QLoRA API")
    config_fields = {field.name for field in fields(GRPOConfig)}
    if not {"max_tool_calling_iterations", "mask_truncated_completions", "model_init_kwargs"}.issubset(config_fields):
        raise RuntimeError("incompatible GRPOConfig API")
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate", "jmespath")}
    if versions["trl"] != "1.14.1" or Version(versions["transformers"]) < Version("5.13.0"):
        raise RuntimeError("install requirements-online-grpo.txt in a separate environment; do not upgrade .venv-training")
    return versions


def validate_update_evidence(*, expected_steps, trainer_steps, optimizer_events,
                             gradient_checks, nonzero_gradient_steps, changed_tensors):
    if not (expected_steps == trainer_steps == optimizer_events == gradient_checks):
        raise RuntimeError('optimizer_or_gradient_event_count_mismatch')
    if nonzero_gradient_steps < 1:
        raise RuntimeError('no_nonzero_policy_gradient_observed')
    if changed_tensors < 1:
        raise RuntimeError('policy_adapter_did_not_change')
    return {'optimizer_steps': trainer_steps, 'optimizer_events': optimizer_events,
            'finite_gradient_checks': gradient_checks,
            'nonzero_gradient_steps': nonzero_gradient_steps,
            'changed_policy_tensors': changed_tensors,
            'actual_policy_update_verified': True}


def make_policy_update_guard(callback_base, torch, model, audit_path):
    """Inspect actual pre-optimizer gradients and exact initial/final LoRA tensors."""
    class PolicyUpdateGuard(callback_base):
        def __init__(self):
            self.initial = {name: param.detach().cpu().clone()
                            for name, param in model.named_parameters() if param.requires_grad}
            if not self.initial or any('.default.' not in name for name in self.initial):
                raise RuntimeError('unexpected_trainable_parameters_in_update_audit')
            self.gradient_checks = self.nonzero_gradient_steps = self.optimizer_events = 0
            audit_path.open('x', encoding='utf-8').close()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            current = {name: param for name, param in model.named_parameters() if param.requires_grad}
            if set(current) != set(self.initial):
                raise RuntimeError('trainable_parameters_changed_during_training')
            with_grad = nonzero = 0
            for name, param in current.items():
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if not torch.isfinite(grad).all().item():
                    raise RuntimeError('nonfinite_policy_gradient:' + name)
                with_grad += 1
                nonzero += int(torch.count_nonzero(grad).item() > 0)
            if not with_grad:
                raise RuntimeError('policy_gradients_missing_before_optimizer')
            self.gradient_checks += 1
            self.nonzero_gradient_steps += int(nonzero > 0)
            record = {'event': 'GRPO_POLICY_GRADIENT_CHECK',
                      'next_optimizer_step': state.global_step + 1,
                      'finite': True, 'tensors_with_grad': with_grad,
                      'tensors_with_nonzero_grad': nonzero}
            with audit_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.optimizer_events += 1

        def verify(self, expected_steps, trainer_steps):
            current = dict(model.named_parameters())
            changed = sum(not torch.equal(current[name].detach().cpu(), initial)
                          for name, initial in self.initial.items())
            return validate_update_evidence(expected_steps=expected_steps, trainer_steps=trainer_steps,
                optimizer_events=self.optimizer_events, gradient_checks=self.gradient_checks,
                nonzero_gradient_steps=self.nonzero_gradient_steps, changed_tensors=changed)
    return PolicyUpdateGuard()


class AuditedReward:
    """Persist every online outcome; stop after three entirely zero-variance batches."""
    def __init__(self, path: Path, group_size: int):
        self.path, self.group_size = path, group_size
        self.batch = 0
        self.no_signal_batches = 0
        self.path.open("x", encoding="utf-8").close()

    def __call__(self, completions, environments, **kwargs):
        rewards = strict_environment_reward(completions, environments, **kwargs)
        if len(environments) != len({id(env) for env in environments}):
            raise RuntimeError("rollout_environments_not_isolated")
        if len(rewards) % self.group_size:
            raise RuntimeError("partial_reward_group")
        mixed = 0
        for start in range(0, len(rewards), self.group_size):
            group = environments[start:start + self.group_size]
            if len({env._task.task_id for env in group}) != 1:
                raise RuntimeError("group_tasks_not_identical")
            mixed += len(set(rewards[start:start + self.group_size])) > 1
        self.batch += 1
        with self.path.open("a", encoding="utf-8") as handle:
            for completion, env, reward in zip(completions, environments, rewards):
                handle.write(json.dumps({"batch": self.batch, "task_id": env._task.task_id,
                    "split": env._task.split, "reward": reward, "verifier_version": VERIFIER_VERSION,
                    "verifier": env._last_verification.to_dict(),
                    "reward_components": env._last_breakdown.to_dict(),
                    "tool_executions": [item.to_dict() for item in env._tools.executions],
                    "completion": completion}, ensure_ascii=False, default=str) + "\n")
        self.no_signal_batches = 0 if mixed else self.no_signal_batches + 1
        print(json.dumps({"event": "ONLINE_REWARD_BATCH", "batch": self.batch,
                          "mixed_groups": mixed, "groups": len(rewards) // self.group_size,
                          "no_signal_batches": self.no_signal_batches}), flush=True)
        if self.no_signal_batches >= 3:
            raise RuntimeError("three_zero_variance_batches; stopped to avoid spending GPU time without relative reward signal")
        return rewards


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    args.manifest, args.output = args.manifest.resolve(), args.output.resolve()
    args.adapter = args.adapter.resolve()
    task_paths = load_task_paths(args.manifest, split=args.split)
    rows, eligibility = build_online_rows(task_paths, system_prompt=ONLINE_SYSTEM_PROMPT)
    eligible_ids = {row["task_id"] for row in rows}
    eligible_paths = [
        path
        for path in task_paths
        if json.loads(path.read_text(encoding="utf-8"))["task_id"] in eligible_ids
    ]
    preflight = {
        "method": "online_environment_grpo",
        "trainer": "trl.GRPOTrainer",
        "model": args.model,
        "split": args.split,
        "reward_mode": args.reward_mode,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "prompt_version": ONLINE_PROMPT_VERSION,
        "prompt_sha256": sha256(ONLINE_SYSTEM_PROMPT.encode()).hexdigest(),
        "verifier_version": VERIFIER_VERSION,
        "tool_observation_version": TOOL_OBSERVATION_VERSION,
        "load_in_4bit": args.load_in_4bit, "smoke_only": args.smoke_only,
        "formal_rl_gate_passed": False,
        "manifest_sha256": sha256_file(args.manifest),
        **eligibility,
        "notes": [
            "Each completion receives a fresh SQLite snapshot.",
            "Tool observations are generated inside the rollout loop.",
            "The verifier reads post-rollout state; test tasks are rejected.",
            "The existing SFT adapter is trainable; TRL 1.14.1 creates a frozen ref copy for KL.",
            "This static-tool runtime differs from the native dynamic-tool pilot; validate it during the bounded smoke.",
        ],
    }
    blockers = []
    try:
        adapter_config = validate_lora_adapter(args.adapter, args.model)
        if adapter_config.get("target_parameters"):
            raise ValueError("target_parameters_adapter_not_supported_by_this_reference_copy_path")
        preflight["adapter"] = str(args.adapter)
        preflight["adapter_sha256"] = sha256_file(args.adapter / "adapter_model.safetensors")
        if not args.readiness_dir:
            raise ValueError("reviewed_train_readiness_dir_required")
        preflight["smoke_evidence"] = validate_smoke_evidence(
            args.readiness_dir, adapter_sha256=preflight["adapter_sha256"],
            manifest_sha256=preflight["manifest_sha256"])
        # Freeze smoke tasks to the reviewed pilot, not unseen task-specific labels.
        pilot = json.loads((args.readiness_dir / "protocol.json").read_text(encoding="utf-8"))
        if (pilot.get("prompt_version") != PROMPT_VERSION or pilot.get("verifier_version") != VERIFIER_VERSION
                or pilot.get("prompt_sha256") != sha256(EXPERIMENT_SYSTEM_PROMPT.encode()).hexdigest()):
            raise ValueError("pilot_prompt_or_reward_version_mismatch")
        ids = set(pilot["task_ids"])
        rows = [row for row in rows if row["task_id"] in ids]
        if {row["task_id"] for row in rows} != ids:
            raise ValueError("reviewed_task_not_online_eligible")
        eligible_paths = [path for path in eligible_paths if json.loads(path.read_text(encoding="utf-8"))["task_id"] in ids]
        from ecommerce_customer_service.models import TaskSpec
        for path in eligible_paths:
            task = TaskSpec.from_file(path)
            if sha256_file(task.database_path) != pilot["source_database_sha256"].get(task.task_id):
                raise ValueError(f"pilot_database_changed:{task.task_id}")
        for relative in ("src/ecommerce_customer_service/communication.py", "src/ecommerce_customer_service/verifier.py",
                         "src/ecommerce_customer_service/communication_facts.py",
                         "src/ecommerce_customer_service/tools.py"):
            if sha256_file(PROJECT_ROOT / relative) != pilot["source_sha256"].get(relative):
                raise ValueError(f"pilot_reward_or_tools_changed:{relative}")
        preflight["smoke_task_ids"] = sorted(ids)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        blockers.append(str(exc))
    if not args.smoke_only:
        blockers.append("explicit_smoke_only_required; formal RL gate has not been passed")
    preflight["blocking_reasons"] = blockers
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.runtime_check:
        preflight["runtime"] = check_runtime()
        print(json.dumps({"event": "GRPO_RUNTIME_API_OK", "versions": preflight["runtime"],
                          "model_loaded": False, "backward_verified": False}, indent=2), flush=True)
        return 0 if rows and not blockers else 2
    if args.preflight_only:
        return 0 if rows and not blockers else 2
    if blockers:
        raise SystemExit("GRPO blocked before model loading: " + "; ".join(blockers))
    if not rows:
        raise SystemExit("No inline-dialogue tasks are eligible for online GRPO")

    try:
        versions = check_runtime()
        import torch
        from datasets import Dataset
        from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install requirements-online-grpo.txt in a separate Python 3.11+ CUDA environment"
        ) from exc

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("single visible CUDA GPU required for this smoke path")
    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported; explicitly use --no-bf16 if appropriate")
    # Reserve a NEW run only after all read-only evidence/runtime checks.
    args.output.mkdir(parents=True, exist_ok=False)
    preflight["runtime"] = versions
    preflight["gpu"] = torch.cuda.get_device_name(0)
    preflight["source_sha256"] = {relative: sha256_file(PROJECT_ROOT / relative) for relative in (
        Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(), "src/ecommerce_customer_service/online_env.py",
        "src/ecommerce_customer_service/experiment_prompt.py", "src/ecommerce_customer_service/grpo_readiness.py",
        "src/ecommerce_customer_service/tools.py", "src/ecommerce_customer_service/verifier.py",
        "src/ecommerce_customer_service/communication.py", "src/ecommerce_customer_service/communication_facts.py")}
    with (args.output / "preflight.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n")

    dataset = Dataset.from_list(rows)
    factories = build_environment_factories(eligible_paths, reward_mode=args.reward_mode)
    config = GRPOConfig(
        output_dir=str(args.output),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        beta=args.beta,
        epsilon=args.epsilon,
        loss_type="grpo",
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_iterations,
        save_steps=args.save_steps,
        logging_steps=1,
        seed=args.seed,
        bf16=args.bf16,
        use_vllm=args.use_vllm,
        remove_unused_columns=False,
        report_to="none",
        log_completions=True,
        top_p=1.0, top_k=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        mask_truncated_completions=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True, padding_side="left")
    base_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.get_vocab() != base_tokenizer.get_vocab() or tokenizer.chat_template != base_tokenizer.chat_template:
        raise ValueError("base_and_adapter_tokenizers_differ")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16) if args.load_in_4bit else None

    class FiniteLogGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in (logs or {}).values()):
                raise RuntimeError("nonfinite_training_metric; stopped")

    os.environ["HF_HUB_OFFLINE"] = "1"
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float16, device_map={"": 0},
        attn_implementation="sdpa", quantization_config=quantization)
    if args.load_in_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True, local_files_only=True)
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=AuditedReward(args.output / "online_rewards.jsonl", args.num_generations),
        environment_factory=factories,
        processing_class=tokenizer,
        callbacks=[FiniteLogGuard()],
    )
    # Do not silently fall back to a disabled-adapter/original-base reference.
    if "ref" not in trainer.model.peft_config:
        raise RuntimeError("frozen_sft_reference_adapter_missing")
    trainer.model.set_adapter("default")
    if not any(param.requires_grad for param in trainer.model.parameters()):
        raise RuntimeError("policy_has_no_trainable_parameters")
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
    update_guard = make_policy_update_guard(TrainerCallback, torch, trainer.model,
                                           args.output / 'gradient_audit.jsonl')
    trainer.add_callback(update_guard)
    with (args.output / "initialization.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"sft_adapter_sha256": preflight["adapter_sha256"],
            "reference": "frozen_copy_of_sft_adapter", "reference_sha256": initial_ref_hash,
            "reference_identical_to_initial_policy": True,
            "trainable_parameters": sum(param.numel() for param in trainer.model.parameters() if param.requires_grad)}, indent=2) + "\n")
    print(json.dumps({'event': 'GRPO_SMOKE_TRAIN_START', 'max_optimizer_steps': args.max_steps,
        'policy': 'trainable_sft_checkpoint_150', 'reference': 'frozen_sft_copy',
        'seven_b_model_loaded': True, 'gradient_and_update_audit_enabled': True}), flush=True)
    trainer.train()
    update_evidence = update_guard.verify(args.max_steps, trainer.state.global_step)
    if reference_digest() != initial_ref_hash:
        raise RuntimeError("sft_reference_changed_during_training")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    # Reload just the saved adapter on the same quantized backbone, avoiding
    # a second full model on a 24-GiB GPU. Exact tensors must survive save/load.
    trained = get_peft_model_state_dict(trainer.model, adapter_name="default")
    trainer.model.load_adapter(str(args.output), adapter_name="reload_probe", is_trainable=False)
    reloaded = get_peft_model_state_dict(trainer.model, adapter_name="reload_probe")
    if set(trained) != set(reloaded) or any(not torch.equal(trained[key], reloaded[key]) for key in trained):
        raise RuntimeError("saved_adapter_reload_mismatch")
    trainer.model.delete_adapter("reload_probe")
    with (args.output / "smoke_complete.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"complete": True, "optimizer_steps": trainer.state.global_step,
            "adapter_save_reload_verified": True, "formal_rl_gate_passed": False,
            "sft_reference_unchanged": True,
            **update_evidence,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "interpretation": "Engineering smoke only; no claim of held-out improvement or passed RL gate."}, indent=2) + "\n")
    print(json.dumps({'event': 'GRPO_SMOKE_COMPLETE', **update_evidence,
                      'adapter_save_reload_verified': True, 'sft_reference_unchanged': True,
                      'output': str(args.output), 'formal_rl_gate_passed': False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
