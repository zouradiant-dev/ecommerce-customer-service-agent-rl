from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.online_env import (  # noqa: E402
    build_environment_factories,
    build_online_rows,
    strict_environment_reward,
)
from ecommerce_customer_service.tasks import load_task_paths  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Online multi-turn GRPO with one isolated ecommerce environment per rollout"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True, help="SFT checkpoint or merged SFT model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev"], default="train")
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
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task_paths = load_task_paths(args.manifest, split=args.split)
    rows, eligibility = build_online_rows(task_paths)
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
        **eligibility,
        "notes": [
            "Each completion receives a fresh SQLite snapshot.",
            "Tool observations are generated inside the rollout loop.",
            "The verifier reads post-rollout state; test tasks are rejected.",
            "Use a merged SFT checkpoint as --model when starting from an adapter-trained policy.",
        ],
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if rows else 2
    if not rows:
        raise SystemExit("No inline-dialogue tasks are eligible for online GRPO")

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install requirements-online-grpo.txt in a separate Python 3.11+ CUDA environment"
        ) from exc

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
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=strict_environment_reward,
        environment_factory=factories,
        peft_config=peft_config,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    trainer.train()
    trainer.save_model(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
