from __future__ import annotations

from typing import Any


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def qwen_7b_profile() -> dict[str, Any]:
    """Return the single source of truth for the project's base model."""

    return {
        "profile_version": "ecommerce_cs_qwen_7b_v2",
        "model_id": DEFAULT_BASE_MODEL,
        "family": "Qwen2.5",
        "parameter_scale": "7B",
        "role": "base_policy_for_baseline_sft_and_grpo",
        "chat_mode": "tool_calling",
        "serving": {
            "protocol": "openai_compatible_chat_completions",
            "default_api_base": "http://127.0.0.1:8000/v1",
        },
        "sft": {
            "method": "qlora",
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
            "max_sequence_length": 4096,
            "assistant_only_loss": True,
            "lora": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "starting_hyperparameters": {
                "learning_rate": 0.0002,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "max_optimizer_steps": 200,
                "epochs": 2,
                "warmup_ratio": 0.03,
            },
        },
        "grpo": {
            "initial_policy": "sft_checkpoint",
            "reward": "strict_binary_business_verifier",
            "num_generations_per_task": 6,
            "temperature": 0.8,
            "max_agent_steps": 20,
            "starting_hyperparameters": {
                "learning_rate": 0.000005,
                "kl_beta": 0.04,
                "max_optimizer_steps_per_iteration": 20,
            },
        },
        "notes": [
            "All numeric training values are starting points and must be selected with dev-set experiments.",
            "The 8 GB local GPU is suitable for environment tests and quantized inference; full 7B post-training should use a larger-GPU machine.",
        ],
    }
