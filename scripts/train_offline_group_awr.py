from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.data import load_jsonl  # noqa: E402
from ecommerce_customer_service.tools import CustomerServiceTools  # noqa: E402
from ecommerce_customer_service.training import (  # noqa: E402
    compute_group_advantages,
    render_with_assistant_mask,
    save_base_model_configs,
    validate_training_row,
)


DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline group-normalized advantage-weighted regression over verified rollouts. "
            "This is an offline baseline, not online GRPO."
        )
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-in", type=Path, required=True, help="SFT adapter or preceding AWR adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, help="AWR checkpoint containing training_state.pt")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--denominator-epsilon", type=float, default=0.25)
    parser.add_argument("--advantage-clip", type=float, default=3.0)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--overlength", choices=["skip", "right-truncate"], default="skip")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_jsonl(args.rollouts)
    selected, advantage_summary = compute_group_advantages(
        rows,
        denominator_epsilon=args.denominator_epsilon,
        advantage_clip=args.advantage_clip,
        require_mixed_outcomes=True,
    )
    structural_failures = [
        f"{row.get('task_id')}:{failure}"
        for row in selected
        for failure in validate_training_row(
            {
                "messages": row.get("messages") or [],
                "loss_mask": [message.get("role") == "assistant" for message in row.get("messages") or []],
            }
        )
    ]
    if structural_failures:
        raise ValueError("Invalid rollout messages: " + "; ".join(structural_failures[:10]))
    preflight = {**advantage_summary, "structural_failures": structural_failures}
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if selected else 2
    if not selected:
        raise SystemExit("No mixed-outcome task groups; GRPO would have zero useful advantage")

    try:
        import torch
        import torch.nn.functional as functional
        from peft import PeftModel, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit("Install requirements-training.txt before GRPO training") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the configured 7B QLoRA run")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tools = CustomerServiceTools.schemas()

    examples: list[dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(selected):
        input_ids, assistant_mask = render_with_assistant_mask(
            tokenizer,
            row.get("messages") or [],
            tools=tools,
        )
        if len(input_ids) > args.max_sequence_length:
            if args.overlength == "skip":
                skipped += 1
                continue
            input_ids = input_ids[: args.max_sequence_length]
            assistant_mask = assistant_mask[: args.max_sequence_length]
        if not any(assistant_mask[1:]):
            skipped += 1
            continue
        examples.append(
            {
                "id": f"{row.get('task_id')}:{row.get('trial', index)}:{index}",
                "input_ids": input_ids,
                "assistant_mask": assistant_mask,
                "advantage": float(row["advantage"]),
            }
        )
    if not examples:
        raise SystemExit("No renderable offline group-AWR rollouts remain")
    random.Random(args.seed).shuffle(examples)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": args.device},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    adapter_source = args.resume_from or args.adapter_in
    model = PeftModel.from_pretrained(model, adapter_source, is_trainable=True)
    model.print_trainable_parameters()

    args.output.mkdir(parents=True, exist_ok=True)
    reference_cache_path = args.output / "reference_nll.pt"
    reference_cache: dict[str, list[float]] = {}
    if args.kl_beta > 0 and reference_cache_path.exists():
        reference_cache = torch.load(reference_cache_path, map_location="cpu")
    if args.kl_beta > 0 and not reference_cache:
        model.eval()
        with torch.no_grad():
            for number, example in enumerate(examples, 1):
                tensor = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
                logits = model(input_ids=tensor).logits[:, :-1, :]
                target = tensor[:, 1:]
                nll = functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    reduction="none",
                )
                reference_cache[example["id"]] = nll.float().cpu().tolist()
                print(f"reference {number}/{len(examples)}", flush=True)
        torch.save(reference_cache, reference_cache_path)
        model.train()

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        length = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), length), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(batch), length), dtype=torch.long)
        mask = torch.zeros((len(batch), length), dtype=torch.float32)
        reference = torch.zeros((len(batch), max(0, length - 1)), dtype=torch.float32)
        for index, item in enumerate(batch):
            size = len(item["input_ids"])
            input_ids[index, :size] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention[index, :size] = 1
            mask[index, :size] = torch.tensor(item["assistant_mask"], dtype=torch.float32)
            if args.kl_beta > 0:
                values = reference_cache[item["id"]]
                reference[index, : len(values)] = torch.tensor(values, dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "assistant_mask": mask,
            "advantage": torch.tensor([item["advantage"] for item in batch], dtype=torch.float32),
            "reference_nll": reference,
        }

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    optimizer_step = 0
    micro_batches_seen = 0
    resume_micro_batches = 0
    if args.resume_from and (args.resume_from / "training_state.pt").exists():
        state = torch.load(args.resume_from / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        optimizer_step = int(state.get("optimizer_step") or 0)
        resume_micro_batches = int(state.get("micro_batches_seen") or 0)

    def save_checkpoint(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        save_base_model_configs(model, path)
        tokenizer.save_pretrained(path)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "optimizer_step": optimizer_step,
                "micro_batches_seen": micro_batches_seen,
                "arguments": vars(args),
                "advantage_summary": advantage_summary,
            },
            path / "training_state.pt",
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    trace: list[dict[str, Any]] = []
    started = time.time()
    while optimizer_step < args.max_steps:
        for batch in loader:
            micro_batches_seen += 1
            if micro_batches_seen <= resume_micro_batches:
                continue
            input_ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            output = model(input_ids=input_ids, attention_mask=attention)
            logits = output.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            token_nll = functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(input_ids.size(0), -1)
            assistant_mask = batch["assistant_mask"][:, 1:].to(device)
            token_count = assistant_mask.sum(dim=1).clamp(min=1.0)
            rollout_nll = (token_nll * assistant_mask).sum(dim=1) / token_count
            policy_loss = (batch["advantage"].to(device) * rollout_nll).mean()
            kl_loss = torch.zeros((), device=device)
            if args.kl_beta > 0:
                reference_nll = batch["reference_nll"].to(device)
                log_reference_over_policy = (token_nll - reference_nll).clamp(-10.0, 10.0)
                token_kl = torch.exp(log_reference_over_policy) - log_reference_over_policy - 1.0
                kl_loss = (token_kl * assistant_mask).sum(dim=1).div(token_count).mean()
            loss = (policy_loss + args.kl_beta * kl_loss) / args.gradient_accumulation
            loss.backward()
            accumulation += 1
            if accumulation < args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0
            optimizer_step += 1
            row = {
                "step": optimizer_step,
                "policy_loss": float(policy_loss.detach().cpu()),
                "kl_loss": float(kl_loss.detach().cpu()),
            }
            trace.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if args.save_every and optimizer_step % args.save_every == 0:
                save_checkpoint(args.output / f"checkpoint-{optimizer_step}")
            if optimizer_step >= args.max_steps:
                break

    save_checkpoint(args.output)
    metrics = {
        "stage": "offline_group_advantage_weighted_regression",
        "base_model": args.base_model,
        "adapter_in": str(args.adapter_in),
        "rollouts": len(rows),
        "usable_rollouts": len(examples),
        "skipped_rollouts": skipped,
        "advantage": advantage_summary,
        "optimizer_steps": optimizer_step,
        "kl_beta": args.kl_beta,
        "elapsed_seconds": time.time() - started,
        "max_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "trace": trace,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
