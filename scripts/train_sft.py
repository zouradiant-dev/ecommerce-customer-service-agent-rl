from __future__ import annotations

import argparse
import json
import math
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
from ecommerce_customer_service.training import (  # noqa: E402
    render_with_assistant_mask,
    save_base_model_configs,
    validate_training_row,
)


DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA SFT with assistant-only loss for ecommerce agent traces")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, help="adapter checkpoint containing training_state.pt")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--overlength", choices=["skip", "right-truncate"], default="skip")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--max-dev-examples",
        type=int,
        help="deterministically sample at most this many dev rows for smoke runs",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=DEFAULT_TARGETS)
    parser.add_argument("--validate-data-only", action="store_true")
    return parser


def validate_rows(rows: list[dict[str, Any]], label: str) -> None:
    failures = []
    for row in rows:
        for failure in validate_training_row(row):
            failures.append(f"{row.get('id')}:{failure}")
    if failures:
        raise ValueError(f"{label} contains invalid rows: " + "; ".join(failures[:10]))


def encode_rows(rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    encoded: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        input_ids, assistant_mask = render_with_assistant_mask(
            tokenizer,
            row.get("messages") or [],
            tools=list(row.get("tools") or []),
            loss_mask=[bool(value) for value in row.get("loss_mask") or []] or None,
        )
        if len(input_ids) > args.max_sequence_length:
            if args.overlength == "skip":
                skipped += 1
                continue
            input_ids = input_ids[: args.max_sequence_length]
            assistant_mask = assistant_mask[: args.max_sequence_length]
        labels = [token if target else -100 for token, target in zip(input_ids, assistant_mask)]
        if all(label == -100 for label in labels):
            skipped += 1
            continue
        encoded.append(
            {
                "id": str(row.get("id") or row.get("task_id") or len(encoded)),
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    if not encoded:
        raise RuntimeError("No SFT rows remain after rendering and length filtering")
    return encoded, skipped


def main() -> int:
    args = build_parser().parse_args()
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev) if args.dev else []
    dev_source_rows = len(dev_rows)
    validate_rows(train_rows, "train")
    validate_rows(dev_rows, "dev")
    if any((row.get("metadata") or {}).get("split") == "test" for row in train_rows + dev_rows):
        raise ValueError("test rows must not enter SFT")
    if args.validate_data_only:
        print(json.dumps({"train_rows": len(train_rows), "dev_rows": len(dev_rows), "valid": True}, indent=2))
        return 0
    if args.max_dev_examples is not None:
        if args.max_dev_examples < 1:
            raise ValueError("max-dev-examples must be positive")
        if len(dev_rows) > args.max_dev_examples:
            sampler = random.Random(args.seed + 1)
            dev_rows = sampler.sample(dev_rows, args.max_dev_examples)

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Install requirements-training.txt before SFT training") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the configured 7B QLoRA run")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples, train_skipped = encode_rows(train_rows, tokenizer, args)
    dev_examples, dev_skipped = encode_rows(dev_rows, tokenizer, args) if dev_rows else ([], 0)
    random.Random(args.seed).shuffle(train_examples)

    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        length = max(len(example["input_ids"]) for example in examples)
        input_ids, attention_mask, labels = [], [], []
        for example in examples:
            padding = length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [tokenizer.pad_token_id] * padding)
            attention_mask.append(example["attention_mask"] + [0] * padding)
            labels.append(example["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    dev_loader = DataLoader(dev_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if dev_examples else None

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": compute_dtype,
        "attn_implementation": "sdpa",
    }
    if args.load_in_4bit:
        model_kwargs.update(
            {
                "device_map": {"": args.device},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[value.strip() for value in args.lora_target_modules.split(",") if value.strip()],
            ),
        )
    if not args.load_in_4bit:
        model.to(device)
    model.print_trainable_parameters()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    warmup_steps = math.ceil(args.max_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)
    optimizer_step = 0
    micro_batches_seen = 0
    resume_micro_batches = 0
    if args.resume_from and (args.resume_from / "training_state.pt").exists():
        state = torch.load(args.resume_from / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        optimizer_step = int(state.get("optimizer_step") or 0)
        resume_micro_batches = int(state.get("micro_batches_seen") or 0)

    def evaluate() -> float | None:
        if dev_loader is None:
            return None
        model.eval()
        values = []
        with torch.no_grad():
            for batch in dev_loader:
                output = model(**{key: value.to(device) for key, value in batch.items()})
                values.append(float(output.loss.detach().cpu()))
        model.train()
        return sum(values) / len(values) if values else None

    def save_checkpoint(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        save_base_model_configs(model, path)
        tokenizer.save_pretrained(path)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer_step": optimizer_step,
                "micro_batches_seen": micro_batches_seen,
                "arguments": vars(args),
            },
            path / "training_state.pt",
        )

    args.output.mkdir(parents=True, exist_ok=True)
    trace: list[dict[str, Any]] = []
    initial_dev_loss = evaluate()
    latest_dev_loss = initial_dev_loss
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    started = time.time()
    while optimizer_step < args.max_steps:
        for batch in train_loader:
            micro_batches_seen += 1
            if micro_batches_seen <= resume_micro_batches:
                continue
            output = model(**{key: value.to(device) for key, value in batch.items()})
            loss = output.loss / args.gradient_accumulation
            loss.backward()
            accumulation += 1
            if accumulation < args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0
            optimizer_step += 1
            row = {
                "step": optimizer_step,
                "train_loss": float(output.loss.detach().cpu()),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            if args.eval_every and optimizer_step % args.eval_every == 0:
                latest_dev_loss = evaluate()
                row["dev_loss"] = latest_dev_loss
            trace.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if args.save_every and optimizer_step % args.save_every == 0:
                save_checkpoint(args.output / f"checkpoint-{optimizer_step}")
            if optimizer_step >= args.max_steps:
                break

    reuse_latest_dev = bool(
        latest_dev_loss is not None and args.eval_every and optimizer_step % args.eval_every == 0
    )
    final_dev_loss = latest_dev_loss if reuse_latest_dev else evaluate()
    save_checkpoint(args.output)
    metrics = {
        "stage": "sft",
        "base_model": args.base_model,
        "train_rows": len(train_examples),
        "dev_rows": len(dev_examples),
        "dev_source_rows": dev_source_rows,
        "train_skipped": train_skipped,
        "dev_skipped": dev_skipped,
        "optimizer_steps": optimizer_step,
        "initial_dev_loss": initial_dev_loss,
        "final_dev_loss": final_dev_loss,
        "elapsed_seconds": time.time() - started,
        "max_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "trace": trace,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
