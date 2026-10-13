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
    parser.add_argument("--model-revision", help="immutable model repository commit used for reproducible loading")
    parser.add_argument("--tokenizer-path", type=Path, help="Explicit saved tokenizer, including its training chat template")
    parser.add_argument("--local-files-only", action="store_true", help="Use only the existing local base, tokenizer and adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        help="existing adapter used only to initialize weights; optimizer, scheduler, and step counters start fresh",
    )
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
    parser.add_argument(
        "--model-dtype",
        choices=("float16", "bfloat16", "float32"),
        help="explicit base-model dtype; overrides --bf16/--no-bf16",
    )
    parser.add_argument(
        "--trainable-params-fp32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="store trainable adapter parameters and accumulated gradients in float32",
    )
    parser.add_argument(
        "--defer-gradient-accumulation-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="backpropagate each raw loss, then average accumulated gradients before clipping",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing-reentrant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=DEFAULT_TARGETS)
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument(
        "--diagnose-gradients-until",
        type=int,
        default=0,
        help="check every trainable gradient through this many micro-batches; zero disables the diagnostic",
    )
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
    if args.init_adapter and args.resume_from:
        raise SystemExit("--init-adapter and --resume-from are mutually exclusive")
    if args.defer_gradient_accumulation_scaling and not args.trainable_params_fp32:
        raise SystemExit("deferred gradient scaling requires --trainable-params-fp32")
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev) if args.dev else []
    dev_source_rows = len(dev_rows)
    validate_rows(train_rows, "train")
    validate_rows(dev_rows, "dev")
    if any(row.get("split") == "test" or (row.get("metadata") or {}).get("split") == "test"
           or (row.get("metadata") or {}).get("test_material") is True for row in train_rows + dev_rows):
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
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path) if args.tokenizer_path else args.base_model,
        revision=None if args.tokenizer_path else args.model_revision,
        local_files_only=args.local_files_only,
        trust_remote_code=not args.local_files_only,
        use_fast=True,
    )
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
            "example_ids": [example["id"] for example in examples],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    dev_loader = DataLoader(dev_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if dev_examples else None

    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    resolved_model_dtype = args.model_dtype or ("bfloat16" if args.bf16 else "float16")
    compute_dtype = dtype_by_name[resolved_model_dtype]
    model_kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "trust_remote_code": not args.local_files_only,
        "low_cpu_mem_usage": True,
        "torch_dtype": compute_dtype,
        "local_files_only": args.local_files_only,
        "attn_implementation": args.attn_implementation,
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
    gradient_checkpointing_kwargs = {"use_reentrant": args.gradient_checkpointing_reentrant}
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if args.gradient_checkpointing_reentrant:
            model.enable_input_require_grads()
    adapter_source = args.resume_from or args.init_adapter
    if adapter_source:
        model = PeftModel.from_pretrained(model, adapter_source, is_trainable=True, local_files_only=args.local_files_only)
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
    if args.trainable_params_fp32:
        for parameter in model.parameters():
            if parameter.requires_grad:
                if not parameter.is_floating_point():
                    raise TypeError(f"trainable parameter is not floating point: {parameter.dtype}")
                parameter.data = parameter.data.to(dtype=torch.float32)
    if not args.load_in_4bit:
        model.to(device)
    model.print_trainable_parameters()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_dtypes: dict[str, int] = {}
    for parameter in parameters:
        key = str(parameter.dtype)
        trainable_parameter_dtypes[key] = trainable_parameter_dtypes.get(key, 0) + parameter.numel()
    if args.trainable_params_fp32 and set(trainable_parameter_dtypes) != {"torch.float32"}:
        raise RuntimeError(f"trainable_parameter_dtype_mismatch:{trainable_parameter_dtypes}")
    print(
        json.dumps(
            {"trainable_parameter_elements_by_dtype": trainable_parameter_dtypes},
            ensure_ascii=False,
        ),
        flush=True,
    )
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
                output = model(**model_inputs(batch))
                value = float(output.loss.detach().cpu())
                if not math.isfinite(value):
                    fail_nonfinite("non_finite_dev_loss", batch, value)
                values.append(value)
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

    def model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(device) for key, value in batch.items() if key != "example_ids"}

    def fail_nonfinite(kind: str, batch: dict[str, Any], value: float, **details: Any) -> None:
        failure = {
            "stage": "sft",
            "reason": kind,
            "value": value,
            "optimizer_step": optimizer_step,
            "micro_batches_seen": micro_batches_seen,
            "example_ids": list(batch.get("example_ids") or []),
            "base_model": args.base_model,
            "base_model_revision": args.model_revision,
            "model_dtype": resolved_model_dtype,
            "trainable_parameter_elements_by_dtype": trainable_parameter_dtypes,
            "gradient_accumulation_scaling": (
                "after_backward" if args.defer_gradient_accumulation_scaling else "before_backward"
            ),
            "gradient_accumulation_storage": (
                "separate_fp32_buffers"
                if args.defer_gradient_accumulation_scaling
                else "parameter_grad"
            ),
            **details,
        }
        (args.output / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"{kind}: examples={failure['example_ids']}")

    initial_dev_loss = evaluate()
    latest_dev_loss = initial_dev_loss
    model.train()
    optimizer.zero_grad(set_to_none=True)
    gradient_accumulators = (
        [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
        if args.defer_gradient_accumulation_scaling
        else []
    )
    gradient_accumulator_used = [False] * len(parameters)
    accumulation = 0
    accumulation_example_ids: list[str] = []
    started = time.time()
    while optimizer_step < args.max_steps:
        for batch in train_loader:
            micro_batches_seen += 1
            if micro_batches_seen <= resume_micro_batches:
                continue
            output = model(**model_inputs(batch))
            raw_loss = float(output.loss.detach().cpu())
            if not math.isfinite(raw_loss):
                fail_nonfinite("non_finite_train_loss", batch, raw_loss)
            loss = (
                output.loss
                if args.defer_gradient_accumulation_scaling
                else output.loss / args.gradient_accumulation
            )
            loss.backward()
            accumulation += 1
            accumulation_example_ids.extend(str(value) for value in batch.get("example_ids") or [])
            if args.diagnose_gradients_until and micro_batches_seen <= args.diagnose_gradients_until:
                named_gradients = [
                    (name, parameter.grad)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                finite_flags = torch.stack(
                    [torch.isfinite(gradient).all() for _, gradient in named_gradients]
                )
                if not bool(finite_flags.all().item()):
                    flags = finite_flags.detach().cpu().tolist()
                    non_finite_parameters = [
                        name for (name, _), finite in zip(named_gradients, flags) if not finite
                    ]
                    fail_nonfinite(
                        "non_finite_microbatch_gradient",
                        batch,
                        raw_loss,
                        accumulation_example_ids=list(accumulation_example_ids),
                        non_finite_gradient_parameters=non_finite_parameters[:32],
                    )
            if args.defer_gradient_accumulation_scaling:
                with torch.no_grad():
                    for index, (parameter, accumulator) in enumerate(
                        zip(parameters, gradient_accumulators)
                    ):
                        if parameter.grad is None:
                            continue
                        accumulator.add_(parameter.grad)
                        gradient_accumulator_used[index] = True
                        parameter.grad = None
            if accumulation < args.gradient_accumulation:
                continue
            if args.defer_gradient_accumulation_scaling:
                with torch.no_grad():
                    for parameter, accumulator, used in zip(
                        parameters,
                        gradient_accumulators,
                        gradient_accumulator_used,
                    ):
                        parameter.grad = (
                            accumulator.div_(args.gradient_accumulation) if used else None
                        )
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu())
            if not math.isfinite(gradient_norm):
                fail_nonfinite(
                    "non_finite_gradient_norm",
                    batch,
                    gradient_norm,
                    raw_loss=raw_loss,
                    accumulation_example_ids=list(accumulation_example_ids),
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if args.defer_gradient_accumulation_scaling:
                for accumulator in gradient_accumulators:
                    accumulator.zero_()
                gradient_accumulator_used = [False] * len(parameters)
            accumulation = 0
            accumulation_example_ids = []
            optimizer_step += 1
            row = {
                "step": optimizer_step,
                "train_loss": raw_loss,
                "gradient_norm": gradient_norm,
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
        "base_model_revision": args.model_revision,
        "adapter_initialization": (
            "resume_checkpoint" if args.resume_from else "existing_adapter" if args.init_adapter else "new_lora"
        ),
        "initial_adapter": str(args.init_adapter.resolve()) if args.init_adapter else None,
        "resume_checkpoint": str(args.resume_from.resolve()) if args.resume_from else None,
        "model_dtype": resolved_model_dtype,
        "trainable_parameter_elements_by_dtype": trainable_parameter_dtypes,
        "gradient_accumulation_scaling": (
            "after_backward" if args.defer_gradient_accumulation_scaling else "before_backward"
        ),
        "gradient_accumulation_storage": (
            "separate_fp32_buffers"
            if args.defer_gradient_accumulation_scaling
            else "parameter_grad"
        ),
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
