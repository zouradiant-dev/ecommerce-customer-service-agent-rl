"""Evaluate learned models on frozen v18 dev.

The default backend uses an OpenAI-compatible server.  ``--backend local-hf``
loads a Qwen base model plus a PEFT LoRA adapter directly with the repository's
existing 4-bit Transformers inference path, so the training image can run the
evaluation without installing a model server.
"""
from __future__ import annotations

import argparse
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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.evaluation_v18 import (  # noqa: E402
    V18_EVALUATION_VERSION,
    compare_v18_stages,
    parse_stage,
    run_task_v18,
    select_v18_dev_tasks,
    summarize_v18_rollouts,
)
from ecommerce_customer_service.experiment_prompt_v18 import (  # noqa: E402
    evaluation_prompt_v18,
)
from ecommerce_customer_service.models import TaskSpec  # noqa: E402
from ecommerce_customer_service.local_hf import (  # noqa: E402
    LocalTransformersAgent,
    TransformersCompleter,
    paired_seed,
    validate_lora_adapter,
)
from ecommerce_customer_service.policy_controller_v18 import (  # noqa: E402
    POLICY_CONTROLLER_V18_VERSION,
    PolicyControlledAgent,
)
from ecommerce_customer_service.simulator import OpenAICompatibleAgent  # noqa: E402
from ecommerce_customer_service.tools_v18 import TOOLS_V18_VERSION  # noqa: E402
from ecommerce_customer_service.verifier_v18 import V18_VERIFIER_VERSION  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def output_must_be_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"evaluation output must be empty:{path}")
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid resume JSONL:{path}:line-{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid resume row:{path}:line-{line_number}")
        rows.append(value)
    return rows


def validate_local_adapter(path: Path, base_model: str) -> dict[str, Any]:
    path = path.resolve()
    config = validate_lora_adapter(path, base_model)
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        if not (path / filename).is_file():
            raise FileNotFoundError(path / filename)
    return config


def load_local_backend(args: argparse.Namespace, adapter: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load the exact base revision and one LoRA adapter in 4-bit for inference."""

    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError("local-hf evaluation requires a valid CUDA device")
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    local_only = not args.allow_download
    common = {
        "revision": args.base_model_revision,
        "local_files_only": local_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **common)
    saved_tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    if tokenizer.get_vocab() != saved_tokenizer.get_vocab() or tokenizer.chat_template != saved_tokenizer.chat_template:
        raise ValueError("base_and_adapter_tokenizers_differ; evaluation prompts would not be comparable")
    for marker in ("<tool_call>", "</tool_call>"):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is None or token_id in tokenizer.all_special_ids:
            raise ValueError(f"unsupported_tool_call_token:{marker}")
    print(
        json.dumps(
            {
                "event": "V18_LOCAL_MODEL_LOADING",
                "base_model": args.base_model,
                "base_model_revision": args.base_model_revision,
                "adapter": str(adapter),
                "quantization": "nf4_double_quant",
                "dtype": str(dtype),
                "cached_only": local_only,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **common,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": args.device},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    if args.max_input_tokens + args.max_tokens > base.config.max_position_embeddings:
        raise ValueError("requested token budgets exceed the model context window")
    saved_config = json.loads((adapter / "config.json").read_text(encoding="utf-8"))
    if saved_config.get("vocab_size") != base.config.vocab_size:
        raise ValueError("adapter_vocab_size_differs_from_base")
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval()
    generation = GenerationConfig(
        max_new_tokens=args.max_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else 1.0,
        top_p=1.0,
        top_k=0 if args.temperature > 0 else 50,
        num_beams=1,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )
    completer = TransformersCompleter(
        model,
        tokenizer,
        torch,
        generation,
        device=f"cuda:{args.device}",
        max_input_tokens=args.max_input_tokens,
    )
    info = {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes")
        },
        "gpu": torch.cuda.get_device_name(args.device),
        "quantization": "nf4_double_quant",
        "dtype": str(dtype),
        "requested_base_revision": args.base_model_revision,
        "loaded_base_revision": getattr(base.config, "_commit_hash", None),
    }
    return model, completer, info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dev-only v18 SFT/checkpoint evaluation"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", action="append", required=True, help="repeat NAME=MODEL_ALIAS")
    parser.add_argument("--backend", choices=("openai", "local-hf"), default="openai")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--base-model-revision",
        default="a09a35458c702b33eeacc393d103063234e8bc28",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=1807)
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--max-tasks-per-scenario", type=int, default=0)
    parser.add_argument(
        "--scenario",
        action="append",
        help="evaluate only this dev scenario; repeat for more scenarios",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument(
        "--correction-retries",
        type=int,
        default=0,
        help="hidden system corrections; keep 0 for formal evaluation",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--policy-controller",
        action="store_true",
        help="gate writes and normalize terminal JSON from public tool evidence",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.trials_per_task < 1:
        raise ValueError("trials_per_task must be positive")
    if args.max_steps < 1 or args.max_tokens < 1 or args.max_input_tokens < 1:
        raise ValueError("max_steps and max_tokens must be positive")
    if args.device < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("device and temperature must be non-negative")
    if args.temperature == 0 and args.trials_per_task != 1:
        raise ValueError("greedy decoding only supports one trial")
    if args.correction_retries < 0:
        raise ValueError("correction_retries must be non-negative")

    parsed_stages = [parse_stage(value) for value in args.stage]
    stages = dict(parsed_stages)
    if len(stages) != len(parsed_stages):
        raise ValueError("stage names must be unique")
    local_configs: dict[str, dict[str, Any]] = {}
    if args.backend == "local-hf":
        stages = {name: str(Path(value).resolve()) for name, value in parsed_stages}
        local_configs = {
            name: validate_local_adapter(Path(adapter), args.base_model)
            for name, adapter in stages.items()
        }
    task_paths, task_plan = select_v18_dev_tasks(
        args.manifest,
        max_tasks_per_scenario=args.max_tasks_per_scenario,
        scenarios=args.scenario,
    )
    effective_prompt_version, effective_system_prompt = evaluation_prompt_v18(
        policy_controller=args.policy_controller
    )
    protocol = {
        "evaluation_version": V18_EVALUATION_VERSION,
        "prompt_version": effective_prompt_version,
        "tools_version": TOOLS_V18_VERSION,
        "verifier_version": V18_VERIFIER_VERSION,
        "task_plan": task_plan,
        "stages": stages,
        "backend": args.backend,
        "api_base": args.api_base if args.backend == "openai" else None,
        "base_model": args.base_model if args.backend == "local-hf" else None,
        "base_model_revision": args.base_model_revision if args.backend == "local-hf" else None,
        "local_adapter_sha256": {
            name: source_hash(Path(adapter) / "adapter_model.safetensors")
            for name, adapter in stages.items()
        } if args.backend == "local-hf" else {},
        "local_lora_r": {
            name: config.get("r") for name, config in local_configs.items()
        },
        "trials_per_task": args.trials_per_task,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "max_input_tokens": args.max_input_tokens,
        "quantization": "nf4_double_quant" if args.backend == "local-hf" else None,
        "cached_only": not args.allow_download if args.backend == "local-hf" else None,
        "correction_retries": args.correction_retries,
        "policy_controller": (
            POLICY_CONTROLLER_V18_VERSION if args.policy_controller else None
        ),
        "test_rows_allowed": False,
        "source_sha256": {
            "evaluate_sft_v18.py": source_hash(ROOT / "scripts/evaluate_sft_v18.py"),
            "evaluation_v18.py": source_hash(ROOT / "src/ecommerce_customer_service/evaluation_v18.py"),
            "experiment_prompt_v18.py": source_hash(ROOT / "src/ecommerce_customer_service/experiment_prompt_v18.py"),
            "tools_v18.py": source_hash(ROOT / "src/ecommerce_customer_service/tools_v18.py"),
            "verifier_v18.py": source_hash(ROOT / "src/ecommerce_customer_service/verifier_v18.py"),
            "local_hf.py": source_hash(ROOT / "src/ecommerce_customer_service/local_hf.py"),
            "policy_controller_v18.py": source_hash(
                ROOT / "src/ecommerce_customer_service/policy_controller_v18.py"
            ),
        },
    }
    if args.validate_only:
        print(json.dumps({"valid": True, "protocol": protocol}, ensure_ascii=False, indent=2))
        return 0

    protocol_path = args.output / "protocol.json"
    if args.resume:
        if not protocol_path.is_file():
            raise FileNotFoundError(f"resume requires protocol.json:{protocol_path}")
        saved_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved_protocol != protocol:
            raise ValueError("resume protocol mismatch; use a new output directory")
    else:
        output_must_be_empty(args.output)
        write_json(protocol_path, protocol)
    rows_by_stage: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    expected = len(task_paths) * args.trials_per_task
    expected_keys = {
        (TaskSpec.from_file(path).task_id, trial)
        for path in task_paths
        for trial in range(args.trials_per_task)
    }
    for stage, model in stages.items():
        stage_dir = args.output / stage
        stage_dir.mkdir(parents=True, exist_ok=args.resume)
        rollout_path = stage_dir / "rollouts.jsonl"
        rows = read_jsonl(rollout_path) if args.resume and rollout_path.is_file() else []
        completed_keys: set[tuple[str, int]] = set()
        for row in rows:
            if row.get("stage") != stage or row.get("model") != model:
                raise ValueError(f"resume stage/model mismatch:{rollout_path}")
            key = (str(row.get("task_id") or ""), int(row.get("trial", -1)))
            if key not in expected_keys:
                raise ValueError(f"unexpected resume task/trial:{key}")
            if key in completed_keys:
                raise ValueError(f"duplicate resume task/trial:{key}")
            completed_keys.add(key)
        backend_model = completer = model_info = None
        if args.backend == "local-hf" and len(completed_keys) < expected:
            backend_model, completer, model_info = load_local_backend(args, Path(model))
            write_json(stage_dir / "model_info.json", model_info)
        stage_started = time.monotonic()
        mode = "a" if rollout_path.exists() else "x"
        with rollout_path.open(mode, encoding="utf-8", newline="\n") as handle:
            for task_path in task_paths:
                task = TaskSpec.from_file(task_path)
                for trial in range(args.trials_per_task):
                    if (task.task_id, trial) in completed_keys:
                        continue
                    print(
                        json.dumps(
                            {
                                "event": "V18_EVAL_TASK_START",
                                "stage": stage,
                                "task_id": task.task_id,
                                "trial": trial,
                                "completed": len(rows),
                                "expected": expected,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if args.backend == "local-hf":
                        assert completer is not None
                        rollout_seed = paired_seed(args.seed, task.task_id, trial)
                        completer.seed(rollout_seed)
                        agent = LocalTransformersAgent(
                            completer,
                            max_steps=args.max_steps,
                            system_prompt=effective_system_prompt,
                        )
                    else:
                        rollout_seed = None
                        agent = OpenAICompatibleAgent(
                            model,
                            api_base=args.api_base,
                            api_key=args.api_key,
                            temperature=args.temperature,
                            max_steps=args.max_steps,
                            max_tokens=args.max_tokens,
                            correction_retries=args.correction_retries,
                            timeout_seconds=args.timeout_seconds,
                            retries=args.request_retries,
                            system_prompt=effective_system_prompt,
                        )
                    if args.policy_controller:
                        agent = PolicyControlledAgent(agent)
                    row = run_task_v18(
                        task,
                        agent,
                        trial=trial,
                        stage=stage,
                        model=model,
                        prompt_version=effective_prompt_version,
                    )
                    if args.policy_controller:
                        row["policy_controller"] = dict(agent.last_audit)
                    row["inference"] = {
                        "backend": args.backend,
                        "seed": rollout_seed,
                    }
                    if row.get("error"):
                        write_json(stage_dir / "last_worker_error.json", row)
                        raise RuntimeError(
                            "infrastructure error; completed rows are preserved and can be resumed:"
                            + str(row["error"])
                        )
                    rows.append(row)
                    completed_keys.add((task.task_id, trial))
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "event": "V18_EVAL_TASK_DONE",
                                "stage": stage,
                                "task_id": task.task_id,
                                "trial": trial,
                                "completed": len(rows),
                                "expected": expected,
                                "reward": row["reward"],
                                "stopped_reason": row["stopped_reason"],
                                "elapsed_seconds": time.monotonic() - stage_started,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if args.progress_every and len(rows) % args.progress_every == 0:
                        passed = sum(float(item["reward"]) >= 1.0 for item in rows)
                        print(
                            json.dumps(
                                {
                                    "event": "V18_EVAL_PROGRESS",
                                    "stage": stage,
                                    "completed": len(rows),
                                    "expected": expected,
                                    "strict_passes": passed,
                                    "worker_errors": sum(bool(item.get("error")) for item in rows),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
        summary = summarize_v18_rollouts(
            rows,
            expected_tasks=len(task_paths),
            trials_per_task=args.trials_per_task,
        )
        summary["elapsed_seconds_this_process"] = time.monotonic() - stage_started
        if args.backend == "local-hf" and backend_model is not None:
            import torch

            summary["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated(args.device) / 1024**3
            summary["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(args.device) / 1024**3
            summary["model_info"] = model_info
        rows_by_stage[stage] = rows
        summaries[stage] = summary
        write_json(stage_dir / "summary.json", summary)
        if args.backend == "local-hf" and backend_model is not None:
            if "agent" in locals():
                del agent
            del completer, backend_model
            import torch

            torch.cuda.empty_cache()

    baseline_name = next(iter(stages))
    comparisons = {
        stage: compare_v18_stages(rows_by_stage[baseline_name], rows)
        for stage, rows in rows_by_stage.items()
        if stage != baseline_name
    }
    final = {
        "schema_version": "ecommerce_cs_v18_multi_stage_evaluation_v1",
        "complete": all(summary["complete"] for summary in summaries.values()),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_stage": baseline_name,
        "protocol": protocol,
        "stage_summaries": summaries,
        "paired_comparisons_to_baseline": comparisons,
    }
    write_json(args.output / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    worker_errors = sum(
        summary["overall"]["worker_errors"] for summary in summaries.values()
    )
    return 2 if worker_errors or not final["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
