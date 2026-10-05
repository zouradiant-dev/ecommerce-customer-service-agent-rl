"""Sample the frozen v18 SFT adapter on train-only tasks for reward review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_sft_v18 import load_local_backend, validate_local_adapter, write_json  # noqa: E402
from ecommerce_customer_service.evaluation_v18 import (  # noqa: E402
    run_train_task_v18,
    summarize_v18_rollouts,
)
from ecommerce_customer_service.experiment_prompt_v18 import (  # noqa: E402
    V18_TRAINED_PROMPT_VERSION,
    V18_TRAINED_SYSTEM_PROMPT,
)
from ecommerce_customer_service.grpo_v18 import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    CALIBRATION_RUNTIME_SOURCES,
    EXPECTED_SFT_ADAPTER_SHA256,
    select_v18_train_tasks,
    source_hashes,
)
from ecommerce_customer_service.local_hf import (  # noqa: E402
    LocalTransformersAgent,
    paired_seed,
    sha256_file,
)
from ecommerce_customer_service.models import TaskSpec  # noqa: E402
from ecommerce_customer_service.tools_v18 import TOOLS_V18_VERSION  # noqa: E402
from ecommerce_customer_service.verifier_v18 import V18_VERIFIER_VERSION  # noqa: E402


DEFAULT_SCENARIOS = (
    "cross_order_item_mismatch",
    "duplicate_refund_request",
    "mixed_order_bulk_cancel",
    "multi_order_disambiguation",
    "order_not_found",
    "partial_quantity_refund",
    "policy_bypass_request",
    "refund_policy_tool_transient_error",
    "refund_quantity_exceeds",
    "update_address",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-model-revision", default=BASE_REVISION)
    parser.add_argument("--scenario", action="append", help="repeat to replace the calibrated scenario set")
    parser.add_argument("--tasks-per-scenario", type=int, default=2)
    parser.add_argument("--trials-per-task", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1807)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.base_model != BASE_MODEL or args.base_model_revision != BASE_REVISION:
        raise ValueError("v18 GRPO calibration requires the pinned base model and revision")
    if args.trials_per_task < 2 or args.tasks_per_scenario < 1:
        raise ValueError("calibration requires at least two trials and one task per scenario")
    if args.device < 0 or min(args.max_steps, args.max_tokens, args.max_input_tokens) < 1:
        raise ValueError("invalid device or token/step budget")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("calibration requires positive finite sampling temperature")


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    args.manifest = args.manifest.resolve()
    args.adapter = args.adapter.resolve()
    args.output = args.output.resolve()
    adapter_config = validate_local_adapter(args.adapter, args.base_model)
    adapter_hash = sha256_file(args.adapter / "adapter_model.safetensors")
    if adapter_hash != EXPECTED_SFT_ADAPTER_SHA256:
        raise ValueError("calibration must start from the frozen original v18 SFT adapter")
    scenarios = tuple(args.scenario or DEFAULT_SCENARIOS)
    selected, plan = select_v18_train_tasks(
        args.manifest,
        tasks_per_scenario=args.tasks_per_scenario,
        scenarios=scenarios,
    )
    protocol = {
        "schema_version": "ecommerce_cs_v18_grpo_calibration_protocol_v1",
        "kind": "v18_grpo_reward_calibration_sampling",
        "split": "train",
        "stage": "sft",
        "tasks": len(selected),
        "task_ids": plan["task_ids"],
        "scenarios": plan["scenarios"],
        "tasks_per_scenario": args.tasks_per_scenario,
        "trials_per_task": args.trials_per_task,
        "rollouts": len(selected) * args.trials_per_task,
        "manifest": str(args.manifest),
        "manifest_sha256": plan["manifest_sha256"],
        "database_sha256": plan["database_sha256"],
        "base_model": args.base_model,
        "base_model_revision": args.base_model_revision,
        "adapter": str(args.adapter),
        "adapter_sha256": adapter_hash,
        "adapter_lora_r": adapter_config.get("r"),
        "prompt_version": V18_TRAINED_PROMPT_VERSION,
        "prompt_sha256": sha256(V18_TRAINED_SYSTEM_PROMPT.encode()).hexdigest(),
        "verifier_version": V18_VERIFIER_VERSION,
        "tools_version": TOOLS_V18_VERSION,
        "tool_surface": "native_dynamic",
        "policy_controller": None,
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": 0,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "max_input_tokens": args.max_input_tokens,
        "correction_retries": 0,
        "cached_only": not args.allow_download,
        "source_sha256": source_hashes(ROOT, CALIBRATION_RUNTIME_SOURCES),
        "formal_rl_gate_passed": False,
    }
    print(json.dumps({"event": "V18_CALIBRATION_PREFLIGHT_OK", "protocol": protocol}, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        return 0
    if args.output.exists():
        raise FileExistsError(f"new calibration output required:{args.output}")
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", protocol)
    model, completer, model_info = load_local_backend(args, args.adapter)
    write_json(args.output / "model_info.json", model_info)
    agent = LocalTransformersAgent(
        completer,
        max_steps=args.max_steps,
        system_prompt=V18_TRAINED_SYSTEM_PROMPT,
    )
    rollout_path = args.output / "rollouts.jsonl"
    pending_path = args.output / "review-pending.jsonl"
    guide_path = args.output / "review-guide.md"
    rows: list[dict] = []
    started = time.monotonic()
    expected = len(selected) * args.trials_per_task
    with guide_path.open("x", encoding="utf-8", newline="\n") as guide:
        guide.write(
            "# v18 GRPO 奖励人工复核\n\n"
            "逐条核对工具事实、最终数据库状态、安全性和沟通内容。复制 `review-pending.jsonl` "
            "为 `reviewed-rollouts.jsonl`，把每条 `human_review.safe` 与 "
            "`human_review.communication_passed` 改成 true/false，并填写 evidence。不要修改模型轨迹。\n\n"
        )
        completed = 0
        for task_path, task in selected:
            source_hash = sha256_file(task.database_path)
            for trial in range(args.trials_per_task):
                completer.seed(paired_seed(args.seed, task.task_id, trial))
                row = run_train_task_v18(
                    TaskSpec.from_file(task_path),
                    agent,
                    trial=trial,
                    stage="sft",
                    model=str(args.adapter),
                    prompt_version=V18_TRAINED_PROMPT_VERSION,
                )
                if sha256_file(task.database_path) != source_hash:
                    raise RuntimeError(f"source database mutated during calibration:{task.task_id}")
                rows.append(row)
                append_jsonl(rollout_path, row)
                pending = {
                    **row,
                    "human_review": {
                        "safe": None,
                        "communication_passed": None,
                        "evidence": "",
                    },
                }
                append_jsonl(pending_path, pending)
                guide.write(
                    f"## {row['rollout_id']}\n\n"
                    f"- shaped reward: `{row['reward_components']['train_shaped_v18']}`\n"
                    f"- strict reward: `{row['reward_components']['strict_binary_v18']}`\n"
                    f"- hard failures: `{row['verification']['hard_safety_failures']}`\n"
                    f"- failures: `{row['verification']['failures']}`\n"
                    f"- final: `{row['final_response']}`\n\n"
                )
                completed += 1
                print(
                    json.dumps(
                        {
                            "event": "V18_CALIBRATION_PROGRESS",
                            "completed": completed,
                            "expected": expected,
                            "task_id": task.task_id,
                            "trial": trial,
                            "strict_reward": row["reward_components"]["strict_binary_v18"],
                            "train_reward": row["reward_components"]["train_shaped_v18"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    summary = summarize_v18_rollouts(
        rows,
        expected_tasks=len(selected),
        trials_per_task=args.trials_per_task,
    )
    result = {
        "complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "protocol_sha256": sha256_file(args.output / "protocol.json"),
        "rollouts_sha256": sha256_file(rollout_path),
        "review_pending_sha256": sha256_file(pending_path),
        "summary": summary,
        "human_review_complete": False,
        "reward_calibrated": False,
        "formal_rl_gate_passed": False,
        "interpretation": "Train-only SFT sampling for human reward calibration; no policy update occurred.",
    }
    write_json(args.output / "summary.json", result)
    print(json.dumps({"event": "V18_CALIBRATION_SAMPLING_COMPLETE", **result}, ensure_ascii=False, indent=2), flush=True)
    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
