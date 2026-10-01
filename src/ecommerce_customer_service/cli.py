from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import (
    assert_no_template_leakage,
    build_offline_group_dataset,
    build_sft_dataset,
    combine_rollout_files,
    curate_trajectories,
    load_jsonl,
    merge_successful_retries,
    write_jsonl,
)
from .diagnostics import analyze_bad_cases, assess_rl_gate
from .evaluation import compare_rollouts, evaluate_rollouts
from .external_data import convert_abcd_dataset
from .model_config import DEFAULT_BASE_MODEL, qwen_7b_profile
from .models import TaskSpec
from .rescoring import rescore_rollouts
from .simulator import OpenAICompatibleAgent, ReferenceAgent, WeakBaselineAgent, collect_rollouts
from .tasks import (
    build_failed_task_manifest,
    build_split_manifest,
    build_task_dataset,
    build_template_pilot_manifest,
    load_task_paths,
)


CRITICAL_TOOLS = {
    "verify_customer_identity",
    "get_order",
    "search_refund_policy",
    "update_shipping_address",
    "cancel_order",
    "create_refund",
    "escalate_to_human",
}


def _render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(value) + "\n", encoding="utf-8")


def _task_paths(args: argparse.Namespace) -> list[Path]:
    paths = load_task_paths(args.manifest, split=getattr(args, "split", None))
    scenario = getattr(args, "scenario", None)
    if scenario:
        paths = [path for path in paths if TaskSpec.from_file(path).scenario == scenario]
    limit = getattr(args, "limit", None)
    return paths[:limit] if limit is not None else paths


def command_build_tasks(args: argparse.Namespace) -> dict[str, Any]:
    summary = build_task_dataset(args.output, replicas=args.replicas, seed=args.seed)
    summary["template_assignments"] = assert_no_template_leakage(args.output / "tasks.jsonl")
    return summary


def command_build_pilot(args: argparse.Namespace) -> dict[str, Any]:
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    templates = None
    if args.templates:
        templates = tuple(value.strip() for value in args.templates.split(",") if value.strip())
    return build_template_pilot_manifest(args.manifest, args.output, splits=splits, templates=templates)


def command_build_retry(args: argparse.Namespace) -> dict[str, Any]:
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    return build_failed_task_manifest(
        args.manifest,
        args.rollouts,
        args.output,
        splits=splits,
        failure_mode=args.failure_mode,
    )


def command_build_split(args: argparse.Namespace) -> dict[str, Any]:
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    return build_split_manifest(args.manifest, args.output, splits=splits)


def command_merge_retries(args: argparse.Namespace) -> dict[str, Any]:
    return merge_successful_retries(args.base, args.retry, args.output)


def command_combine_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    return combine_rollout_files(args.input, args.output)


def command_run_scripted(args: argparse.Namespace) -> dict[str, Any]:
    agent_class = ReferenceAgent if args.agent == "reference" else WeakBaselineAgent
    return collect_rollouts(
        _task_paths(args),
        lambda task, trial: agent_class(),
        args.output,
        trials_per_task=args.trials_per_task,
    )


def command_run_model(args: argparse.Namespace) -> dict[str, Any]:
    agent_class = OpenAICompatibleAgent
    if args.runtime == "langgraph":
        try:
            from .langgraph_agent import LangGraphCustomerServiceAgent
        except ImportError as exc:
            raise SystemExit('Install the optional Agent runtime with: pip install -e ".[agent]"') from exc
        agent_class = LangGraphCustomerServiceAgent
    return collect_rollouts(
        _task_paths(args),
        lambda task, trial: agent_class(
            args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            temperature=args.temperature,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            correction_retries=args.correction_retries,
        ),
        args.output,
        trials_per_task=args.trials_per_task,
        verbose=args.verbose,
    )


def command_build_sft(args: argparse.Namespace) -> dict[str, Any]:
    return build_sft_dataset(args.rollouts, args.output, tool_scope=args.tool_scope)


def command_import_abcd(args: argparse.Namespace) -> dict[str, Any]:
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    return convert_abcd_dataset(
        args.source,
        args.output,
        splits=splits,
        use_delexed=not args.original_text,
    )


def command_curate(args: argparse.Namespace) -> dict[str, Any]:
    kept, rejected = curate_trajectories(
        load_jsonl(args.rollouts),
        keep_successes=True,
        keep_valuable_failures=args.keep_valuable_failures,
        max_per_scenario=args.max_per_scenario,
    )
    write_jsonl(args.output, kept)
    write_jsonl(args.rejected, rejected)
    return {"kept": len(kept), "rejected": len(rejected)}


def command_build_grpo(args: argparse.Namespace) -> dict[str, Any]:
    return build_offline_group_dataset(args.rollouts, args.output)


def command_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_rollouts(load_jsonl(args.rollouts))


def command_rescore(args: argparse.Namespace) -> dict[str, Any]:
    return rescore_rollouts(args.rollouts, args.manifest, args.output)


def command_compare(args: argparse.Namespace) -> dict[str, Any]:
    return compare_rollouts(load_jsonl(args.base), load_jsonl(args.candidate))


def command_badcases(args: argparse.Namespace) -> dict[str, Any]:
    return analyze_bad_cases(load_jsonl(args.rollouts))


def command_rl_gate(args: argparse.Namespace) -> dict[str, Any]:
    scores = [float(value) for value in args.checkpoint_pass_rates.split(",") if value.strip()]
    critical = {value.strip() for value in args.critical_tools.split(",") if value.strip()}
    return assess_rl_gate(load_jsonl(args.rollouts), scores, critical_tools=critical)


def command_model_info(args: argparse.Namespace) -> dict[str, Any]:
    return qwen_7b_profile()


def command_local_demo(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.output / "dataset"
    rollout_dir = args.output / "rollouts"
    report_dir = args.output / "reports"
    summary = build_task_dataset(dataset_dir, replicas=args.replicas, seed=args.seed)
    manifest = dataset_dir / "tasks.jsonl"
    assert_no_template_leakage(manifest)
    task_paths = load_task_paths(manifest)
    baseline_path = rollout_dir / "scripted_baseline.jsonl"
    teacher_path = rollout_dir / "verified_teacher.jsonl"
    baseline_collection = collect_rollouts(
        task_paths,
        lambda task, trial: WeakBaselineAgent(),
        baseline_path,
        trials_per_task=1,
    )
    teacher_collection = collect_rollouts(
        task_paths,
        lambda task, trial: ReferenceAgent(),
        teacher_path,
        trials_per_task=1,
    )
    baseline_rows = load_jsonl(baseline_path)
    teacher_rows = load_jsonl(teacher_path)
    baseline_eval = evaluate_rollouts(baseline_rows)
    teacher_eval = evaluate_rollouts(teacher_rows)
    badcases = analyze_bad_cases(baseline_rows)
    comparison = compare_rollouts(baseline_rows, teacher_rows)
    sft_summary = build_sft_dataset(teacher_path, args.output / "sft")
    _write_json(report_dir / "baseline_eval.json", baseline_eval)
    _write_json(report_dir / "teacher_eval.json", teacher_eval)
    _write_json(report_dir / "baseline_badcases.json", badcases)
    _write_json(report_dir / "scripted_comparison.json", comparison)
    payload = {
        "note": "Scripted agents validate the environment only; these numbers are not model-training claims.",
        "dataset": summary,
        "baseline_collection": baseline_collection,
        "teacher_collection": teacher_collection,
        "baseline_pass_1": baseline_eval["pass_1"],
        "teacher_pass_1": teacher_eval["pass_1"],
        "sft": sft_summary,
        "next_step": "Run run-model against a real base checkpoint with multiple trials.",
    }
    _write_json(args.output / "summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ecommerce customer-service agent data and evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-tasks", help="build task-isolated SQLite databases")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--replicas", type=int, default=3)
    build.add_argument("--seed", type=int, default=7)
    build.set_defaults(handler=command_build_tasks)

    pilot = sub.add_parser("build-pilot", help="select one task per template for a small rollout pilot")
    pilot.add_argument("--manifest", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--splits", default="train,dev", help="comma-separated train/dev/test splits")
    pilot.add_argument("--templates", help="optional comma-separated template ids")
    pilot.set_defaults(handler=command_build_pilot)

    retry = sub.add_parser("build-retry", help="select failed tasks for a targeted rerun")
    retry.add_argument("--manifest", type=Path, required=True)
    retry.add_argument("--rollouts", type=Path, required=True)
    retry.add_argument("--output", type=Path, required=True)
    retry.add_argument("--splits", default="train,dev", help="comma-separated eligible splits")
    retry.add_argument("--failure-mode", choices=["all", "worker_error"], default="all")
    retry.set_defaults(handler=command_build_retry)

    split_manifest = sub.add_parser("build-split", help="select every task in requested splits")
    split_manifest.add_argument("--manifest", type=Path, required=True)
    split_manifest.add_argument("--output", type=Path, required=True)
    split_manifest.add_argument("--splits", required=True, help="comma-separated train/dev/test splits")
    split_manifest.set_defaults(handler=command_build_split)

    merge = sub.add_parser("merge-retries", help="replace failed base rows with successful retries")
    merge.add_argument("--base", type=Path, required=True)
    merge.add_argument("--retry", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(handler=command_merge_retries)

    combine = sub.add_parser("combine-rollouts", help="combine disjoint rollout files")
    combine.add_argument("--input", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.set_defaults(handler=command_combine_rollouts)

    scripted = sub.add_parser("run-scripted", help="run the integration baseline or verified reference")
    scripted.add_argument("--manifest", type=Path, required=True)
    scripted.add_argument("--output", type=Path, required=True)
    scripted.add_argument("--agent", choices=["baseline", "reference"], default="baseline")
    scripted.add_argument("--split", choices=["train", "dev", "test"])
    scripted.add_argument("--trials-per-task", type=int, default=1)
    scripted.set_defaults(handler=command_run_scripted)

    model = sub.add_parser("run-model", help="collect real model rollouts through an OpenAI-compatible endpoint")
    model.add_argument("--manifest", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help=f"served model id (default: {DEFAULT_BASE_MODEL})",
    )
    model.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    model.add_argument("--api-key")
    model.add_argument("--runtime", choices=["native", "langgraph"], default="native")
    model.add_argument("--split", choices=["train", "dev", "test"])
    model.add_argument("--scenario", help="run only one named business scenario")
    model.add_argument("--limit", type=int, help="run at most this many matching tasks")
    model.add_argument("--trials-per-task", type=int, default=4)
    model.add_argument("--temperature", type=float, default=0.0)
    model.add_argument("--max-steps", type=int, default=20)
    model.add_argument("--max-tokens", type=int, default=512, help="maximum new tokens per model turn")
    model.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable provider reasoning; Qwen3.7-Plus defaults to disabled to reduce latency and cost",
    )
    model.add_argument("--correction-retries", type=int, default=3, help="maximum state-aware corrective prompts")
    model.add_argument("--verbose", action="store_true", help="print each user, assistant and tool turn")
    model.set_defaults(handler=command_run_model)

    sft = sub.add_parser("build-sft", help="build assistant-only SFT rows from verified teacher rollouts")
    sft.add_argument("--rollouts", type=Path, required=True)
    sft.add_argument("--output", type=Path, required=True)
    sft.add_argument(
        "--tool-scope",
        choices=["all", "trajectory", "stateful_turn"],
        default="all",
        help="declare all tools, trajectory tools, or deployment-matched tools for each assistant turn",
    )
    sft.set_defaults(handler=command_build_sft)

    abcd = sub.add_parser("import-abcd", help="convert a local ABCD JSON/JSON.GZ file to language-only SFT")
    abcd.add_argument("--source", type=Path, required=True)
    abcd.add_argument("--output", type=Path, required=True)
    abcd.add_argument("--splits", default="train,dev", help="comma-separated train/dev splits; test is rejected")
    abcd.add_argument(
        "--original-text",
        action="store_true",
        help="use original fictional customer data instead of the safer delexicalized turns",
    )
    abcd.set_defaults(handler=command_import_abcd)

    curate = sub.add_parser("curate", help="deduplicate, balance and retain valuable failures")
    curate.add_argument("--rollouts", type=Path, required=True)
    curate.add_argument("--output", type=Path, required=True)
    curate.add_argument("--rejected", type=Path, required=True)
    curate.add_argument("--keep-valuable-failures", action=argparse.BooleanOptionalAction, default=True)
    curate.add_argument("--max-per-scenario", type=int)
    curate.set_defaults(handler=command_curate)

    grpo = sub.add_parser(
        "build-grpo",
        help="legacy alias: build mixed static rollout groups for offline AWR, not online GRPO",
    )
    grpo.add_argument("--rollouts", type=Path, required=True)
    grpo.add_argument("--output", type=Path, required=True)
    grpo.set_defaults(handler=command_build_grpo)

    offline_groups = sub.add_parser(
        "build-offline-groups",
        help="build mixed static rollout groups for the offline group-AWR baseline",
    )
    offline_groups.add_argument("--rollouts", type=Path, required=True)
    offline_groups.add_argument("--output", type=Path, required=True)
    offline_groups.set_defaults(handler=command_build_grpo)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--rollouts", type=Path, required=True)
    evaluate.set_defaults(handler=command_evaluate)

    rescore = sub.add_parser("rescore", help="recompute communication and required-tool checks")
    rescore.add_argument("--rollouts", type=Path, required=True)
    rescore.add_argument("--manifest", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    rescore.set_defaults(handler=command_rescore)

    compare = sub.add_parser("compare")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.set_defaults(handler=command_compare)

    badcases = sub.add_parser("badcases")
    badcases.add_argument("--rollouts", type=Path, required=True)
    badcases.set_defaults(handler=command_badcases)

    gate = sub.add_parser("rl-gate")
    gate.add_argument("--rollouts", type=Path, required=True)
    gate.add_argument("--checkpoint-pass-rates", required=True, help="comma-separated SFT validation pass^1 values")
    gate.add_argument("--critical-tools", default=",".join(sorted(CRITICAL_TOOLS)))
    gate.set_defaults(handler=command_rl_gate)

    model_info = sub.add_parser("model-info", help="show the Qwen 7B base-model and training profile")
    model_info.set_defaults(handler=command_model_info)

    demo = sub.add_parser("run-local-demo", help="run a dependency-free environment/data/eval smoke test")
    demo.add_argument("--output", type=Path, default=Path("outputs/local-demo"))
    demo.add_argument("--replicas", type=int, default=3)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(handler=command_local_demo)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = args.handler(args)
    print(_render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
