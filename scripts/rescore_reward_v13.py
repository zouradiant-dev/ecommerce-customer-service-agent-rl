"""CPU replay a saved v12 train pilot into a new explicit-v13 derived report.

No generation, training or source writes. Original v12 files are never relabelled.
Generation hashes and the active scoring snapshot are separately recorded.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.data import load_jsonl
from ecommerce_customer_service.evaluation import evaluate_rollouts
from ecommerce_customer_service.grpo_readiness import sampling_checks, validate_smoke_evidence
from ecommerce_customer_service.local_hf import sha256_file
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.online_env import EcommerceCustomerServiceEnv
from ecommerce_customer_service.tasks import load_task_paths
from ecommerce_customer_service.verifier import VERIFIER_VERSION, V13_VERIFIER_VERSION

FROZEN_V12_VERIFIER_SHA256 = "60883540b6226c43ac9ec8c90da616d0b9ec95476aeae3772c4e6720c0270086"
SCORING_SOURCES = (
    "src/ecommerce_customer_service/communication.py",
    "src/ecommerce_customer_service/communication_facts.py",
    "src/ecommerce_customer_service/communication_facts_v13.py",
    "src/ecommerce_customer_service/verifier.py",
    "src/ecommerce_customer_service/rescoring.py",
    "src/ecommerce_customer_service/online_env.py",
    "src/ecommerce_customer_service/tools.py",
    "src/ecommerce_customer_service/rewards.py",
    "src/ecommerce_customer_service/grpo_readiness.py",
    "scripts/rescore_reward_v13.py",
)
DERIVED_FIELDS = {"verifier", "verifier_version", "reward", "reward_components", "rescore_metadata"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normal(value):
    return json.loads(json.dumps(value))


def file_hashes(directory):
    return {str(path.relative_to(directory)): sha256_file(path)
            for path in directory.rglob("*") if path.is_file()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(pilot: Path, manifest: Path, output: Path) -> dict:
    pilot, manifest, output = (path.resolve() for path in (pilot, manifest, output))
    require(not output.exists() and not output.is_relative_to(pilot) and not pilot.is_relative_to(output),
            "new_non_nested_output_required; no overwrites")
    require(pilot.is_dir(), "source_pilot_missing")
    original_hashes = file_hashes(pilot)
    source_protocol, source_summary = load(pilot / "protocol.json"), load(pilot / "summary.json")
    info, original_review = load(pilot / "model_info.json"), load(pilot / "reward_review.json")
    raw_path = pilot / "sft/rollouts.jsonl"
    raw_hash = sha256_file(raw_path)
    require(source_summary["complete"] and source_protocol == source_summary["protocol"], "incomplete_or_stale_source_summary")
    require(source_protocol["split"] == "train" and source_protocol["trials_per_task"] >= 2,
            "inline_train_multi_trial_pilot_required")
    require(source_protocol["verifier_version"] == VERIFIER_VERSION == "deterministic_semantic_v12",
            "original_frozen_v12_pilot_required")
    require(source_protocol["manifest_sha256"] == sha256_file(manifest), "manifest_hash_mismatch")
    require(original_review["rollouts_sha256"] == raw_hash, "source_review_hash_mismatch")
    generation_sources = source_protocol["source_sha256"]
    require(generation_sources.get("src/ecommerce_customer_service/verifier.py") == FROZEN_V12_VERIFIER_SHA256,
            "unsupported_generation_verifier_snapshot")
    for relative, digest in generation_sources.items():
        # Only the version router changed. Frozen v12 semantics are reproduced
        # row-for-row below; never claim its new source bytes generated old rows.
        if relative != "src/ecommerce_customer_service/verifier.py":
            require(sha256_file(ROOT / relative) == digest, "generation_dependency_changed:" + relative)
    scoring_hashes = {name: sha256_file(ROOT / name) for name in SCORING_SOURCES}
    rows = load_jsonl(raw_path)
    require(len(rows) == source_protocol["rollouts"], "incomplete_source_rollouts")
    original_checks = sampling_checks(rows, trials_per_task=source_protocol["trials_per_task"],
                                      expected_task_ids=source_protocol["task_ids"])
    require(original_checks == load(pilot / "readiness.json") == source_summary["sampling_checks"],
            "stale_source_readiness")
    require(all(original_checks["checks"][name] for name in (
        "complete_train_groups", "multiple_trials", "no_worker_errors",
        "no_invalid_calls_or_truncation", "no_max_step_exhaustion")), "invalid_source_execution_groups")
    require(load(pilot / "sft/report.json")["evaluation"] == evaluate_rollouts(rows), "stale_source_evaluation")
    tasks = {task.task_id: (path, task) for path in load_task_paths(manifest, split="train")
             for task in (TaskSpec.from_file(path),) if task.task_id in source_protocol["task_ids"]}
    require(set(tasks) == set(source_protocol["task_ids"]), "source_tasks_missing")
    database_hashes = {key: sha256_file(task.database_path) for key, (_, task) in tasks.items()}
    require(database_hashes == source_protocol["source_database_sha256"], "source_database_mismatch")
    reviewed = {(item["task_id"], item["trial"]): item for item in original_review.get("rows", [])}
    derived, changes, new_reviews = [], [], []
    for index, row in enumerate(rows, 1):
        require(row.get("split") == "train" and row.get("verifier_version") == VERIFIER_VERSION,
                "source_row_version_mismatch")
        task_path, task = tasks[row["task_id"]]
        require(not task.metadata.get("scripted_user_replies"), "non_inline_task")
        require([message["content"] for message in row["messages"] if message["role"] == "user"] == [task.user_prompt],
                "source_user_prompt_mismatch")
        env = EcommerceCustomerServiceEnv(task_path=task_path)
        try:
            env.reset()
            for call in row["tool_executions"]:
                require(env._tools.execute(call["name"], call["arguments"]).to_dict() == call,
                        f"tool_replay_mismatch:{index}")
            final = row["final_response"]
            old_reward = env._evaluate(final, verifier_version=VERIFIER_VERSION)
            old_verifier = normal(env._last_verification.to_dict())
            require(old_reward == row["reward"] and old_verifier == row["verifier"],
                    f"frozen_v12_replay_mismatch:{index}")
            new_reward = env._evaluate(final, verifier_version=V13_VERIFIER_VERSION)
            new_verifier = normal(env._last_verification.to_dict())
            for component in ("state_passed", "policy_passed", "required_tools_passed"):
                require(new_verifier[component] == old_verifier[component], "non_fact_component_changed")
            new_row = deepcopy(row)
            new_row.update({"reward": new_reward, "verifier": new_verifier, "verifier_version": V13_VERIFIER_VERSION,
                            "reward_components": env._last_breakdown.to_dict(), "rescore_metadata": {
                                "method": "isolated_cpu_tool_replay", "source_verifier_version": VERIFIER_VERSION,
                                "target_verifier_version": V13_VERIFIER_VERSION, "source_reward": old_reward,
                                "original_rollouts_sha256": raw_hash, "original_row_1based": index,
                                "scoring_source_sha256": scoring_hashes, "generation_source_sha256": generation_sources,
                                "model_rerun": False, "original_report_unchanged": True,
                            }})
            require({key: value for key, value in row.items() if key not in DERIVED_FIELDS}
                    == {key: value for key, value in new_row.items() if key not in DERIVED_FIELDS},
                    "unscored_row_fields_changed")
            derived.append(new_row)
            if new_reward != old_reward or new_verifier != old_verifier:
                changes.append({"row_1based": index, "task_id": row["task_id"], "trial": row["trial"],
                                "reward_before": old_reward, "reward_after": new_reward,
                                "removed_failures": sorted(set(old_verifier["failures"]) - set(new_verifier["failures"])),
                                "added_failures": sorted(set(new_verifier["failures"]) - set(old_verifier["failures"])),
                                "final_response_unchanged": final})
            source_review = reviewed.get((row["task_id"], row["trial"]), {})
            if source_review:
                require(source_review.get("recorded_reward") == old_reward, "stale_source_review_row")
            expected = source_review.get("expected_reward")
            new_reviews.append({"task_id": row["task_id"], "trial": row["trial"], "recorded_reward": new_reward,
                                "expected_reward": expected, "status": "agent_replay_matches_prior_audit_not_signoff"
                                if expected == new_reward else "pending_or_disputed",
                                "evidence": source_review.get("evidence", ""),
                                "source_review_status": source_review.get("status"),
                                "original_recorded_reward": old_reward, "human_signoff": False})
        finally:
            env._cleanup()
    require(original_hashes == file_hashes(pilot), "original_source_files_modified")
    require(database_hashes == {key: sha256_file(task.database_path) for key, (_, task) in tasks.items()},
            "source_database_modified")
    require(scoring_hashes == {name: sha256_file(ROOT / name) for name in SCORING_SOURCES}, "scorer_changed_during_replay")
    timestamp = datetime.now(timezone.utc).isoformat()
    checks = sampling_checks(derived, trials_per_task=source_protocol["trials_per_task"],
                             expected_task_ids=source_protocol["task_ids"])
    protocol = deepcopy(source_protocol)
    protocol.update({"kind": "train_readiness_offline_rescore", "verifier_version": V13_VERIFIER_VERSION,
                     "generation_protocol": source_protocol, "generation_source_sha256": generation_sources,
                     "source_sha256": {**{name: sha256_file(ROOT / name) for name in generation_sources}, **scoring_hashes},
                     "source_sha256_role": "active_scoring_snapshot; NOT new model generation",
                     "scoring_source_sha256": scoring_hashes, "rescore": {
                         "method": "isolated_cpu_tool_replay", "completed_at_utc": timestamp,
                         "source_pilot": str(pilot), "source_protocol_sha256": original_hashes["protocol.json"],
                         "source_rollouts_sha256": raw_hash, "model_rerun": False,
                         "frozen_v12_rewards_reproduced": True, "training_authorized": False,
                     }})
    report = {"complete": True, "completed_at_utc": timestamp, "rollouts": len(rows),
              "source_verifier_version": VERIFIER_VERSION, "verifier_version": V13_VERIFIER_VERSION,
              "passed_before": sum(row["reward"] == 1.0 for row in rows),
              "passed_after": sum(row["reward"] == 1.0 for row in derived), "changed_verifier_rows": len(changes),
              "zero_to_one": sum(item["reward_before"] == 0 and item["reward_after"] == 1 for item in changes),
              "one_to_zero": sum(item["reward_before"] == 1 and item["reward_after"] == 0 for item in changes),
              "all_frozen_v12_rewards_and_tools_reproduced": True, "state_policy_and_required_tools_preserved": True,
              "original_files_and_databases_unchanged": True,
              "prior_audit_expected_labels_all_match_new_rewards": all(item["expected_reward"] == item["recorded_reward"] for item in new_reviews),
              "source_files_sha256": original_hashes, "source_database_sha256": database_hashes,
              "generation_source_sha256": generation_sources, "scoring_source_sha256": scoring_hashes,
              "sampling_checks": checks, "model_rerun": False, "training_started": False,
              "server_deployed": False, "training_default_version": VERIFIER_VERSION,
              "reward_calibrated": False, "human_signoff": False, "formal_rl_gate_passed": False,
              "interpretation": "CPU correction of saved train rewards, not model improvement or held-out evidence;"
              " new scoring is explicit opt-in and this artifact does not authorize GRPO.",
              "output": str(output)}
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "sft/rollouts.jsonl", derived)
    write_jsonl(output / "changes.jsonl", changes)
    write_json(output / "sft/report.json", {"evaluation": evaluate_rollouts(derived),
                                           "source_databases_unchanged": True, "rescored": True})
    write_json(output / "protocol.json", protocol)
    write_json(output / "model_info.json", info)
    write_json(output / "readiness.json", checks)
    write_json(output / "reward_review.json", {"reviewer": "Codex agent replay, not human signoff", "reviewed_at_utc": timestamp,
                                               "rollouts_sha256": sha256_file(output / "sft/rollouts.jsonl"), "rows": new_reviews,
                                               "reward_calibrated": False, "human_signoff": False})
    write_json(output / "summary.json", {"complete": True, "protocol": protocol, "model_info": info,
                                         "sampling_checks": checks, "evaluation": evaluate_rollouts(derived),
                                         "interpretation": report["interpretation"], "formal_rl_gate_passed": False})
    try:
        validate_smoke_evidence(output, adapter_sha256=protocol["adapter_sha256"], manifest_sha256=protocol["manifest_sha256"])
    except ValueError as error:
        report["smoke_validator_rejection"] = str(error)
    else:
        raise ValueError("unsigned_derivation_unexpectedly_permits_training")
    write_json(output / "rescore_report.json", report)
    write_json(output / "provenance.json", {"source_pilot": str(pilot), "source_files_sha256": original_hashes,
              "generation_protocol_sha256": original_hashes["protocol.json"], "generation_source_sha256": generation_sources,
              "scoring_source_sha256": scoring_hashes, "source_rollouts_sha256": raw_hash,
              "rescored_rollouts_sha256": sha256_file(output / "sft/rollouts.jsonl"),
              "source_protocol_and_reports_unchanged": True, "original_generation_provenance_retained": True,
              "model_rerun": False, "training_started": False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = run(args.pilot, args.manifest, args.output)
    keys = ("complete", "rollouts", "source_verifier_version", "verifier_version", "passed_before", "passed_after",
            "changed_verifier_rows", "zero_to_one", "one_to_zero", "all_frozen_v12_rewards_and_tools_reproduced",
            "original_files_and_databases_unchanged", "prior_audit_expected_labels_all_match_new_rewards",
            "sampling_checks", "smoke_validator_rejection", "training_started", "formal_rl_gate_passed", "output")
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
