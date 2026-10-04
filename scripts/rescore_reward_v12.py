"""CPU replay of saved train trajectories; preserve v11 files and training rewards."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.data import load_jsonl, write_jsonl
from ecommerce_customer_service.local_hf import sha256_file
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.online_env import EcommerceCustomerServiceEnv, _final_response
from ecommerce_customer_service.tasks import load_task_paths
from ecommerce_customer_service.verifier import LEGACY_VERIFIER_VERSION, VERIFIER_VERSION

SCORING_SOURCES = (
    "src/ecommerce_customer_service/communication.py",
    "src/ecommerce_customer_service/communication_facts.py",
    "src/ecommerce_customer_service/verifier.py",
    "src/ecommerce_customer_service/online_env.py",
    "src/ecommerce_customer_service/tools.py", "src/ecommerce_customer_service/rewards.py",
    "scripts/rescore_reward_v12.py",
)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def group_stats(rows, *, smoke):
    groups = []
    if smoke:
        for start in range(0, len(rows), 4):
            group = rows[start:start + 4]
            if len(group) != 4 or len({(row["task_id"], row["batch"]) for row in group}) != 1:
                raise ValueError("invalid_saved_smoke_group")
            groups.append(group)
    else:
        for task_id in sorted({row["task_id"] for row in rows}):
            group = [row for row in rows if row["task_id"] == task_id]
            if len(group) != 4 or {row["trial"] for row in group} != set(range(4)):
                raise ValueError("invalid_saved_pilot_group")
            groups.append(group)
    return {"groups": len(groups), "mixed_groups": sum(len({row["reward"] for row in group}) > 1 for group in groups)}


def run(smoke_run: Path, pilot: Path, manifest: Path, output: Path):
    smoke_run, pilot, manifest, output = (path.resolve() for path in (smoke_run, pilot, manifest, output))
    if output.exists() or any(output == source or output.is_relative_to(source) or source.is_relative_to(output)
                              for source in (smoke_run, pilot)):
        raise ValueError("new_non_nested_output_required; no overwrites")
    training = smoke_run / "training"
    preflight = json.loads((training / "preflight.json").read_text(encoding="utf-8"))
    protocol = json.loads((pilot / "protocol.json").read_text(encoding="utf-8"))
    if any(item["manifest_sha256"] != sha256_file(manifest) for item in (preflight, protocol)):
        raise ValueError("manifest_hash_mismatch")
    if protocol.get("split") != "train" or set(protocol["task_ids"]) != set(preflight["smoke_task_ids"]):
        raise ValueError("reviewed_train_task_set_mismatch")
    if any(item["verifier_version"] != LEGACY_VERIFIER_VERSION for item in (preflight, protocol)):
        raise ValueError("this_replay_requires_original_v11_evidence")
    if sha256_file(ROOT / "src/ecommerce_customer_service/tools.py") != preflight["source_sha256"]["src/ecommerce_customer_service/tools.py"]:
        raise ValueError("business_tools_changed; cannot claim identical replay")
    source_files = [training / "preflight.json", training / "online_rewards.jsonl",
                    pilot / "protocol.json", pilot / "sft/rollouts.jsonl",
                    pilot / "summary.json", pilot / "readiness.json", pilot / "reward_review.json"]
    hashes_before = {str(path): sha256_file(path) for path in source_files}
    tasks = {task.task_id: (path, task) for path in load_task_paths(manifest, split="train")
             for task in (TaskSpec.from_file(path),) if task.task_id in protocol["task_ids"]}
    if set(tasks) != set(protocol["task_ids"]):
        raise ValueError("task_set_missing_from_manifest")
    database_hashes = {task_id: sha256_file(task.database_path) for task_id, (_, task) in tasks.items()}
    if database_hashes != protocol["source_database_sha256"]:
        raise ValueError("original_database_hash_mismatch")
    original = {"smoke": load_jsonl(training / "online_rewards.jsonl"), "pilot": load_jsonl(pilot / "sft/rollouts.jsonl")}
    derived, reports, changes = {}, {}, []
    for stage, rows in original.items():
        current = []
        for index, row in enumerate(rows, 1):
            if row["task_id"] not in tasks or row.get("split") != "train" or row.get("verifier_version") != LEGACY_VERIFIER_VERSION:
                raise ValueError(f"invalid_train_source_row:{stage}:{index}")
            env = EcommerceCustomerServiceEnv(task_path=tasks[row["task_id"]][0])
            try:
                env.reset()
                for call in row["tool_executions"]:
                    if env._tools.execute(call["name"], call["arguments"]).to_dict() != call:
                        raise ValueError(f"tool_replay_mismatch:{stage}:{index}")
                final = _final_response(row["completion"]) if stage == "smoke" else row["final_response"]
                old_reward = env._evaluate(final, verifier_version=LEGACY_VERIFIER_VERSION)
                old_verifier = json.loads(json.dumps(env._last_verification.to_dict()))
                if old_reward != row["reward"] or old_verifier != row["verifier"]:
                    raise ValueError(f"legacy_reward_replay_mismatch:{stage}:{index}")
                reward = env._evaluate(final)
                verifier = json.loads(json.dumps(env._last_verification.to_dict()))
                for field in ("state_passed", "required_tools_passed"):
                    if verifier[field] != old_verifier[field]:
                        raise ValueError(f"immutable_component_changed:{stage}:{index}:{field}")
                new = deepcopy(row)
                new.update(reward=reward, verifier=verifier, verifier_version=VERIFIER_VERSION,
                           reward_components=env._last_breakdown.to_dict(), final_response=final,
                           rescore_metadata={"method": "isolated_cpu_tool_replay", "source_verifier_version": LEGACY_VERIFIER_VERSION,
                               "source_reward": row["reward"], "original_training_reward_unchanged": True})
                current.append(new)
                if verifier != old_verifier or reward != old_reward:
                    changes.append({"stage": stage, "row": index, "task_id": row["task_id"],
                        "reward_before": old_reward, "reward_after": reward,
                        "removed_failures": sorted(set(old_verifier["failures"]) - set(verifier["failures"])),
                        "added_failures": sorted(set(verifier["failures"]) - set(old_verifier["failures"])), "final_response": final})
            finally:
                env._cleanup()
        reports[stage] = {
            "rollouts": len(rows), "successes_before": sum(row["reward"] == 1 for row in rows),
            "successes_after": sum(row["reward"] == 1 for row in current),
            "groups_before": group_stats(rows, smoke=stage == "smoke"),
            "groups_after": group_stats(current, smoke=stage == "smoke"),
            "zero_to_one_rows": [item["row"] for item in changes if item["stage"] == stage and item["reward_before"] == 0 and item["reward_after"] == 1],
            "one_to_zero_rows": [item["row"] for item in changes if item["stage"] == stage and item["reward_before"] == 1 and item["reward_after"] == 0],
            "failure_counts": dict(Counter(failure for row in current for failure in row["verifier"]["failures"])),
        }
        derived[stage] = current
    hashes_after = {str(path): sha256_file(path) for path in source_files}
    if hashes_before != hashes_after or database_hashes != {key: sha256_file(task.database_path) for key, (_, task) in tasks.items()}:
        raise RuntimeError("source_modified_during_cpu_replay")
    report = {"complete": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_verifier_version": LEGACY_VERIFIER_VERSION, "verifier_version": VERIFIER_VERSION,
        "method": "isolated_cpu_tool_replay", "legacy_reward_replay_matches": True,
        "source_files_unchanged": True, "source_database_sha256": database_hashes, "source_files_sha256": hashes_after,
        "scoring_source_sha256": {name: sha256_file(ROOT / name) for name in SCORING_SOURCES},
        "stages": reports, "changed_rows": len(changes), "model_rerun": False, "training_rewards_rewritten": False,
        "formal_rl_gate_passed": False, "new_runtime_gpu_validated": False,
        "interpretation": "Offline correction of saved train rewards, not model improvement, new sampling, held-out evaluation or permission for long training."}
    output.mkdir(parents=True, exist_ok=False)
    for stage, rows in derived.items():
        write_jsonl(output / stage / "rollouts.jsonl", rows)
    write_jsonl(output / "changes.jsonl", changes)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("smoke-run", "pilot", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = run(args.smoke_run, args.pilot, args.manifest, args.output)
    print(json.dumps({key: report[key] for key in ("complete", "verifier_version", "legacy_reward_replay_matches",
        "source_files_unchanged", "stages", "formal_rl_gate_passed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
