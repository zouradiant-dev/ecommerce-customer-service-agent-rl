"""CPU guards for a bounded v13 experiment. Never certifies the formal RL gate."""
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from .experiment_prompt import EXPERIMENT_SYSTEM_PROMPT, PROMPT_VERSION
from .grpo_readiness import sampling_checks, select_train_tasks
from .local_hf import sha256_file
from .models import TaskSpec
from .online_env import EcommerceCustomerServiceEnv, _final_response
from .verifier import V13_VERIFIER_VERSION
from .grpo_memory_v13 import MEMORY_POLICY, summarize_memory

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SCORING_FILES = (
    "src/ecommerce_customer_service/verifier.py",
    "src/ecommerce_customer_service/communication.py",
    "src/ecommerce_customer_service/communication_facts.py",
    "src/ecommerce_customer_service/communication_facts_v13.py",
    "src/ecommerce_customer_service/tools.py",
    "src/ecommerce_customer_service/rewards.py",
)
RUNTIME_FILES = SCORING_FILES + (
    "src/ecommerce_customer_service/online_env.py",
    "src/ecommerce_customer_service/grpo_experiment_v13.py",
    "src/ecommerce_customer_service/grpo_memory_v13.py",
    "src/ecommerce_customer_service/experiment_prompt.py",
    "src/ecommerce_customer_service/grpo_readiness.py",
    "src/ecommerce_customer_service/models.py",
    "src/ecommerce_customer_service/policies.py",
    "src/ecommerce_customer_service/training.py",
    "src/ecommerce_customer_service/simulator.py",
    "scripts/train_grpo_smoke_v12_r1.py",
    "scripts/audit_grpo_smoke_result.py",
    "scripts/train_grpo_experiment_v13.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_hashes(root):
    return {relative: sha256_file(root / relative) for relative in RUNTIME_FILES}


def validate_review(review_path, rows, rollout_path):
    review = read_json(review_path)
    if (review.get("human_signoff") is not True or not review.get("reviewer")
            or not review.get("reviewed_at_utc")):
        raise ValueError("human_reward_review_required; agent replay is not human signoff")
    timestamp = datetime.fromisoformat(review["reviewed_at_utc"].replace("Z", "+00:00"))
    if timestamp.utcoffset() is None:
        raise ValueError("review_timestamp_must_include_timezone")
    if (review.get("verifier_version") != V13_VERIFIER_VERSION
            or review.get("rollouts_sha256") != sha256_file(rollout_path)):
        raise ValueError("review_version_or_rollout_hash_mismatch")
    keys = {(row["task_id"], row["trial"]): row["reward"] for row in rows}
    verdicts = review.get("rows") or []
    reviewed = {(row.get("task_id"), row.get("trial")): row for row in verdicts}
    if len(keys) != len(rows) or len(verdicts) != len(keys) or set(reviewed) != set(keys):
        raise ValueError("all_positive_and_negative_rows_require_review")
    for key, reward in keys.items():
        row = reviewed[key]
        if (row.get("status") != "confirmed" or row.get("recorded_reward") != reward
                or row.get("expected_reward") != reward or not row.get("evidence")):
            raise ValueError(f"reward_review_pending_or_disputed:{key}")
    return {"review_sha256": sha256_file(review_path), "reviewer": review["reviewer"],
            "reviewed_at_utc": review["reviewed_at_utc"], "human_review_confirmed": True}


def replay_row(row, task_path):
    """Replay actual tools on a fresh DB, including authorization and state assertions."""
    env = EcommerceCustomerServiceEnv(task_path=task_path, verifier_version=V13_VERIFIER_VERSION)
    try:
        env.reset()
        for call in row["tool_executions"]:
            if env._tools.execute(call["name"], call["arguments"]).to_dict() != call:
                raise ValueError(f"tool_replay_mismatch:{row['task_id']}:{call['name']}")
        final = row.get("final_response") if "final_response" in row else _final_response(row["completion"])
        reward = env._evaluate(final)
        verification = json.loads(json.dumps(env._last_verification.to_dict()))
        if reward != row["reward"] or verification != row["verifier"]:
            raise ValueError(f"reward_or_state_replay_mismatch:{row['task_id']}")
    finally:
        env._cleanup()


def validate_evidence(directory, review_path, *, root, manifest, adapter_sha256):
    protocol = read_json(directory / "protocol.json")
    report = read_json(directory / "rescore_report.json")
    rows_path = directory / "sft/rollouts.jsonl"
    rows = read_rows(rows_path)
    if not report.get("complete") or protocol.get("split") != "train":
        raise ValueError("complete_train_evidence_required")
    if (protocol.get("verifier_version") != V13_VERIFIER_VERSION
            or protocol.get("adapter_sha256") != adapter_sha256
            or protocol.get("manifest_sha256") != sha256_file(manifest)):
        raise ValueError("evidence_reward_weights_or_manifest_mismatch")
    if (protocol.get("prompt_version") != PROMPT_VERSION
            or protocol.get("prompt_sha256") != sha256(EXPERIMENT_SYSTEM_PROMPT.encode()).hexdigest()
            or protocol.get("tool_surface") != "native_dynamic"):
        raise ValueError("unexpected_evidence_generation_prompt")
    for relative in SCORING_FILES:
        current = sha256_file(root / relative)
        if current != protocol["source_sha256"].get(relative) or current != report["scoring_source_sha256"].get(relative):
            raise ValueError("evidence_scoring_source_changed:" + relative)
    if (any(row.get("verifier_version") != V13_VERIFIER_VERSION or row.get("synthetic_fixture_not_readiness_evidence")
            for row in rows) or len(rows) != protocol.get("rollouts")
            or sum(row["reward"] for row in rows) != report.get("passed_after")):
        raise ValueError("evidence_rows_incomplete_wrong_version_or_synthetic")
    fresh = sampling_checks(rows, trials_per_task=protocol["trials_per_task"], expected_task_ids=protocol["task_ids"])
    if not fresh["engineering_sampling_checks_passed"] or fresh != read_json(directory / "readiness.json"):
        raise ValueError("engineering_sampling_checks_failed_or_stale")
    templates = tuple(sorted(protocol["template_ids"]))
    tasks = select_train_tasks(manifest, templates=templates, per_template=10)
    paths = {task.task_id: path for path, task in tasks}
    if not set(protocol["task_ids"]).issubset(paths):
        raise ValueError("evidence_tasks_not_inline_train")
    for row in rows:
        path = paths[row["task_id"]]
        database = TaskSpec.from_file(path).database_path
        if sha256_file(database) != protocol["source_database_sha256"].get(row["task_id"]):
            raise ValueError("evidence_source_database_changed")
        replay_row(row, path)
        if sha256_file(database) != protocol["source_database_sha256"][row["task_id"]]:
            raise RuntimeError("source_database_mutated_during_replay")
    review = validate_review(review_path, rows, rows_path)
    return {"protocol_sha256": sha256_file(directory / "protocol.json"),
            "rollouts_sha256": sha256_file(rows_path), "templates": list(templates),
            "evidence_task_ids": sorted(protocol["task_ids"]), **review,
            "native_to_static_runtime_gap": True, "formal_rl_gate_passed": False}, tasks


def select_experiment_tasks(tasks, evidence_task_ids, phase):
    selected = tasks if phase == "experiment" else [item for item in tasks if item[1].task_id in evidence_task_ids]
    if phase not in {"qualification", "experiment"} or not selected:
        raise ValueError("invalid_or_empty_experiment_selection")
    if phase == "experiment" and (len(selected) < 20 or min(Counter(task.scenario for _, task in selected).values()) < 4):
        raise ValueError("larger_experiment_requires_broad_balanced_train_selection")
    return selected


def validate_qualification(directory, *, context, paths, root):
    """A saved JSON flag alone cannot authorize the 100-step phase."""
    preflight = read_json(directory / "preflight.json")
    result = read_json(directory / "complete.json")
    expected = {**context, "phase": "qualification", "max_steps": 10}
    if any(preflight.get(key) != value for key, value in expected.items()):
        raise ValueError("qualification_context_mismatch")
    if (result.get("complete") is not True or result.get("phase") != "qualification"
            or result.get("optimizer_steps") != 10 or result.get("nonzero_gradient_steps", 0) < 1
            or any(result.get(key) is not True for key in (
                "actual_policy_update_verified", "adapter_save_reload_verified", "sft_reference_unchanged",
                "source_databases_unchanged")) or result.get("memory_headroom_gib", 0) < 1.0
            or result.get("memory_policy") != MEMORY_POLICY):
        raise ValueError("qualification_not_successful_or_insufficient_memory_headroom")
    for name, expected_hash in result["artifacts_sha256"].items():
        if name not in {"preflight.json", "online_rewards.jsonl", "gradient_audit.jsonl", "adapter_model.safetensors", "initialization.json", "ref/adapter_model.safetensors", "memory_audit.jsonl", "memory_report.json"}:
            raise ValueError("unexpected_qualification_artifact")
        if sha256_file(directory / name) != expected_hash:
            raise ValueError("qualification_artifact_changed:" + name)
    if set(result["artifacts_sha256"]) != {"preflight.json", "online_rewards.jsonl", "gradient_audit.jsonl", "adapter_model.safetensors", "initialization.json", "ref/adapter_model.safetensors", "memory_audit.jsonl", "memory_report.json"}:
        raise ValueError("missing_qualification_artifact_hash")
    memory = summarize_memory(read_rows(directory / "memory_audit.jsonl"), 10)
    if (read_json(directory / "memory_report.json") != memory or not memory["memory_safety_passed"]
            or result["memory_headroom_gib"] != memory["memory_headroom_gib"]
            or result.get("peak_allocated_gib") != memory["peak_allocated_gib"]
            or result.get("peak_reserved_gib") != memory["peak_reserved_gib"]):
        raise ValueError("qualification_memory_evidence_mismatch_or_insufficient")
    gradients = read_rows(directory / "gradient_audit.jsonl")
    if len(gradients) != 10 or [row["next_optimizer_step"] for row in gradients] != list(range(1, 11)):
        raise ValueError("qualification_gradient_events_incomplete")
    if any(row.get("finite") is not True for row in gradients):
        raise ValueError("qualification_nonfinite_gradient")
    rows = read_rows(directory / "online_rewards.jsonl")
    if len(rows) != 80 or any(row.get("verifier_version") != V13_VERIFIER_VERSION or row.get("split") != "train" for row in rows):
        raise ValueError("qualification_online_rows_incomplete_or_wrong_version")
    for batch in range(1, 11):
        batch_rows = [row for row in rows if row.get("batch") == batch]
        if len(batch_rows) != 8 or any(len({r["task_id"] for r in batch_rows[start:start + 4]}) != 1 for start in (0, 4)):
            raise ValueError("qualification_reward_groups_incomplete")
    for row in rows:
        if row["task_id"] not in preflight["task_ids"] or row["task_id"] not in paths:
            raise ValueError("qualification_unselected_task")
        replay_row(row, paths[row["task_id"]])
    if preflight["source_sha256"] != source_hashes(root):
        raise ValueError("qualification_training_sources_changed")
    return {"directory": str(directory.resolve()), "complete_sha256": sha256_file(directory / "complete.json"),
            "online_rollouts_replayed": len(rows), "formal_rl_gate_passed": False}
