"""Export a human review packet, or record a human's explicit all-rows confirmation.

Do not invoke --confirm-reviewed on behalf of the user without explicit approval
of the attached trajectory-level review. Preparing the packet does NOT sign it.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.grpo_experiment_v13 import read_json, read_rows
from ecommerce_customer_service.local_hf import sha256_file
from ecommerce_customer_service.verifier import V13_VERIFIER_VERSION


def prepare(evidence, output, *, reviewer=None, confirm_reviewed=False):
    evidence, output = evidence.resolve(), output.resolve()
    if output.exists() or evidence == output or evidence in output.parents:
        raise ValueError("new_review_output_outside_original_evidence_required")
    protocol = read_json(evidence / "protocol.json")
    original_review = read_json(evidence / "reward_review.json")
    rollouts = evidence / "sft/rollouts.jsonl"
    rows = read_rows(rollouts)
    if (protocol.get("verifier_version") != V13_VERIFIER_VERSION
            or original_review.get("rollouts_sha256") != sha256_file(rollouts)):
        raise ValueError("v13_review_rollout_hash_mismatch")
    verdicts = {(row["task_id"], row["trial"]): row for row in original_review["rows"]}
    if len(verdicts) != len(rows):
        raise ValueError("incomplete_prior_review")
    reviewed = []
    markdown = ["# v13 奖励逐条审阅", "", "这是 train 轨迹审阅，不是 dev/test 结果，也不代表通过正式 RL gate。",
                "", "请检查实际工具调用和最终回答是否支持奖励。仅在认可全部逐条判定后，才执行单独的确认命令。",
                "", f"原始轨迹 SHA256：`{sha256_file(rollouts)}`", ""]
    for index, row in enumerate(rows, 1):
        verdict = deepcopy(verdicts[row["task_id"], row["trial"]])
        if verdict["expected_reward"] != row["reward"] or not verdict.get("evidence"):
            raise ValueError("prior_audit_disputed_or_incomplete")
        verdict.update(status="confirmed" if confirm_reviewed else "pending_human_review", recorded_reward=row["reward"])
        reviewed.append(verdict)
        markdown.extend([f"## {index}. {row['task_id']} / trial {row['trial']} / reward {row['reward']}", "",
            "审计依据：" + verdict["evidence"], "", "失败项：`" + json.dumps(row["verifier"]["failures"], ensure_ascii=False) + "`", "",
            "最终回答：", "", row["final_response"], "", "实际工具执行：", "", "```json",
            json.dumps(row["tool_executions"], ensure_ascii=False, indent=2), "```", ""])
    if confirm_reviewed and (not reviewer or "codex" in reviewer.lower() or "agent" in reviewer.lower()):
        raise ValueError("named_human_reviewer_required")
    output.mkdir(parents=True, exist_ok=False)
    review = {"reviewer": reviewer if confirm_reviewed else None,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat() if confirm_reviewed else None,
        "human_signoff": confirm_reviewed, "verifier_version": V13_VERIFIER_VERSION,
        "rollouts_sha256": sha256_file(rollouts), "rows": reviewed,
        "formal_rl_gate_passed": False, "scope": "bounded_v13_train_template_experiment"}
    (output / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "逐条奖励审阅.md").write_text("\n".join(markdown), encoding="utf-8")
    return {"review": str(output / "review.json"), "packet": str(output / "逐条奖励审阅.md"),
            "human_signoff": confirm_reviewed, "gpu_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer")
    parser.add_argument("--confirm-reviewed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.evidence, args.output, reviewer=args.reviewer,
                             confirm_reviewed=args.confirm_reviewed), ensure_ascii=False, indent=2))
