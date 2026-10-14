"""Frozen inputs and artifact identities for a small online-protocol SFT experiment."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from ..training import validate_training_row
from .artifacts import BASE_REVISION, FILE_HASHES, file_sha256, verify_adapter
from .policy import get_policy

REPAIR_VERSION = "online_repair_pilot_v1"
CONFIG = {"schema_version": REPAIR_VERSION, "seed": 20261013,
    "training": {"smoke_steps":4,"pilot_steps":64,"batch_size":1,"gradient_accumulation":8,
        "learning_rate":2e-5,"warmup_ratio":0.1,"max_sequence_length":2304,
        "load_in_4bit":True,"model_dtype":"bfloat16","trainable_parameters_fp32":True,
        "defer_gradient_accumulation_scaling":True,"gradient_checkpointing_reentrant":False},
    "evaluation": {"policy_profile":"binding_compact_v3","precision":"bf16",
        "context_budget":8192,"recent_turns":6,"temperature":0,"max_new_tokens":512,
        "formal_test_allowed":False,"automatic_promotion":False},
    "limitations":["small_synthetic_repair_dataset","single_seed_single_greedy_trial",
        "online_regression_is_not_the_old_nine_tool_offline_dev","dev_loss_is_not_task_success"]}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_bundle(data: Path, original_adapter: Path | None = None) -> dict:
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version")!="online_repair_sft_v1" or manifest.get("test_sft_rows")!=0:
        raise ValueError("unexpected_repair_dataset")
    if manifest["policy"]!=get_policy("binding_compact_v3").metadata():
        raise ValueError("repair_policy_mismatch")
    for name in ("train.jsonl","dev.jsonl"):
        if file_sha256(data/name)!=manifest["files_sha256"][name]:
            raise ValueError("repair_data_hash_mismatch:"+name)
    result={}
    tasks={}
    for split, expected in (("train",512),("dev",128)):
        samples=rows(data/(split+".jsonl"))
        if len(samples)!=expected: raise ValueError("repair_row_count_mismatch")
        seen=set()
        for row in samples:
            if row.get("split")!=split or row.get("real_customer_data") is not False or row.get("teacher_is_llm") is not False:
                raise ValueError("wrong_repair_split_or_provenance")
            if row["id"] in seen: raise ValueError("duplicate_repair_row")
            seen.add(row["id"])
            if validate_training_row(row): raise ValueError("invalid_repair_training_row")
            stats=row.get("validation",{})
            if (stats.get("reference_business_state_verified") is not True
                    or stats.get("native_tool_format_verified") is not True
                    or not 0<stats.get("input_tokens",0)<=CONFIG["training"]["max_sequence_length"]):
                raise ValueError("unverified_or_overlength_repair_row")
        tasks[split]={row["task_id"] for row in samples}
        result[split]={"rows":len(samples),"episodes":len(tasks[split]),"categories":dict(Counter(row["category"] for row in samples))}
    if tasks["train"] & tasks["dev"]: raise ValueError("repair_task_leakage")
    if any(path.name.startswith("test") for path in data.iterdir()):
        raise ValueError("training_bundle_must_exclude_sealed_test_files")
    tokenizer_hashes=manifest["tokenizer_files_sha256"]
    if any(tokenizer_hashes[name]!=FILE_HASHES[name] for name in tokenizer_hashes):
        raise ValueError("repair_tokenizer_identity_mismatch")
    if original_adapter: verify_adapter(original_adapter)
    return {"schema_version":REPAIR_VERSION,"data_valid":True,"splits":result,"test_files_present":False,
            "files_sha256":{name:file_sha256(data/name) for name in ("train.jsonl","dev.jsonl","manifest.json")}}


def training_arguments(data: Path, base: Path, adapter: Path, output: Path, *, smoke: bool) -> list[str]:
    if output.resolve()==adapter.resolve() or output.resolve().is_relative_to(adapter.resolve()):
        raise ValueError("cannot_write_inside_initial_adapter")
    train=CONFIG["training"]
    return ["--train",str(data/"train.jsonl"),"--dev",str(data/"dev.jsonl"),
        "--base-model",str(base),"--model-revision",BASE_REVISION,"--tokenizer-path",str(adapter),"--local-files-only",
        "--init-adapter",str(adapter),"--output",str(output),
        "--max-steps",str(train["smoke_steps"] if smoke else train["pilot_steps"]),
        "--batch-size","1","--gradient-accumulation","8","--learning-rate",str(train["learning_rate"]),
        "--warmup-ratio","0.1","--max-sequence-length",str(train["max_sequence_length"]),"--overlength","skip",
        "--eval-every","4" if smoke else "32","--save-every","4" if smoke else "32",
        "--max-dev-examples","16" if smoke else "128","--seed",str(CONFIG["seed"]),
        "--load-in-4bit","--bf16","--model-dtype","bfloat16","--trainable-params-fp32",
        "--defer-gradient-accumulation-scaling","--no-gradient-checkpointing-reentrant","--diagnose-gradients-until","32"]


def parameter_delta(original: Path, candidate: Path) -> dict:
    """Check actual FP32 tensor changes, not serialization-dependent file hashes."""
    import numpy as np
    from safetensors import safe_open
    changed=0;elements=0;squared=0.0;maximum=0.0
    with safe_open(original,framework="np") as before, safe_open(candidate,framework="np") as after:
        names=set(before.keys())
        if not names or names!=set(after.keys()): raise ValueError("repair_adapter_tensor_keys_changed")
        for name in sorted(names):
            left,right=before.get_tensor(name),after.get_tensor(name)
            if left.shape!=right.shape or left.dtype!=np.float32 or right.dtype!=np.float32:
                raise ValueError("repair_adapter_shape_or_dtype_changed")
            if not np.isfinite(left).all() or not np.isfinite(right).all(): raise ValueError("repair_adapter_nonfinite_tensor")
            difference=right.astype(np.float64)-left.astype(np.float64)
            changed+=int(np.any(difference));elements+=int(left.size)
            squared+=float(np.sum(difference*difference));maximum=max(maximum,float(np.max(np.abs(difference))))
    if not changed: raise ValueError("repair_adapter_tensors_did_not_change")
    return {"changed_tensors":changed,"tensors_checked":len(names),"elements_checked":elements,
            "delta_l2_norm":squared**0.5,"delta_max_abs":maximum,"all_tensors_finite":True}


def compare_runs(baseline: Path, candidate: Path) -> dict:
    before=json.loads((baseline/"summary.json").read_text(encoding="utf-8"))
    after=json.loads((candidate/"summary.json").read_text(encoding="utf-8"))
    if not before["complete"] or not after["complete"]: raise ValueError("incomplete_repair_comparison")
    if any(run.get("real_model_execution") is not True for run in (before,after)):
        raise ValueError("repair_comparison_requires_real_model")
    controls=("case_sha256","case_ids","context_counter","context_budget","context_policy","recent_turns",
              "system_prompt_sha256","tools_sha256","source_sha256")
    if any(before[key]!=after[key] for key in controls): raise ValueError("nonpaired_repair_protocol")
    if before.get('knowledge_index_sha256') != after.get('knowledge_index_sha256'):
        raise ValueError('nonpaired_knowledge_index')
    if before.get('cancel_reply_check_version', 'legacy_v1') != after.get('cancel_reply_check_version', 'legacy_v1'):
        raise ValueError('nonpaired_cancel_reply_check')
    old=json.loads((baseline/"artifact.json").read_text(encoding="utf-8"))
    new=json.loads((candidate/"artifact.json").read_text(encoding="utf-8"))
    if any(old[key]!=new[key] for key in ("base_revision","precision","template_sha256","vocab_sha256","generation")):
        raise ValueError("nonpaired_repair_artifacts")
    if old["adapter_sha256"]==new["adapter_sha256"]: raise ValueError("repair_weights_did_not_change")
    outcomes=lambda folder:{path.parent.name:json.loads(path.read_text(encoding="utf-8")) for path in folder.glob("cases/*/result.json")}
    left,right=outcomes(baseline),outcomes(candidate)
    if set(left)!=set(right) or set(left)!=set(before["case_ids"]): raise ValueError("missing_paired_repair_cases")
    improved=[name for name in left if not left[name]["passed"] and right[name]["passed"]]
    regressed=[name for name in left if left[name]["passed"] and not right[name]["passed"]]
    categories=sorted({row["category"] for row in left.values()})
    return {"schema_version":REPAIR_VERSION,"comparison_kind":"paired_synthetic_online_dev",
        "baseline_passed":before["passed_cases"],"candidate_passed":after["passed_cases"],"cases":len(left),
        "baseline_pass_rate":before["case_pass_rate"],"candidate_pass_rate":after["case_pass_rate"],
        "improved_cases":improved,"regressed_cases":regressed,
        "baseline_output_failures":before["model_output_failures"],"candidate_output_failures":after["model_output_failures"],
        "baseline_by_group":before.get("by_regression_group",{}),"candidate_by_group":after.get("by_regression_group",{}),
        "by_category":{kind:{"cases":sum(row["category"]==kind for row in left.values()),
            "baseline_passed":sum(row["category"]==kind and row["passed"] for row in left.values()),
            "candidate_passed":sum(row["category"]==kind and row["passed"] for row in right.values())} for kind in categories},
        "automatic_promotion":False,"formal_test_performed":False,"old_offline_nine_tool_regression_performed":False,
        "limitations":CONFIG["limitations"]}
