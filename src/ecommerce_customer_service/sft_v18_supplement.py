"""Build and validate a train-only v18 corrective SFT supplement.

The supplement is generated from fresh SQLite states.  Frozen v18 dev rows are
used only for loss monitoring and never copied into the training split.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any

from .data import load_jsonl
from .database import create_database
from .experiment_prompt_v18 import V18_SYSTEM_PROMPT
from .models import TaskSpec
from .reference_v18 import build_reference_sft_v18
from .sft_v18_48gb import sha256_file
from .tasks import _task_for_template
from .tasks_v18 import _context, upgrade_task_v18
from .training import validate_training_row


SUPPLEMENT_VERSION = "ecommerce_cs_v18_supplement_r1"
FOCUS_SCENARIOS = ("contradictory_order", "cross_order_item_mismatch")

_FOCUS_PROMPTS: dict[str, tuple[str, ...]] = {
    "contradictory_order": (
        "我想处理一笔订单，但现在记不清准确编号。请先列出我的订单；即使列表里只有一笔，也要等我明确确认订单号，这一轮不要替我确认或修改任何状态。",
        "我前后说的目标订单有冲突，可能是 {order_id}，也可能是 {conflict_order_id}。请只列出本人订单并让我确认准确编号，不要把查询结果当成我的确认。",
        "先帮我查看本人有哪些订单。我还没有确认要处理哪一笔，所以看到最近订单或唯一订单后也不能自行选定，请等我回复准确订单号。",
        "我可能把订单号说错了。这次只做订单列表查询，并明确要求我确认目标订单；不要查询某一单的详情，也不要执行写操作。",
        "目标订单存在歧义，不能按最近、金额或状态替我猜。请展示候选订单，然后停下来等我确认准确订单号。",
        "我需要先核对自己的订单列表，暂时没有授权处理任何一单。请列单后要求我确认，不要把‘查到了订单’说成‘目标已确认’。",
    ),
    "cross_order_item_mismatch": (
        "我要核对订单 {order_id} 和商品项 {mismatched_item_id}。商品项可能属于另一笔订单；若绑定不一致就立即停止，不能改用该订单里的其他商品项，也不能创建退款。",
        "订单号是 {order_id}，我给的商品项是 {mismatched_item_id}。请按这两个原始标识检查归属；串单时只说明不匹配，不要替换参数或执行退款。",
        "我确认想退款，但前提是 {mismatched_item_id} 确实属于 {order_id}。请先核对绑定；如果不是同一订单，即使我已确认也不得调用退款写操作。",
        "请检查 {order_id} 中是否真的包含 {mismatched_item_id}。如果订单详情里出现的是另一个商品项，不能拿那个商品项顶替我给的编号，必须安全停止。",
        "核验订单 {order_id} 与商品项 {mismatched_item_id} 的对应关系。政策查询若返回商品项不属于已检查订单，就报告串单并结束，禁止继续创建退款。",
        "帮我处理 {order_id} 的 {mismatched_item_id}，但不要猜测或自动纠正标识。两者不属于同一订单时请拒绝退款，并说明没有执行写入。",
    ),
}


def _identity_prefix(task: TaskSpec, variant: int) -> str:
    values = _context(task)
    choices = (
        "账户核验信息：姓名{name}，手机号后四位{phone_last4}。",
        "请先验证客户：{name}，联系电话尾号{phone_last4}。",
        "我是{name}，手机最后四位是{phone_last4}。",
        "身份信息是{name}，手机号后四位{phone_last4}。",
    )
    return choices[variant % len(choices)].format(**values)


def _focus_prompt(task: TaskSpec, variant: int) -> tuple[str, str | None]:
    values = _context(task)
    conflict_order_id: str | None = None
    if task.scenario == "contradictory_order":
        digest = sha256(f"{task.task_id}|conflict".encode()).hexdigest()[:10]
        conflict_order_id = f"ord_conflict_{digest}"
        values["conflict_order_id"] = conflict_order_id
    template = _FOCUS_PROMPTS[task.scenario][variant % len(_FOCUS_PROMPTS[task.scenario])]
    return _identity_prefix(task, variant) + template.format(**values), conflict_order_id


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _user_prompts(rows: list[dict[str, Any]]) -> set[str]:
    prompts: set[str] = set()
    for row in rows:
        for message in row.get("messages") or []:
            if message.get("role") == "user":
                prompts.add(str(message.get("content") or ""))
    return prompts


def _sample_replay(
    rows: list[dict[str, Any]], *, per_scenario: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scenario") or "")].append(row)
    selected: list[dict[str, Any]] = []
    for scenario in sorted(grouped):
        candidates = list(grouped[scenario])
        if len(candidates) < per_scenario:
            raise ValueError(f"insufficient_replay_rows:{scenario}:{len(candidates)}:{per_scenario}")
        rng = random.Random(f"{SUPPLEMENT_VERSION}|{seed}|{scenario}")
        chosen = rng.sample(candidates, per_scenario)
        for row in chosen:
            copied = dict(row)
            copied["messages"] = [dict(message) for message in row.get("messages") or []]
            if not copied["messages"] or copied["messages"][0].get("role") != "system":
                raise ValueError(f"replay_system_message_missing:{row.get('id')}")
            copied["messages"][0]["content"] = V18_SYSTEM_PROMPT
            copied["metadata"] = {**dict(row.get("metadata") or {}), "supplement_role": "replay"}
            selected.append(copied)
    return selected


def build_v18_supplement(
    output_dir: Path,
    *,
    original_train: Path,
    original_dev: Path,
    focus_tasks_per_scenario: int = 160,
    replay_rows_per_scenario: int = 20,
    seed: int = 1818,
) -> dict[str, Any]:
    if focus_tasks_per_scenario < 1 or replay_rows_per_scenario < 1:
        raise ValueError("supplement counts must be positive")
    output_dir = output_dir.resolve()
    original_train = original_train.resolve()
    original_dev = original_dev.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"supplement output must be empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(FOCUS_SCENARIOS):
        for local_index in range(focus_tasks_per_scenario):
            replica = 2_000_000 + scenario_index * 100_000 + local_index
            task_id = f"v18_train_supplement_{scenario}_{local_index:04d}"
            task_dir = output_dir / "tasks" / task_id
            database_path = create_database(task_dir / "state.sqlite", overwrite=False)
            source = _task_for_template(scenario, replica, database_path, seed=seed)
            upgraded = upgrade_task_v18(
                source,
                task_id=task_id,
                split="train",
                prompt_family="train_direct_r1",
                seed=seed,
            )
            prompt, conflict_order_id = _focus_prompt(upgraded, local_index)
            metadata = {
                **dict(upgraded.metadata),
                "dataset_version": SUPPLEMENT_VERSION,
                "prompt_family_v18": "train_supplement_contrast_r1",
                "state_family_v18": "train_supplement_state_namespace_r1",
                "supplement_role": "focus",
                "supplement_prompt_variant": local_index % len(_FOCUS_PROMPTS[scenario]),
            }
            if conflict_order_id:
                metadata["conflicting_candidate_order_id"] = conflict_order_id
            task = replace(upgraded, user_prompt=prompt, metadata=metadata)
            task_path = task_dir / "task.json"
            task_path.write_text(
                json.dumps(task.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            row = task.to_dict(relative_database=f"tasks/{task_id}/state.sqlite")
            row["task_file"] = f"tasks/{task_id}/task.json"
            row["database_sha256"] = sha256_file(database_path)
            row["task_file_sha256"] = sha256_file(task_path)
            task_rows.append(row)

    task_rows.sort(key=lambda row: str(row["task_id"]))
    manifest = output_dir / "tasks.jsonl"
    _write_jsonl(manifest, task_rows)
    reference_dir = output_dir / "reference-focus"
    reference_summary = build_reference_sft_v18(manifest, reference_dir)
    focus_rows = load_jsonl(reference_dir / "train.jsonl")
    for row in focus_rows:
        row["metadata"] = {**dict(row.get("metadata") or {}), "supplement_role": "focus"}

    source_train_rows = load_jsonl(original_train)
    source_dev_rows = load_jsonl(original_dev)
    replay_rows = _sample_replay(
        source_train_rows,
        per_scenario=replay_rows_per_scenario,
        seed=seed,
    )
    train_rows = focus_rows + replay_rows
    random.Random(seed).shuffle(train_rows)

    failures: list[str] = []
    ids: set[str] = set()
    for row in train_rows:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in ids:
            failures.append(f"duplicate_or_missing_id:{row_id}")
        ids.add(row_id)
        metadata = dict(row.get("metadata") or {})
        if metadata.get("split") != "train" or metadata.get("test_material") is not False:
            failures.append(f"non_train_material:{row_id}")
        failures.extend(f"{row_id}:{failure}" for failure in validate_training_row(row))
    prompt_overlap = _user_prompts(train_rows) & _user_prompts(source_dev_rows)
    if prompt_overlap:
        failures.append(f"dev_prompt_overlap:{len(prompt_overlap)}")
    dev_task_ids = {str(row.get("task_id") or "") for row in source_dev_rows}
    train_task_ids = {str(row.get("task_id") or "") for row in train_rows}
    task_overlap = train_task_ids & dev_task_ids
    if task_overlap:
        failures.append(f"dev_task_overlap:{len(task_overlap)}")
    if failures:
        raise ValueError("invalid_v18_supplement:" + ";".join(failures[:20]))

    train_path = output_dir / "train.jsonl"
    _write_jsonl(train_path, train_rows)
    scenario_rows = Counter(str(row["scenario"]) for row in train_rows)
    role_rows = Counter(str((row.get("metadata") or {}).get("supplement_role")) for row in train_rows)
    summary = {
        "schema_version": "ecommerce_cs_v18_supplement_summary_v1",
        "supplement_version": SUPPLEMENT_VERSION,
        "seed": seed,
        "focus_scenarios": list(FOCUS_SCENARIOS),
        "focus_tasks_per_scenario": focus_tasks_per_scenario,
        "focus_tasks": len(task_rows),
        "focus_rows": len(focus_rows),
        "replay_rows_per_scenario": replay_rows_per_scenario,
        "replay_rows": len(replay_rows),
        "train_rows": len(train_rows),
        "scenario_rows": dict(sorted(scenario_rows.items())),
        "role_rows": dict(sorted(role_rows.items())),
        "dev_rows_in_train": 0,
        "test_rows_in_train": 0,
        "dev_prompt_overlap": 0,
        "dev_task_overlap": 0,
        "original_train": {"path": str(original_train), "sha256": sha256_file(original_train)},
        "frozen_dev": {
            "path": str(original_dev),
            "rows": len(source_dev_rows),
            "sha256": sha256_file(original_dev),
        },
        "files": {
            "tasks.jsonl": {"rows": len(task_rows), "sha256": sha256_file(manifest)},
            "train.jsonl": {"rows": len(train_rows), "sha256": sha256_file(train_path)},
            "reference-focus/summary.json": {"sha256": sha256_file(reference_dir / "summary.json")},
        },
        "reference": reference_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_supplement_training_file(path: Path, *, expected_rows: int) -> dict[str, Any]:
    rows = load_jsonl(path)
    if len(rows) != expected_rows:
        raise ValueError(f"supplement_row_count_mismatch:{len(rows)}:{expected_rows}")
    roles: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()
    for row in rows:
        failures = validate_training_row(row)
        if failures:
            raise ValueError(f"invalid_supplement_row:{row.get('id')}:{','.join(failures)}")
        metadata = dict(row.get("metadata") or {})
        if metadata.get("split") != "train" or metadata.get("test_material") is not False:
            raise ValueError(f"supplement_split_leak:{row.get('id')}")
        roles[str(metadata.get("supplement_role"))] += 1
        scenarios[str(row.get("scenario"))] += 1
    if set(FOCUS_SCENARIOS) - set(scenarios):
        raise ValueError("supplement_focus_coverage_missing")
    if set(roles) != {"focus", "replay"}:
        raise ValueError(f"supplement_roles_invalid:{dict(roles)}")
    return {"rows": len(rows), "roles": dict(roles), "scenarios": dict(sorted(scenarios.items()))}
