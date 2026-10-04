"""Deterministic v18 protocol controller for verified SFT seed traces.

This is data-generation infrastructure, not a learned-model result.  It is
used to prove every generated task has at least one valid trajectory and to
teach the base tool/state/terminal protocol before real-model evaluation.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

from .experiment_prompt_v18 import V18_PROMPT_VERSION, V18_SYSTEM_PROMPT
from .models import TaskSpec
from .tasks import load_task_paths
from .tools_v18 import CustomerServiceToolsV18
from .training import validate_training_row
from .verifier_v18 import V18_VERIFIER_VERSION, score_v18, verify_task_v18


REFERENCE_POLICY_VERSION = "deterministic_reference_v18_r1"


def _money(cents: Any) -> str:
    return f"{int(cents) / 100:.2f}"


def _arguments(task: TaskSpec, name: str, occurrence: int) -> dict[str, Any]:
    meta = task.metadata
    if name == "verify_customer_identity":
        return dict(meta.get("presented_identity") or meta["identity"])
    if name == "list_orders":
        return {"customer_id": task.user_id}
    if name == "get_order":
        order_id = str(meta.get("latest_order_id") or meta.get("order_id") or "")
        if task.scenario == "mixed_order_bulk_cancel" and occurrence == 2:
            order_id = str(meta["shipped_order_id"])
        return {"customer_id": task.user_id, "order_id": order_id}
    if name == "get_shipment":
        return {"customer_id": task.user_id, "order_id": str(meta["order_id"])}
    if name == "search_refund_policy":
        item_id = meta.get("mismatched_item_id") if task.scenario == "cross_order_item_mismatch" else meta.get("item_id")
        return {"customer_id": task.user_id, "order_item_id": str(item_id)}
    if name == "cancel_order":
        return {"customer_id": task.user_id, "order_id": str(meta["order_id"]), "confirmed": True}
    if name == "update_shipping_address":
        return {
            "customer_id": task.user_id,
            "order_id": str(meta["order_id"]),
            "new_address": str(meta["target_address"]),
            "confirmed": True,
        }
    if name == "create_refund":
        quantity = 2 if task.scenario == "full_quantity_refund" else 1
        return {
            "customer_id": task.user_id,
            "order_id": str(meta["order_id"]),
            "order_item_id": str(meta["item_id"]),
            "quantity": quantity,
            "reason": "商品故障，按用户确认和退款政策处理",
            "confirmed": True,
        }
    if name == "escalate_to_human":
        reason = "身份信息不一致" if task.scenario == "identity_mismatch" else "订单查询服务连续失败"
        return {"issue": "电商售后请求", "reason": reason}
    raise ValueError(f"unsupported_reference_tool:{name}")


def _successful_output(calls: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for call in reversed(calls):
        if call.get("name") == name and call.get("ok"):
            return dict(call.get("output") or {})
    return {}


def _message(task: TaskSpec, calls: list[dict[str, Any]]) -> str:
    meta = task.metadata
    scenario = task.scenario
    order_id = str(meta.get("order_id") or "")
    if scenario in {"shipment_status", "identity_tool_transient_error"}:
        shipment = _successful_output(calls, "get_shipment").get("shipment") or {}
        return f"订单正在运输中，承运商为{shipment.get('carrier')}，物流单号 {shipment.get('tracking_number')}。"
    if scenario == "latest_order_lookup":
        return (
            f"最新订单是 {meta['latest_order_id']}，尚未发货，实付金额 "
            f"{_money(meta['latest_amount_cents'])} 元；我没有修改订单状态。"
        )
    if scenario in {"cancel_unshipped", "transient_tool_error"}:
        return f"订单已取消，{_money(meta['paid_amount_cents'])} 元将原路退款。"
    if scenario == "multi_order_disambiguation":
        return f"已核对订单并确认目标，订单已取消，{_money(meta['paid_amount_cents'])} 元将原路退款。"
    if scenario == "cancel_without_captured_payment":
        return "订单已取消；由于未实际扣款，因此无需退款。"
    if scenario == "cancel_already_cancelled":
        return "该订单已经取消并且已经退款，我没有重复操作。"
    if scenario == "mixed_order_bulk_cancel":
        return (
            f"订单 {order_id} 已取消；订单 {meta['shipped_order_id']} 已发货，无法取消，"
            "我没有对已发货订单执行写操作。"
        )
    if scenario == "persistent_tool_error_escalation":
        ticket = _successful_output(calls, "escalate_to_human").get("ticket_id")
        return f"订单查询服务持续异常，我没有猜测取消结果，已经为您创建人工工单 {ticket}。"
    if scenario == "identity_mismatch":
        ticket = _successful_output(calls, "escalate_to_human").get("ticket_id")
        return f"身份信息不一致，我不能查询或操作订单，已经为您创建人工工单 {ticket}。"
    if scenario == "update_address":
        return f"收货地址已更新为{meta['target_address']}。"
    if scenario == "address_change_same_value":
        return f"当前地址已经是{meta['target_address']}，无需修改；我没有重复写入。"
    if scenario == "address_change_after_shipment":
        return "订单已发货，无法修改地址；我没有执行地址变更。"
    if scenario == "cancel_after_shipment":
        return "订单已发货，无法取消；我没有创建退款。"
    if scenario in {"damaged_item_refund", "refund_tool_transient_error", "refund_policy_tool_transient_error"}:
        return f"已按实付金额 {_money(meta['refund_amount_cents'])} 元创建退货并原路退款。"
    if scenario == "partial_quantity_refund":
        return f"已按单件实付金额 {_money(meta['refund_amount_cents'])} 元原路退款，订单中剩余1件未退。"
    if scenario == "full_quantity_refund":
        return f"已退回全部2件，并按实付总额 {_money(meta['refund_amount_cents'])} 元原路退款。"
    if scenario == "refund_quantity_exceeds":
        return "订单记录显示只购买了1件，不能按2件退款；我没有创建超额退款。"
    if scenario == "negative_confirmation":
        return "收到，已停止操作，没有创建退货或退款。"
    if scenario == "late_refund":
        return "该商品已超过30天退货期限，当前不能直接创建退款。"
    if scenario == "refund_before_delivery":
        return "订单尚未签收，不能创建售后退货；我没有提前退款。"
    if scenario == "duplicate_refund_request":
        return "查询到该商品已经退款，不能重复退款；我没有再创建新的退款。"
    if scenario == "cross_order_item_mismatch":
        return "核对发现商品项不属于指定订单，订单与商品参数不一致；我没有执行退款。"
    if scenario in {"final_sale_refund", "policy_bypass_request"}:
        return "根据退款政策，定制商品不支持退货，不能绕过政策直接退款。"
    if scenario == "contradictory_order":
        return "您提供的信息存在矛盾，请确认准确订单号后我再操作。"
    if scenario == "order_not_found":
        return f"您的订单列表中未找到订单 {meta['requested_order_id']}，我没有执行任何操作。"
    raise ValueError(f"unsupported_reference_scenario:{scenario}")


def _final_response(task: TaskSpec, calls: list[dict[str, Any]]) -> str:
    expectation = task.metadata["terminal_v18"]
    action_id = None
    if expectation.get("action_taken"):
        output = _successful_output(calls, str(expectation["action_tool"]))
        action_id = output.get(str(expectation["action_id_field"]))
    return json.dumps(
        {
            "decision": expectation["decision"],
            "action_taken": bool(expectation["action_taken"]),
            "action_id": action_id,
            "message": _message(task, calls),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def run_reference_task_v18(task: TaskSpec) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ecommerce-v18-reference-") as temporary:
        database = Path(temporary) / "state.sqlite"
        shutil.copy2(task.database_path, database)
        isolated = replace(task, database_path=database)
        tools = CustomerServiceToolsV18(database, isolated)
        tools.observe_user_message(isolated.user_prompt)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": V18_SYSTEM_PROMPT},
            {"role": "user", "content": isolated.user_prompt},
        ]
        turn_tools: list[list[dict[str, Any]]] = []
        occurrences: dict[str, int] = {}
        for call_index, name in enumerate(isolated.metadata["required_tool_sequence_v18"]):
            visible = tools.available_schemas()
            visible_names = {item["function"]["name"] for item in visible}
            if name not in visible_names:
                raise RuntimeError(f"reference_tool_not_visible:{isolated.task_id}:{name}:{sorted(visible_names)}")
            turn_tools.append(visible)
            occurrences[name] = occurrences.get(name, 0) + 1
            arguments = _arguments(isolated, name, occurrences[name])
            execution = tools.execute(name, arguments)
            call_id = f"call_{call_index}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(
                            {"ok": execution.ok, "tool": name, **execution.output},
                            ensure_ascii=False,
                        ),
                    },
                ]
            )
        calls = [item.to_dict() for item in tools.executions]
        final_response = _final_response(isolated, calls)
        turn_tools.append(tools.available_schemas())
        messages.append({"role": "assistant", "content": final_response})
        verification = verify_task_v18(isolated, final_response, calls)
        reward = score_v18(isolated, verification, calls)
        if not verification.passed or reward.strict_binary_v18 != 1.0:
            raise RuntimeError(
                f"reference_v18_failed:{isolated.task_id}:{','.join(verification.failures)}"
            )
        return {
            "schema_version": "ecommerce_cs_v18_reference_rollout_v1",
            "task_id": isolated.task_id,
            "split": isolated.split,
            "scenario": isolated.scenario,
            "trial": 0,
            "reward": reward.strict_binary_v18,
            "messages": messages,
            "turn_tools": turn_tools,
            "tool_executions": calls,
            "final_response": final_response,
            "verifier_version": V18_VERIFIER_VERSION,
            "verification": verification.to_dict(),
            "reward_components": reward.to_dict(),
            "stopped_reason": "completed",
            "error": None,
            "policy_version": REFERENCE_POLICY_VERSION,
            "prompt_version": V18_PROMPT_VERSION,
            "learned_model_result": False,
        }


def _training_rows(rollout: dict[str, Any]) -> Iterable[dict[str, Any]]:
    messages = list(rollout["messages"])
    assistant_positions = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    turn_tools = list(rollout["turn_tools"])
    if len(assistant_positions) != len(turn_tools):
        raise ValueError(f"assistant_tool_surface_mismatch:{rollout['task_id']}")
    for turn, (index, visible_tools) in enumerate(zip(assistant_positions, turn_tools)):
        training_messages = messages[: index + 1]
        if messages[index].get("tool_calls"):
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                training_messages.append(messages[cursor])
                cursor += 1
        row = {
            "id": f"sft_v18_{rollout['task_id']}_turn{turn}",
            "task_id": rollout["task_id"],
            "scenario": rollout["scenario"],
            "format_version": "ecommerce_cs_agent_sft_v18",
            "messages": training_messages,
            "tools": visible_tools,
            "loss_mask": [position == index for position in range(len(training_messages))],
            "metadata": {
                "source": REFERENCE_POLICY_VERSION,
                "split": rollout["split"],
                "strict_binary_v18": 1.0,
                "assistant_turn": turn,
                "test_material": False,
                "learned_model_result": False,
            },
        }
        failures = validate_training_row(row)
        if failures:
            raise ValueError(f"invalid_v18_training_row:{row['id']}:{','.join(failures)}")
        yield row


def build_reference_sft_v18(
    manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"v18 reference output must be empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = load_task_paths(manifest)
    handles = {
        "train": (output_dir / "train.jsonl").open("x", encoding="utf-8", newline="\n"),
        "dev": (output_dir / "dev.jsonl").open("x", encoding="utf-8", newline="\n"),
    }
    rollout_handle = (output_dir / "reference-rollouts.jsonl").open("x", encoding="utf-8", newline="\n")
    task_counts = {"train": 0, "dev": 0}
    row_counts = {"train": 0, "dev": 0}
    scenario_counts: dict[str, int] = {}
    try:
        for path in paths:
            task = TaskSpec.from_file(path)
            if task.split not in handles:
                raise ValueError(f"test_or_unknown_split_in_v18_reference:{task.task_id}:{task.split}")
            rollout = run_reference_task_v18(task)
            rollout_handle.write(json.dumps(rollout, ensure_ascii=False, sort_keys=True) + "\n")
            task_counts[task.split] += 1
            scenario_counts[task.scenario] = scenario_counts.get(task.scenario, 0) + 1
            for row in _training_rows(rollout):
                handles[task.split].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                row_counts[task.split] += 1
    finally:
        rollout_handle.close()
        for handle in handles.values():
            handle.close()

    files = {}
    for name in ("train.jsonl", "dev.jsonl", "reference-rollouts.jsonl"):
        path = output_dir / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()}
    summary = {
        "schema_version": "ecommerce_cs_v18_reference_sft_summary_v1",
        "policy_version": REFERENCE_POLICY_VERSION,
        "prompt_version": V18_PROMPT_VERSION,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "tasks": task_counts,
        "training_rows": row_counts,
        "scenario_task_counts": dict(sorted(scenario_counts.items())),
        "all_reference_strict_pass": True,
        "test_tasks": 0,
        "learned_model_result": False,
        "files": files,
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
