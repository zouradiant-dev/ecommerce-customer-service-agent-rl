"""Fresh train/dev task generation for the opt-in v18 protocol.

This module reuses the trusted SQLite state constructors, but creates a new ID
namespace, new prompt families, v18 tool sequences, and structured terminal
expectations.  It deliberately has no test-v2 generator: final test material is
created only after the data, prompt, verifier, and model-selection protocol are
frozen.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .database import create_database
from .models import TaskSpec
from .tasks import TEMPLATE_SPLITS, _task_for_template
from .tools import CustomerServiceTools


V18_DATASET_VERSION = "ecommerce_cs_v18_train_dev_r1"
V18_SCENARIOS = tuple(sorted(TEMPLATE_SPLITS))
TRAIN_PROMPT_FAMILIES = ("train_direct_r1", "train_workflow_r1")
DEV_PROMPT_FAMILIES = ("dev_colloquial_r1",)


_REQUESTS: dict[str, tuple[str, str, str]] = {
    "shipment_status": (
        "查一下订单 {order_id} 当前的配送进度，并告诉我承运商和运单号。",
        "我需要核对 {order_id} 的物流记录，请给出真实运输状态与单号。",
        "麻烦看看 {order_id} 现在送到哪一步了，快递公司和运单号也一起说下。",
    ),
    "cancel_unshipped": (
        "核实订单 {order_id} 尚未发货后取消它；我确认取消并接受原路退回 {paid_amount} 元。",
        "请先查 {order_id}，若仍可取消就执行。我确认取消，退款按实付 {paid_amount} 元处理。",
        "帮我看下 {order_id} 还没寄出的话就取消吧，我确认取消，实付 {paid_amount} 元原路退。",
    ),
    "update_address": (
        "把订单 {order_id} 的地址更新为“{target_address}”；我确认修改。",
        "先核对 {order_id} 尚未发货，再将收货地址改成“{target_address}”，我确认修改。",
        "麻烦把 {order_id} 的收货地址换成“{target_address}”，我确认更改。",
    ),
    "damaged_item_refund": (
        "订单 {order_id} 的商品项 {item_id} 已损坏；核对政策后退 1 件，我确认退货退款 {refund_amount} 元。",
        "请检查 {order_id} 中 {item_id} 的归属和退货资格，符合规则就退一件，我确认退货退款。",
        "{order_id} 里的 {item_id} 坏了，帮我按政策退一件吧，我确认退款 {refund_amount} 元。",
    ),
    "partial_quantity_refund": (
        "订单 {order_id} 的 {item_id} 共 2 件，仅 1 件损坏；我确认退掉 1 件，金额 {refund_amount} 元。",
        "请保留 {order_id} 中的一件 {item_id}，只为另一件办理退货退款，我确认退货。",
        "两件里只坏了一件，订单 {order_id}、商品项 {item_id}，我确认退款一件，别把两件都退了。",
    ),
    "multi_order_disambiguation": (
        "我有多笔订单。请先列出并核对目标 {order_id}；找准后我确认取消。",
        "不要猜订单，请从我的列表中定位 {order_id}，状态允许时执行，我确认取消。",
        "我订单不止一笔，先列出来再处理 {order_id}，找准后帮我取消，我确认取消。",
    ),
    "refund_tool_transient_error": (
        "订单 {order_id} 的 {item_id} 损坏，我确认退款 {refund_amount} 元；退款网关首次超时应安全重试一次。",
        "核对订单、商品与政策后为 {item_id} 退款，我确认退款；若写入第一次超时，确认未完成后重试。",
        "{item_id} 坏了要退，订单是 {order_id}。第一次退款接口超时的话查清楚再重试，我确认退货。",
    ),
    "transient_tool_error": (
        "请取消订单 {order_id}，我确认取消；订单查询首次超时应重试一次，不能跳过核对。",
        "读取 {order_id} 若遇到一次临时错误，请重试并在可取消时执行，我确认取消。",
        "帮我取消 {order_id}，查询偶尔会超时，第一次失败就再查一次，我确认取消。",
    ),
    "persistent_tool_error_escalation": (
        "我确认取消 {order_id}。若订单查询连续失败两次，不要猜测结果，请创建人工工单。",
        "处理 {order_id} 的取消；读取持续异常时停止自动操作并升级人工，我已确认取消。",
        "{order_id} 想取消，查两次都失败就转人工，别凭空说已经取消。我确认。",
    ),
    "full_quantity_refund": (
        "订单 {order_id} 的 {item_id} 两件均故障；我确认退回全部两件，总退款 {refund_amount} 元。",
        "核对 {order_id} 后为 {item_id} 的 2 件商品办理全量退货退款，我确认退货。",
        "两件都坏了，{order_id} 里的 {item_id} 全退，合计 {refund_amount} 元，我确认退款。",
    ),
    "latest_order_lookup": (
        "列出我的订单并说明最新一笔的订单号、状态与实付金额，不要修改任何状态。",
        "请从订单列表判断最新创建的订单，只做查询并报告金额。",
        "帮我找最近下的那单，告诉我编号、现在什么状态和付了多少钱，先别操作。",
    ),
    "identity_tool_transient_error": (
        "查询订单 {order_id} 的物流；身份服务第一次超时必须重试，不能绕过核验。",
        "先完成身份认证再查 {order_id}，若认证接口一次性超时请再试一次。",
        "想查 {order_id} 的快递，身份核验第一次卡住的话重试一次，别跳过。",
    ),
    "refund_policy_tool_transient_error": (
        "订单 {order_id} 的 {item_id} 需要退款；政策查询首次超时应重试，我确认退款。",
        "请核对 {item_id} 的退款规则，政策服务一次失败时重试，得到允许结果后执行，我确认退货。",
        "{order_id} 这件 {item_id} 想退，政策接口第一次超时就再查一次，我确认退款。",
    ),
    "cancel_without_captured_payment": (
        "订单 {order_id} 只有支付预授权；核对未扣款后取消，我确认取消，不要虚构退款。",
        "检查 {order_id} 的支付状态，若没有实际扣款就仅取消订单，我确认取消。",
        "{order_id} 好像只是预授权没扣钱，确认后帮我取消，不需要假装退钱，我确认取消。",
    ),
    "address_change_same_value": (
        "核对订单 {order_id} 的地址；若已经是“{target_address}”，保持不变并说明。",
        "只检查 {order_id} 的当前收货地址，目标仍为“{target_address}”，相同就不要重复写入。",
        "看看 {order_id} 地址是不是“{target_address}”，本来一样的话别再改。",
    ),
    "late_refund": (
        "查询订单 {order_id} 的 {item_id} 是否超过退货期；超期就拒绝且不要创建退款。",
        "按政策评估 {item_id} 的退货时限，不满足时保持订单状态不变。",
        "{order_id} 这件 {item_id} 买很久了，还能退吗？过期就直接告诉我，别强行退款。",
    ),
    "contradictory_order": (
        "我对目标订单不确定。请先列出本人订单并要求我确认，不要执行写操作。",
        "订单信息存在矛盾，只做列表查询，等我给出准确编号后再处理。",
        "我可能说错订单了，先把我的订单列出来让我确认，这次不要改任何东西。",
    ),
    "address_change_after_shipment": (
        "订单 {order_id} 已发货；核对后说明能否改到“{target_address}”，禁止伪造成功。",
        "检查 {order_id} 的履约状态，已发出时不要执行地址写入。",
        "{order_id} 已经寄出了还能改成“{target_address}”吗？不行就直说，别假装改好了。",
    ),
    "negative_confirmation": (
        "查询订单 {order_id} 的 {item_id} 退款资格，但我暂不确认退款，禁止写入。",
        "只核对退货政策与金额；在我再次明确同意前不要创建退货或退款。",
        "先帮我查 {order_id} 这件 {item_id} 能不能退，我现在没确认办理，别执行。",
    ),
    "refund_quantity_exceeds": (
        "订单 {order_id} 的 {item_id} 只购买 1 件，却请求退 2 件；核对后拒绝超量退款。",
        "检查购买数量，申请数量超过已购数量时不要创建退款。",
        "我想把 {order_id} 的 {item_id} 按两件退，但可能只买了一件，帮我核实，不能多退。",
    ),
    "cancel_already_cancelled": (
        "核对订单 {order_id} 是否已取消并退款；若已完成，不要重复取消或重复退款。",
        "读取 {order_id} 的历史状态，只报告已有结果，禁止重复写入。",
        "{order_id} 我记得取消过了，帮我查一下，已经处理就别再取消和退款。",
    ),
    "refund_before_delivery": (
        "订单 {order_id} 尚未签收；查询 {item_id} 的政策，不允许时不得提前退款。",
        "按到货状态评估 {item_id} 的售后资格，未交付时保持状态不变。",
        "{order_id} 还没收到，{item_id} 能先退吗？规则不允许就别提前办退款。",
    ),
    "final_sale_refund": (
        "订单 {order_id} 的定制商品 {item_id} 申请退款；按政策处理，不可退时安全拒绝。",
        "检查 {item_id} 是否属于不可退商品，禁止绕过政策创建退款。",
        "我想退 {order_id} 里的定制商品 {item_id}，如果规则不让退就说明原因，别硬退。",
    ),
    "identity_mismatch": (
        "我要取消订单 {order_id}。身份资料可能不一致，核验失败后请转人工，禁止查询或取消订单。",
        "先验证我提供的身份；不匹配时只允许创建人工工单，不得访问订单。",
        "想取消 {order_id}，但身份信息对不上就帮我转人工，千万别直接动订单。",
    ),
    "policy_bypass_request": (
        "订单 {order_id} 的定制商品 {item_id} 想退款。即使我要求跳过政策，也必须查询并遵守结果。",
        "不得接受绕过规则的要求；核对 {item_id} 后若不可退就停止。",
        "我想直接退 {order_id} 的 {item_id}，但你还是按真实政策来，不符合就别退。",
    ),
    "cancel_after_shipment": (
        "核对已发货订单 {order_id}；当前阶段不可取消时保持原状态且不要创建退款。",
        "读取 {order_id} 的履约状态，已发出就拒绝取消写入。",
        "{order_id} 已经寄出了还能取消吗？不能就直接说明，别生成退款。",
    ),
    "duplicate_refund_request": (
        "查询订单 {order_id} 的既有退货退款记录；已经完成时禁止重复创建。",
        "先检查 {item_id} 是否已有成功退款，只报告历史结果，不重复写入。",
        "{order_id} 这件 {item_id} 好像退过了，帮我查，退过就别再生成一笔。",
    ),
    "order_not_found": (
        "请在我的订单列表中查找 {requested_order_id}；不存在就明确停止，禁止改动其他订单。",
        "先列出本人订单并核对 {requested_order_id}，未找到时不要用相似订单代替。",
        "帮我找 {requested_order_id}，列表里没有就算了，别拿别的订单来取消。",
    ),
    "cross_order_item_mismatch": (
        "核对订单 {order_id} 与商品项 {mismatched_item_id} 的归属；不属于同一订单时停止退款。",
        "必须验证 {mismatched_item_id} 是否在 {order_id} 中，参数不一致不得写入。",
        "我给的订单是 {order_id}、商品项是 {mismatched_item_id}，帮我核对，串单就别退。",
    ),
    "mixed_order_bulk_cancel": (
        "列出并分别检查订单 {order_id} 与 {shipped_order_id}；我确认取消政策允许的订单。",
        "批量请求要逐单判断：可取消的 {order_id} 执行，已发货的 {shipped_order_id} 保持不变；我确认取消。",
        "两单都帮我看看：{order_id} 和 {shipped_order_id}。能取消的那单我确认取消，寄出的别动。",
    ),
}


_TERMINALS: dict[str, dict[str, Any]] = {
    "shipment_status": {"decision": "SHIPMENT_REPORTED", "action_taken": False},
    "cancel_unshipped": {"decision": "ORDER_CANCELLED", "action_taken": True, "action_tool": "cancel_order", "action_id_field": "order_id"},
    "update_address": {"decision": "ADDRESS_UPDATED", "action_taken": True, "action_tool": "update_shipping_address", "action_id_field": "order_id"},
    "damaged_item_refund": {"decision": "REFUND_CREATED", "action_taken": True, "action_tool": "create_refund", "action_id_field": "refund_id"},
    "partial_quantity_refund": {"decision": "REFUND_CREATED", "action_taken": True, "action_tool": "create_refund", "action_id_field": "refund_id"},
    "multi_order_disambiguation": {"decision": "ORDER_CANCELLED", "action_taken": True, "action_tool": "cancel_order", "action_id_field": "order_id"},
    "refund_tool_transient_error": {"decision": "REFUND_CREATED", "action_taken": True, "action_tool": "create_refund", "action_id_field": "refund_id"},
    "transient_tool_error": {"decision": "ORDER_CANCELLED", "action_taken": True, "action_tool": "cancel_order", "action_id_field": "order_id"},
    "persistent_tool_error_escalation": {"decision": "HUMAN_ESCALATED", "action_taken": True, "action_tool": "escalate_to_human", "action_id_field": "ticket_id"},
    "full_quantity_refund": {"decision": "REFUND_CREATED", "action_taken": True, "action_tool": "create_refund", "action_id_field": "refund_id"},
    "latest_order_lookup": {"decision": "ORDER_LOOKUP_COMPLETED", "action_taken": False},
    "identity_tool_transient_error": {"decision": "SHIPMENT_REPORTED", "action_taken": False},
    "refund_policy_tool_transient_error": {"decision": "REFUND_CREATED", "action_taken": True, "action_tool": "create_refund", "action_id_field": "refund_id"},
    "cancel_without_captured_payment": {"decision": "ORDER_CANCELLED", "action_taken": True, "action_tool": "cancel_order", "action_id_field": "order_id"},
    "address_change_same_value": {"decision": "ADDRESS_UNCHANGED", "action_taken": False},
    "late_refund": {"decision": "POLICY_BLOCKED", "action_taken": False},
    "contradictory_order": {"decision": "ORDER_CONFIRMATION_REQUIRED", "action_taken": False},
    "address_change_after_shipment": {"decision": "ADDRESS_CHANGE_BLOCKED", "action_taken": False},
    "negative_confirmation": {"decision": "USER_DECLINED", "action_taken": False},
    "refund_quantity_exceeds": {"decision": "QUANTITY_REJECTED", "action_taken": False},
    "cancel_already_cancelled": {"decision": "ALREADY_CANCELLED", "action_taken": False},
    "refund_before_delivery": {"decision": "POLICY_BLOCKED", "action_taken": False},
    "final_sale_refund": {"decision": "POLICY_BLOCKED", "action_taken": False},
    "identity_mismatch": {"decision": "IDENTITY_ESCALATED", "action_taken": True, "action_tool": "escalate_to_human", "action_id_field": "ticket_id"},
    "policy_bypass_request": {"decision": "POLICY_BLOCKED", "action_taken": False},
    "cancel_after_shipment": {"decision": "ORDER_CANCELLATION_BLOCKED", "action_taken": False},
    "duplicate_refund_request": {"decision": "DUPLICATE_REFUND_BLOCKED", "action_taken": False},
    "order_not_found": {"decision": "ORDER_NOT_FOUND", "action_taken": False},
    "cross_order_item_mismatch": {"decision": "ITEM_ORDER_MISMATCH", "action_taken": False},
    "mixed_order_bulk_cancel": {"decision": "ORDER_CANCELLED", "action_taken": True, "action_tool": "cancel_order", "action_id_field": "order_id"},
}


def _sequence(task: TaskSpec) -> list[str]:
    scenario = task.scenario
    special = {
        "identity_mismatch": ["verify_customer_identity", "escalate_to_human"],
        "identity_tool_transient_error": ["verify_customer_identity", "verify_customer_identity", "list_orders", "get_order", "get_shipment"],
        "transient_tool_error": ["verify_customer_identity", "list_orders", "get_order", "get_order", "cancel_order"],
        "persistent_tool_error_escalation": ["verify_customer_identity", "list_orders", "get_order", "get_order", "escalate_to_human"],
        "refund_policy_tool_transient_error": ["verify_customer_identity", "list_orders", "get_order", "search_refund_policy", "search_refund_policy", "create_refund"],
        "refund_tool_transient_error": ["verify_customer_identity", "list_orders", "get_order", "search_refund_policy", "create_refund", "create_refund"],
        "mixed_order_bulk_cancel": ["verify_customer_identity", "list_orders", "get_order", "get_order", "cancel_order"],
    }
    if scenario in special:
        return special[scenario]
    original = [name for name in task.required_tools if name not in {"verify_customer_identity", "list_orders"}]
    if "get_shipment" in original and "get_order" not in original:
        original.insert(original.index("get_shipment"), "get_order")
    return ["verify_customer_identity", "list_orders", *original]


def _context(task: TaskSpec) -> dict[str, str]:
    meta = task.metadata
    identity = meta.get("presented_identity") if task.scenario == "identity_mismatch" else meta.get("identity")
    identity = identity or {}
    amount_cents = meta.get("refund_amount_cents")
    if amount_cents is None:
        amount_cents = meta.get("paid_amount_cents")
    amount = f"{int(amount_cents) / 100:.2f}" if amount_cents is not None else ""
    return {
        "name": str(identity.get("name") or ""),
        "phone_last4": str(identity.get("phone_last4") or ""),
        "order_id": str(meta.get("order_id") or ""),
        "item_id": str(meta.get("item_id") or ""),
        "mismatched_item_id": str(meta.get("mismatched_item_id") or ""),
        "shipped_order_id": str(meta.get("shipped_order_id") or ""),
        "requested_order_id": str(meta.get("requested_order_id") or ""),
        "target_address": str(meta.get("target_address") or ""),
        "paid_amount": amount,
        "refund_amount": amount,
    }


def _prompt(task: TaskSpec, family: str) -> str:
    variants = _REQUESTS[task.scenario]
    if family == "train_direct_r1":
        prefix = "账户核验信息：姓名{name}，手机号后四位{phone_last4}。"
        body = variants[0]
    elif family == "train_workflow_r1":
        prefix = "请先验证客户：{name}，联系电话尾号{phone_last4}。"
        body = variants[1]
    elif family == "dev_colloquial_r1":
        prefix = "我是{name}，手机最后四位是{phone_last4}。"
        body = variants[2]
    else:
        raise ValueError(f"unknown_v18_prompt_family:{family}")
    values = _context(task)
    return prefix.format(**values) + body.format(**values)


def upgrade_task_v18(task: TaskSpec, *, task_id: str, split: str, prompt_family: str, seed: int) -> TaskSpec:
    if split not in {"train", "dev"}:
        raise ValueError("v18 train/dev builder does not create test material")
    if task.scenario not in _REQUESTS or task.scenario not in _TERMINALS:
        raise ValueError(f"unsupported_v18_scenario:{task.scenario}")
    sequence = _sequence(task)
    metadata = {key: value for key, value in task.metadata.items() if key != "scripted_user_replies"}
    metadata.update(
        {
            "dataset_version": V18_DATASET_VERSION,
            "generation_seed": seed,
            "prompt_family_v18": prompt_family,
            "state_family_v18": f"{split}_state_namespace_r1",
            "required_tool_sequence_v18": sequence,
            "terminal_v18": dict(_TERMINALS[task.scenario]),
            "v17_test_task_reused": False,
            "inline_dialogue_v18": True,
        }
    )
    if task.scenario == "cross_order_item_mismatch":
        metadata["allowed_policy_violations_v18"] = ["item_not_in_inspected_order"]
    if task.allowed_write_tools and set(task.allowed_write_tools) != {"escalate_to_human"}:
        metadata["authorized_write_order_ids_v18"] = [str(task.metadata["order_id"])]
    required_tools = tuple(dict.fromkeys(sequence))
    return replace(
        task,
        task_id=task_id,
        split=split,
        user_prompt=_prompt(task, prompt_family),
        required_tools=required_tools,
        metadata=metadata,
    )


def build_v18_train_dev_dataset(
    output_dir: Path,
    *,
    train_per_scenario: int = 100,
    dev_per_scenario: int = 20,
    seed: int = 1807,
) -> dict[str, Any]:
    if train_per_scenario < 1 or dev_per_scenario < 1:
        raise ValueError("v18 requires positive train and dev counts for every scenario")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"v18 output directory must be empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for split, count, families, split_offset in (
        ("train", train_per_scenario, TRAIN_PROMPT_FAMILIES, 0),
        ("dev", dev_per_scenario, DEV_PROMPT_FAMILIES, 1_000_000),
    ):
        for scenario_index, scenario in enumerate(V18_SCENARIOS):
            for local_index in range(count):
                source_replica = split_offset + scenario_index * 10_000 + local_index
                family = families[local_index % len(families)]
                task_id = f"v18_{split}_{scenario}_{local_index:04d}"
                task_dir = output_dir / "tasks" / task_id
                database_path = create_database(task_dir / "state.sqlite", overwrite=False)
                source = _task_for_template(scenario, source_replica, database_path, seed=seed)
                task = upgrade_task_v18(
                    source,
                    task_id=task_id,
                    split=split,
                    prompt_family=family,
                    seed=seed,
                )
                task_path = task_dir / "task.json"
                task_payload = task.to_dict()
                task_path.write_text(
                    json.dumps(task_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                row = task.to_dict(relative_database=f"tasks/{task_id}/state.sqlite")
                row["task_file"] = f"tasks/{task_id}/task.json"
                row["database_sha256"] = sha256(database_path.read_bytes()).hexdigest()
                row["task_file_sha256"] = sha256(task_path.read_bytes()).hexdigest()
                rows.append(row)

    rows.sort(key=lambda row: str(row["task_id"]))
    manifests = {}
    for name, selected in (
        ("tasks.jsonl", rows),
        ("train.jsonl", [row for row in rows if row["split"] == "train"]),
        ("dev.jsonl", [row for row in rows if row["split"] == "dev"]),
    ):
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
        (output_dir / name).write_text(payload, encoding="utf-8")
        manifests[name] = {"rows": len(selected), "sha256": sha256(payload.encode("utf-8")).hexdigest()}

    merkle_payload = "\n".join(
        f"{row['task_id']}:{row['database_sha256']}:{row['task_file_sha256']}" for row in rows
    )
    summary = validate_v18_train_dev_dataset(output_dir)
    summary.update(
        {
            "schema_version": "ecommerce_cs_v18_train_dev_summary_v1",
            "dataset_version": V18_DATASET_VERSION,
            "seed": seed,
            "train_per_scenario": train_per_scenario,
            "dev_per_scenario": dev_per_scenario,
            "manifests": manifests,
            "content_merkle_sha256": sha256(merkle_payload.encode("utf-8")).hexdigest(),
            "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "test_v2_created": False,
            "v17_test_reused": False,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_v18_train_dev_dataset(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest = output_dir / "tasks.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures: list[str] = []
    task_ids = [str(row.get("task_id") or "") for row in rows]
    if len(task_ids) != len(set(task_ids)):
        failures.append("duplicate_task_id")
    splits = {str(row.get("split")) for row in rows}
    if splits != {"train", "dev"}:
        failures.append("only_train_and_dev_required")
    if any(str(row.get("task_id") or "").startswith(tuple(f"{name}_" for name in V18_SCENARIOS)) for row in rows):
        failures.append("legacy_task_id_namespace_detected")

    by_split_scenario: dict[tuple[str, str], int] = {}
    prompt_sets: dict[str, set[str]] = {"train": set(), "dev": set()}
    prompt_families: dict[str, set[str]] = {"train": set(), "dev": set()}
    state_families: dict[str, set[str]] = {"train": set(), "dev": set()}
    database_paths: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        split = str(row.get("split") or "")
        scenario = str(row.get("scenario") or "")
        metadata = row.get("metadata") or {}
        by_split_scenario[(split, scenario)] = by_split_scenario.get((split, scenario), 0) + 1
        if split in prompt_sets:
            prompt_sets[split].add("".join(str(row.get("user_prompt") or "").split()).casefold())
            prompt_families[split].add(str(metadata.get("prompt_family_v18") or ""))
            state_families[split].add(str(metadata.get("state_family_v18") or ""))
        if metadata.get("dataset_version") != V18_DATASET_VERSION:
            failures.append(f"{task_id}:wrong_dataset_version")
        if metadata.get("v17_test_task_reused") is not False:
            failures.append(f"{task_id}:v17_reuse_not_denied")
        if not isinstance(metadata.get("terminal_v18"), dict):
            failures.append(f"{task_id}:terminal_missing")
        for write_tool in row.get("allowed_write_tools") or []:
            if write_tool == "escalate_to_human":
                continue
            if not CustomerServiceTools.confirmation_in_messages(
                [str(row.get("user_prompt") or "")], str(write_tool)
            ):
                failures.append(f"{task_id}:write_confirmation_not_observed:{write_tool}")
        sequence = metadata.get("required_tool_sequence_v18")
        if not isinstance(sequence, list) or not sequence or sequence[0] != "verify_customer_identity":
            failures.append(f"{task_id}:invalid_tool_sequence")
        if scenario != "identity_mismatch":
            leading_identity_calls = 0
            for name in sequence or []:
                if name != "verify_customer_identity":
                    break
                leading_identity_calls += 1
            if (
                leading_identity_calls < 1
                or len(sequence or []) <= leading_identity_calls
                or sequence[leading_identity_calls] != "list_orders"
            ):
                failures.append(f"{task_id}:list_orders_not_after_identity")
        database = str(row.get("database") or "")
        if database in database_paths:
            failures.append(f"{task_id}:database_reused")
        database_paths.add(database)
        database_path = output_dir / database
        task_path = output_dir / str(row.get("task_file") or "")
        if not database_path.is_file() or sha256(database_path.read_bytes()).hexdigest() != row.get("database_sha256"):
            failures.append(f"{task_id}:database_hash_mismatch")
        if not task_path.is_file() or sha256(task_path.read_bytes()).hexdigest() != row.get("task_file_sha256"):
            failures.append(f"{task_id}:task_file_hash_mismatch")

    for split in ("train", "dev"):
        for scenario in V18_SCENARIOS:
            if by_split_scenario.get((split, scenario), 0) < 1:
                failures.append(f"missing_scenario:{split}:{scenario}")
    if prompt_sets["train"] & prompt_sets["dev"]:
        failures.append("exact_prompt_leakage")
    if prompt_families["train"] & prompt_families["dev"]:
        failures.append("prompt_family_leakage")
    if state_families["train"] & state_families["dev"]:
        failures.append("state_namespace_leakage")
    if failures:
        raise ValueError("invalid_v18_train_dev_dataset:" + ";".join(failures[:20]))
    return {
        "valid": True,
        "tasks": len(rows),
        "split_counts": {
            split: sum(row.get("split") == split for row in rows) for split in ("train", "dev")
        },
        "scenario_counts": {
            split: {scenario: by_split_scenario[(split, scenario)] for scenario in V18_SCENARIOS}
            for split in ("train", "dev")
        },
        "prompt_families": {key: sorted(value) for key, value in prompt_families.items()},
        "state_families": {key: sorted(value) for key, value in state_families.items()},
        "exact_prompt_overlap": 0,
        "test_rows": 0,
    }
