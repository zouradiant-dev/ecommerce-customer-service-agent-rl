"""Synthetic, executable reference supervision for the four-tool online protocol."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from uuid import UUID
from unittest.mock import patch

from ..training import canonicalize_tool_schemas, normalize_messages, render_with_assistant_mask, validate_training_row
from .context_builder import ContextBuilder
from .online_cases import prepare as prepare_fixture, say, tool
from .online_evaluation import run_case
from .policy import get_policy

KINDS = ("unique_exact", "unique_alias", "no_match", "true_ambiguity", "query_only", "shipped", "unpaid",
         "withdraw", "change_target", "chat_confirm", "cross_owner", "read_retry", "confirmed_result", "query_then_cancel", "handoff")
PRODUCTS = {
    "train": [("咖啡豆","咖啡"),("瑜伽垫","垫子"),("蓝牙音箱","音箱"),("运动鞋","鞋子"),
              ("电热水壶","水壶"),("枕头","枕头"),("电风扇","风扇"),("台灯","台灯")],
    "dev": [("电饭煲","饭煲"),("双肩背包","背包"),("电磁炉","炉子"),("书桌","桌子")],
    "test": [("空气净化器","净化器"),("行车记录仪","记录仪"),("卷发棒","卷发棒"),("电烤箱","烤箱")],
}
REQUESTS = {
    "train": ("帮我取消{product}订单。", "我不想要{product}了，请取消这单。"),
    "dev": ("请处理{product}的取消申请。", "{product}这单可以帮我取消吗？"),
    "test": ("想撤销{product}这笔订单，请帮忙处理。", "计划取消{product}，麻烦查查这笔。"),
}


def prepare(order_id: str, *, listing: bool = True) -> list[dict]:
    responses = prepare_fixture(order_id, listing=listing)
    responses[-1] = say("已生成取消预览，请点击界面确认按钮确认操作。")
    return responses


def episode(split: str, kind: str, variant: int, seed: int) -> dict:
    pool, kind_index = PRODUCTS[split], KINDS.index(kind)
    product, alias = pool[(variant + kind_index) % len(pool)]
    fingerprint = sha256(f"{seed}/{split}/{kind}/{variant}".encode()).hexdigest()
    owned_count = (1, 3, 5)[variant % 3]
    if kind in {"true_ambiguity", "change_target"}:
        owned_count = max(2, owned_count)
    orders = []
    distractors = [name for name, _ in pool if name != product]
    replacement = distractors[variant % len(distractors)]
    for index in range(owned_count + 1):
        order_id = "EO" + sha256(f"{fingerprint}/{index}".encode()).hexdigest()[:10].upper()
        name = product if index == 0 else distractors[(variant + index - 1) % len(distractors)]
        if kind == "change_target" and index == 1:
            name = replacement
        elif kind == "change_target" and index >= 2:
            alternatives = [name for name in distractors if name != replacement]
            name = alternatives[(variant + index) % len(alternatives)]
        orders.append({"id": order_id, "customer_id": "customer_alice" if index < owned_count else "customer_bob",
            "product": name, "status": "paid", "paid_cents": 8900 + variant * 700 + ((index + variant) % owned_count) * 10000, "version": 1})
    target = orders[0]["id"]
    if kind == "true_ambiguity":
        orders[1]["product"] = product
    if kind == "shipped": orders[0]["status"] = "shipped"
    if kind == "unpaid": orders[0].update(status="pending", paid_cents=0)

    def message(text, save="latest"): return {"kind":"message","text":text,"save_as":save}
    def confirm(ref="latest", error=None): return {"kind":"confirm","ref":ref, **({"expected_error":error} if error else {})}
    def checkpoint(count): return {"kind":"checkpoint","assertions":{"pending_count":count,"cancellation_count":0}}
    request = REQUESTS[split][variant % 2].format(product=alias if kind == "unique_alias" else product)
    steps, reference, cancelled, pending = [message(request), confirm()], prepare(target), target, 0
    if kind == "no_match":
        absent = {"train":"储物盒","dev":"挂钟","test":"户外帐篷"}[split]
        steps, reference, cancelled = [message(REQUESTS[split][variant%2].format(product=absent)),checkpoint(0)], [tool("list_orders"),say("没有找到与该商品匹配的本人订单，请核对商品或订单号。")], None
    elif kind == "true_ambiguity":
        steps = [message(request),checkpoint(0),message(f"取消金额 {orders[0]['paid_cents']/100:.2f} 元的那一笔。"),confirm()]
        reference = [tool("list_orders"),say("找到两笔同名商品订单，请按金额或订单号指定一笔。")]+prepare(target,listing=False)
    elif kind == "query_only":
        steps, reference, cancelled = [message(f"只查询{product}订单，暂时不要取消。"),checkpoint(0)], [tool("list_orders"),tool("get_order",order_id=target),say("订单已付款，目前没有取消申请。")], None
    elif kind == "shipped":
        steps, reference, cancelled = [message(request),checkpoint(0)], [tool("list_orders"),tool("get_order",order_id=target),say("该订单已发货，不能直接取消。")], None
    elif kind == "withdraw":
        steps, reference, cancelled = [message(request,"old"),checkpoint(1),message("先保留订单，不取消了。"),confirm("old","confirmation_invalidated")], prepare(target)+[say("好的，订单保持原状，不再继续取消申请。")], None
    elif kind == "change_target":
        cancelled = orders[1]["id"]
        steps = [message(request,"old"),checkpoint(1),message(f"改为取消{orders[1]['product']}，{product}保留。","new"),confirm("old","confirmation_invalidated"),confirm("new")]
        reference = prepare(target)+prepare(cancelled,listing=False)
    elif kind == "chat_confirm":
        steps, reference, cancelled, pending = [message(request),checkpoint(1),message("确认取消。"),checkpoint(1)], prepare(target)+prepare(target,listing=False), None, 1
    elif kind == "cross_owner":
        steps, reference, cancelled = [message(f"取消订单 {orders[-1]['id']}。"),checkpoint(0)], [tool("get_order",order_id=orders[-1]["id"]),say("未找到您名下该订单，无法处理。")], None
    elif kind == "read_retry":
        steps = [{"kind":"event","name":"read_timeout_once"},message(request),confirm()]
        reference = [tool("list_orders"),tool("get_order",order_id=target)]+prepare(target,listing=False)
    elif kind == "confirmed_result":
        steps = [message(request),confirm(),message("我已经点击了界面确认按钮，现在处理结果怎么样？")]
        reference = prepare(target)+[tool("get_order",order_id=target),say("订单已经取消，退款处理中，尚未确认到账。")]
    elif kind == "query_then_cancel":
        steps = [message(f"先查询{product}订单，不要操作。"),checkpoint(0),message(request),confirm()]
        reference = [tool("list_orders"),tool("get_order",order_id=target),say("已查询订单，暂不创建取消申请。")]+prepare(target,listing=False)
    elif kind == "handoff":
        steps, reference, cancelled = [message("请转人工客服处理。")], [tool("escalate_to_human",reason="用户主动要求人工帮助")], None
    states = {order["id"]: order["status"] for order in orders}
    amount = next((order["paid_cents"] if order["status"]=="paid" else 0 for order in orders if order["id"]==cancelled),0)
    if cancelled: states[cancelled] = "cancelled"
    return {"schema_version":"online_repair_episode_v1","case_id":f"repair_{split}_{kind}_{variant:03d}","split":split,
        "category":kind,"customer_id":"customer_alice","initial_orders":orders,"steps":steps,
        "expected":{"order_statuses":states,"refund_count":int(bool(cancelled and amount)),"cancellation_count":int(bool(cancelled)),
            "pending_count":pending,"conversation_status":"handed_off" if kind == "handoff" else "active","forbidden_reply_phrases":["退款已到账","退款已经到账"],
            **({"refund_status":"pending"} if cancelled and amount else {})},
        "smoke_responses":reference,"provenance":"synthetic_programmatic_reference_not_customer_logs"}


class RecordingReference:
    model = "verified_programmatic_reference_not_llm"
    def __init__(self, case: dict):
        self.case, self.responses, self.samples = case, deepcopy(case["smoke_responses"]), []
    def complete(self, messages, tools):
        if not self.responses: raise ValueError("reference_exhausted")
        response = self.responses.pop(0)
        response["role"] = "assistant"
        for index, call in enumerate(response.get("tool_calls",[])):
            call["id"] = f"target_{len(self.samples):03d}_{index}"
        full = normalize_messages([*messages,response])
        self.samples.append({"id":f"{self.case['case_id']}_target_{len(self.samples):03d}","task_id":self.case["case_id"],
            "split":self.case["split"],"category":self.case["category"],"sample_kind":"assistant_prefix",
            "messages":full,"tools":canonicalize_tool_schemas(tools),"loss_mask":[False]*(len(full)-1)+[True],
            "source":"executed_synthetic_reference","real_customer_data":False,"teacher_is_llm":False})
        return response


def execute_reference(case: dict, directory: Path, counter=None) -> tuple[list[dict], dict]:
    if case["split"] not in {"train","dev"}: raise ValueError("sealed_test_is_not_replayed_for_supervision")
    reference = RecordingReference(case)
    ordinal = 0
    def identifier():
        nonlocal ordinal
        ordinal += 1
        return UUID(hex=sha256(f"{case['case_id']}/{ordinal}".encode()).hexdigest()[:32])
    context = ContextBuilder(counter=counter,max_input_tokens=8192 if counter else 32768)
    with patch("ecommerce_customer_service.serving.service.uuid4",side_effect=identifier), patch("time.time",return_value=1700000000.0):
        result = run_case(case,directory,reference,context,policy_profile="binding_compact_v3")
    if not result["passed"] or reference.responses:
        raise ValueError("reference_validation_failed:"+case["case_id"]+":"+str(result["failures"]))
    policy = get_policy("binding_compact_v3")
    for row in reference.samples:
        failures = validate_training_row(row)
        if failures: raise ValueError("invalid_reference_prefix:"+str(failures))
        # Only declared tools/parameters may be supervised, even when the source is synthetic.
        for call in row["messages"][-1].get("tool_calls",[]):
            function = call["function"]
            schema = next((tool["function"] for tool in policy.tools if tool["function"]["name"]==function["name"]),None)
            if not schema or set(function["arguments"]) != set(schema["parameters"]["required"]):
                raise ValueError("invalid_supervised_tool")
            if any(not isinstance(value,str) or not value.strip() for value in function["arguments"].values()):
                raise ValueError("invalid_supervised_argument")
    return reference.samples,result


def validate_tokenized_sample(row: dict, tokenizer) -> dict:
    ids, mask = render_with_assistant_mask(tokenizer,row["messages"],tools=row["tools"],loss_mask=row["loss_mask"])
    if len(ids)>8192: raise ValueError("repair_sample_exceeds_token_budget")
    if sum(mask)>512: raise ValueError("repair_target_exceeds_generation_budget")
    from ..local_hf import parse_qwen_response
    supervised = tokenizer.decode([token for token,target in zip(ids,mask) if target],skip_special_tokens=True)
    parsed = parse_qwen_response(supervised)
    expected = row["messages"][-1].get("tool_calls") or []
    actual = parsed.get("tool_calls") or []
    if (parsed.get("content") or "").strip() != (row["messages"][-1].get("content") or "").strip():
        raise ValueError("rendered_target_content_mismatch")
    if len(expected)!=len(actual): raise ValueError("rendered_target_tool_count_mismatch")
    for before,after in zip(expected,actual):
        if before["function"]["name"]!=after["function"]["name"] or before["function"]["arguments"]!=json.loads(after["function"]["arguments"]):
            raise ValueError("rendered_target_arguments_mismatch")
    return {"input_tokens":len(ids),"supervised_tokens":sum(mask),"native_tool_format_verified":True}
