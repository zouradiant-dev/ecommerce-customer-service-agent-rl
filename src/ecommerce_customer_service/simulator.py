from __future__ import annotations

from dataclasses import dataclass, replace
from http.client import RemoteDisconnected
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import RolloutRecord, TaskSpec, VerificationResult
from .rewards import score_verification
from .training import canonicalize_tool_schemas
from .tools import CustomerServiceTools
from .verifier import VERIFIER_VERSION, verify_task


SYSTEM_PROMPT = """你是电商售后客服Agent，必须通过工具完成业务操作，不能只用文字描述将要执行的动作。
如果用户尚未提供姓名和手机号后四位，明确向用户询问；信息已提供时立即调用身份核验工具。
身份核验成功后，只能使用工具返回的内部 customer_id，不能把姓名、订单号或其他字段当作 customer_id。
先查询订单、物流或退款政策，再依据工具结果决定下一步。修改地址、取消订单或退款必须获得用户明确确认。写工具在确认前会暂时隐藏；这不表示系统缺少该工具。查询完成但写工具尚未出现时，应询问用户确认原请求，不能因此升级人工。
没有成功的写工具结果时，不得声称已经取消、退款、修改或处理成功。只能调用当前轮实际提供的工具；未提供升级工具时不得请求升级。临时工具错误后应重试原工具，直到升级工具在后续轮次出现。
用户对订单号表示不确定或要求先不要操作时，只能列出订单并请用户确认准确订单号，不能替用户选择目标订单。
政策明确拒绝请求时应解释原因并停止，不得擅自创建人工工单。只有身份核验失败或工具反复失败时才能升级人工。
升级人工成功后，下一轮只能用文字说明失败原因和工单号并立即结束；绝对不能再输出任何 tool_call，也不得继续查询订单。
完成操作后，明确告知用户真实处理结果、金额和下一步。每轮只会提供当前状态允许使用的工具。"""


@dataclass(frozen=True)
class AgentRun:
    messages: tuple[dict[str, Any], ...]
    final_response: str
    stopped_reason: str = "completed"


class Agent(Protocol):
    def run(self, task: TaskSpec, tools: CustomerServiceTools) -> AgentRun: ...


class _ScriptedTranscript:
    """Build a teacher transcript in the same order that tools and user turns occur."""

    def __init__(self, task: TaskSpec, tools: CustomerServiceTools) -> None:
        self.tools = tools
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.user_prompt},
        ]
        self.replies = [dict(item) for item in task.metadata.get("scripted_user_replies") or []]
        self.call_index = 0

    def request_user(self, kind: str, prompt: str) -> bool:
        for index, reply in enumerate(self.replies):
            if reply.get("kind") != kind:
                continue
            content = str(reply.get("content") or "").strip()
            self.replies.pop(index)
            self.messages.extend(
                [
                    {"role": "assistant", "content": prompt},
                    {"role": "user", "content": content},
                ]
            )
            self.tools.observe_user_message(content)
            return True
        return False

    def call(self, name: str, arguments: dict[str, Any]):
        execution = self.tools.execute(name, arguments)
        call_id = f"call_{self.call_index}"
        self.call_index += 1
        self.messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": execution.name,
                                "arguments": json.dumps(execution.arguments, ensure_ascii=False),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": execution.name,
                    "content": json.dumps(execution.output, ensure_ascii=False),
                },
            ]
        )
        return execution

    def finish(self, final_response: str) -> AgentRun:
        self.messages.append({"role": "assistant", "content": final_response})
        return AgentRun(tuple(self.messages), final_response)


class ReferenceAgent:
    """Deterministic verified teacher used only for local integration and data-schema tests."""

    def run(self, task: TaskSpec, tools: CustomerServiceTools) -> AgentRun:
        meta = task.metadata
        transcript = _ScriptedTranscript(task, tools)
        transcript.request_user("identity", "为保护账户安全，请提供姓名和手机号后四位。")
        identity = dict(meta.get("presented_identity") or meta["identity"])
        verify = transcript.call(
            "verify_customer_identity",
            identity,
        )
        scenario = task.scenario
        order_id = str(meta.get("order_id") or "")
        item_id = str(meta.get("item_id") or "")

        if scenario == "identity_tool_transient_error" and not verify.ok:
            verify = transcript.call("verify_customer_identity", identity)

        if scenario == "identity_mismatch":
            transcript.call(
                "escalate_to_human",
                {"issue": "订单取消", "reason": "身份信息不一致"},
            )
            final = "身份信息不一致，我不能直接操作该订单，已经为您创建人工工单。"
        elif not verify.ok:
            final = "身份核验失败，我不能查询或修改订单。"
        elif scenario in {"shipment_status", "identity_tool_transient_error"}:
            shipment = transcript.call("get_shipment", {"customer_id": task.user_id, "order_id": order_id})
            value = shipment.output.get("shipment") or {}
            final = f"订单正在运输中，承运商为{value.get('carrier')}，物流单号 {value.get('tracking_number')}。"
        elif scenario == "latest_order_lookup":
            transcript.call("list_orders", {"customer_id": task.user_id})
            latest_id = str(meta["latest_order_id"])
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": latest_id})
            amount = int(meta["latest_amount_cents"]) / 100
            final = f"最新订单是 {latest_id}，尚未发货，实付金额 {amount:.2f} 元；我没有修改订单状态。"
        elif scenario in {"cancel_unshipped", "transient_tool_error"}:
            inspected = transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            if not inspected.ok:
                inspected = transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            if inspected.ok:
                amount = int(meta["paid_amount_cents"]) / 100
                transcript.request_user(
                    "confirmation",
                    f"订单尚未发货，可取消并原路退回 {amount:.2f} 元。请明确确认是否执行。",
                )
                transcript.call(
                    "cancel_order",
                    {"customer_id": task.user_id, "order_id": order_id, "confirmed": True},
                )
            amount = int(meta["paid_amount_cents"]) / 100
            final = f"订单已取消，{amount:.2f} 元将原路退款。"
        elif scenario == "multi_order_disambiguation":
            transcript.call("list_orders", {"customer_id": task.user_id})
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            amount = int(meta["paid_amount_cents"]) / 100
            transcript.request_user(
                "confirmation",
                f"已核对两笔订单，尚未发货且实付 {amount:.2f} 元的是 {order_id}。请确认是否取消。",
            )
            transcript.call(
                "cancel_order",
                {"customer_id": task.user_id, "order_id": order_id, "confirmed": True},
            )
            final = f"已核对订单并确认目标，订单已取消，{amount:.2f} 元将原路退款。"
        elif scenario == "cancel_without_captured_payment":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.request_user(
                "confirmation",
                "该订单只有支付预授权、未实际扣款，可以取消且无需退款。请明确确认。",
            )
            transcript.call(
                "cancel_order",
                {"customer_id": task.user_id, "order_id": order_id, "confirmed": True},
            )
            final = "订单已取消；由于未实际扣款，因此无需退款。"
        elif scenario == "cancel_already_cancelled":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            final = "该订单已经取消并且已经退款，我没有重复操作。"
        elif scenario == "mixed_order_bulk_cancel":
            shipped_order_id = str(meta["shipped_order_id"])
            transcript.call("list_orders", {"customer_id": task.user_id})
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": shipped_order_id})
            transcript.call(
                "cancel_order",
                {"customer_id": task.user_id, "order_id": order_id, "confirmed": True},
            )
            final = (
                f"订单 {order_id} 已取消；订单 {shipped_order_id} 已发货，无法取消，"
                "我没有对已发货订单执行写操作。"
            )
        elif scenario == "persistent_tool_error_escalation":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.call(
                "escalate_to_human",
                {"issue": "订单取消", "reason": "订单查询服务持续异常"},
            )
            final = "订单查询服务持续异常，我没有猜测取消结果，已经为您创建人工工单。"
        elif scenario == "update_address":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            address = str(meta["target_address"])
            transcript.request_user(
                "confirmation",
                f"订单尚未发货，可以把地址修改为{address}。请明确确认是否执行。",
            )
            transcript.call(
                "update_shipping_address",
                {"customer_id": task.user_id, "order_id": order_id, "new_address": address, "confirmed": True},
            )
            final = f"收货地址已更新为{address}。"
        elif scenario == "address_change_same_value":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            address = str(meta["target_address"])
            final = f"当前地址已经是{address}，无需修改；我没有重复写入。"
        elif scenario == "address_change_after_shipment":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            final = "订单已发货，无法修改地址；我没有执行地址变更。"
        elif scenario == "cancel_after_shipment":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            final = "订单已发货，无法取消；我没有创建退款。"
        elif scenario in {"damaged_item_refund", "refund_tool_transient_error", "refund_policy_tool_transient_error"}:
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            policy = transcript.call(
                "search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id}
            )
            if scenario == "refund_policy_tool_transient_error" and not policy.ok:
                policy = transcript.call(
                    "search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id}
                )
            if policy.output.get("allowed"):
                amount = int(meta["refund_amount_cents"]) / 100
                transcript.request_user(
                    "confirmation",
                    f"该商品符合退货政策，可按实付金额 {amount:.2f} 元原路退款。请明确确认是否执行。",
                )
                refunded = transcript.call(
                    "create_refund",
                    {
                        "customer_id": task.user_id,
                        "order_id": order_id,
                        "order_item_id": item_id,
                        "quantity": 1,
                        "reason": "商品损坏",
                        "confirmed": True,
                    },
                )
                if scenario == "refund_tool_transient_error" and not refunded.ok:
                    transcript.call(
                        "create_refund",
                        {
                            "customer_id": task.user_id,
                            "order_id": order_id,
                            "order_item_id": item_id,
                            "quantity": 1,
                            "reason": "商品损坏，首次退款请求超时后重试",
                            "confirmed": True,
                        },
                    )
            amount = int(meta["refund_amount_cents"]) / 100
            final = f"已按实付金额 {amount:.2f} 元创建退货并原路退款。"
        elif scenario == "partial_quantity_refund":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            policy = transcript.call(
                "search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id}
            )
            amount = int(meta["refund_amount_cents"]) / 100
            if policy.output.get("allowed"):
                transcript.request_user(
                    "confirmation",
                    f"仅退1件可原路退回 {amount:.2f} 元，另1件保留。请明确确认。",
                )
                transcript.call(
                    "create_refund",
                    {
                        "customer_id": task.user_id,
                        "order_id": order_id,
                        "order_item_id": item_id,
                        "quantity": 1,
                        "reason": "两件中一件损坏",
                        "confirmed": True,
                    },
                )
            final = f"已按单件实付金额 {amount:.2f} 元原路退款，订单中剩余1件未退。"
        elif scenario == "full_quantity_refund":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            policy = transcript.call(
                "search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id}
            )
            amount = int(meta["refund_amount_cents"]) / 100
            if policy.output.get("allowed"):
                transcript.request_user(
                    "confirmation",
                    f"全部2件符合政策，退款总额 {amount:.2f} 元。请明确确认。",
                )
                transcript.call(
                    "create_refund",
                    {
                        "customer_id": task.user_id,
                        "order_id": order_id,
                        "order_item_id": item_id,
                        "quantity": 2,
                        "reason": "两件商品均有故障",
                        "confirmed": True,
                    },
                )
            final = f"已退回全部2件，并按实付总额 {amount:.2f} 元原路退款。"
        elif scenario == "refund_quantity_exceeds":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.call("search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id})
            final = "订单记录显示只购买了1件，不能按2件退款；我没有创建超额退款。"
        elif scenario == "negative_confirmation":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.call("search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id})
            transcript.request_user(
                "confirmation",
                "该商品符合政策，请明确确认是否创建退货退款。",
            )
            final = "收到，已停止操作，没有创建退货或退款。"
        elif scenario == "late_refund":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.call("search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id})
            final = "该商品已超过30天退货期限，当前不能直接创建退款。"
        elif scenario == "refund_before_delivery":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.call("search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id})
            final = "订单尚未签收，不能创建售后退货；我没有提前退款。"
        elif scenario == "duplicate_refund_request":
            order = transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            refunds = order.output.get("refunds") or []
            if refunds:
                final = "查询到该商品已经退款，不能重复退款；我没有再创建新的退款。"
            else:
                final = "没有查到已有退款记录，需要进一步核对。"
        elif scenario == "cross_order_item_mismatch":
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.call(
                "search_refund_policy",
                {"customer_id": task.user_id, "order_item_id": str(meta["mismatched_item_id"])},
            )
            final = "核对发现商品项不属于指定订单，订单与商品参数不一致；我没有执行退款。"
        elif scenario in {"final_sale_refund", "policy_bypass_request"}:
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            transcript.call("search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id})
            final = "根据退款政策，定制商品不支持退货，不能绕过政策直接退款。"
        elif scenario == "contradictory_order":
            transcript.call("list_orders", {"customer_id": task.user_id})
            final = "您提供的信息存在矛盾，请确认准确订单号后我再操作。"
        elif scenario == "order_not_found":
            transcript.call("list_orders", {"customer_id": task.user_id})
            requested = str(meta["requested_order_id"])
            final = f"您的订单列表中未找到订单 {requested}，我没有执行任何操作。"
        else:
            final = "暂时无法处理该请求。"
        return transcript.finish(final)


class WeakBaselineAgent:
    """A deliberately conservative non-LLM baseline for verifying the local experiment harness."""

    def run(self, task: TaskSpec, tools: CustomerServiceTools) -> AgentRun:
        transcript = _ScriptedTranscript(task, tools)
        transcript.request_user("identity", "请提供姓名和手机号后四位。")
        identity = dict(task.metadata.get("presented_identity") or task.metadata["identity"])
        transcript.call("verify_customer_identity", identity)
        order_id = str(task.metadata.get("order_id") or "")
        item_id = str(task.metadata.get("item_id") or "")
        if task.scenario == "shipment_status":
            result = transcript.call("get_shipment", {"customer_id": task.user_id, "order_id": order_id})
            shipment = result.output.get("shipment") or {}
            final = f"订单正在运输中，物流单号 {shipment.get('tracking_number')}。"
        elif task.scenario in {"late_refund", "final_sale_refund", "policy_bypass_request"}:
            transcript.call("get_order", {"customer_id": task.user_id, "order_id": order_id})
            policy = transcript.call("search_refund_policy", {"customer_id": task.user_id, "order_item_id": item_id})
            final = str(policy.output.get("message") or "当前不能退款。")
            if task.scenario in {"final_sale_refund", "policy_bypass_request"}:
                final = "定制商品不支持退货。"
        elif task.scenario == "contradictory_order":
            transcript.call("list_orders", {"customer_id": task.user_id})
            final = "请确认准确订单号。"
        elif task.scenario == "identity_mismatch":
            final = "身份信息不一致，无法操作。"
        else:
            final = "我已经记录您的请求，但暂时无法完成操作。"
        return transcript.finish(final)


class OpenAICompatibleAgent:
    def __init__(
        self,
        model: str,
        *,
        api_base: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_steps: int = 20,
        max_tokens: int = 512,
        enable_thinking: bool | None = None,
        correction_retries: int = 3,
        timeout_seconds: int = 180,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
        self.temperature = temperature
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        model_name = self.model.rsplit("/", 1)[-1].lower()
        self.enable_thinking = False if enable_thinking is None and model_name.startswith("qwen3.7-plus") else enable_thinking
        self.correction_retries = correction_retries
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    @staticmethod
    def _parse_sse_message(body: str) -> dict[str, Any]:
        """Merge OpenAI chat-completion SSE deltas into one assistant message."""
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        tool_calls: dict[int, dict[str, Any]] = {}
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if not payload or payload == "[DONE]":
                continue
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(f"model_stream_error:{event['error']}")
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or choices[0].get("message") or {}
            if delta.get("role"):
                message["role"] = str(delta["role"])
            if delta.get("content") is not None:
                message["content"] += str(delta["content"])
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index", len(tool_calls)))
                target = tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call.get("id"):
                    target["id"] += str(call["id"])
                if call.get("type"):
                    target["type"] = str(call["type"])
                function = call.get("function") or {}
                if function.get("name"):
                    target["function"]["name"] += str(function["name"])
                if function.get("arguments") is not None:
                    arguments = function["arguments"]
                    target["function"]["arguments"] += (
                        arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
                    )
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        return message

    def _request_body(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if tool_schemas:
            request_body["tools"] = canonicalize_tool_schemas(tool_schemas)
            request_body["tool_choice"] = "auto"
        else:
            request_body["tool_choice"] = "none"
        if self.enable_thinking is not None:
            request_body["enable_thinking"] = self.enable_thinking
        return request_body

    def _complete(self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]) -> dict[str, Any]:
        request_body = self._request_body(messages, tool_schemas)
        payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.api_base}/chat/completions",
            data=payload,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type or body.lstrip().startswith("data:"):
                    return self._parse_sse_message(body)
                value = json.loads(body)
                return dict(value["choices"][0]["message"])
            except (
                HTTPError,
                URLError,
                TimeoutError,
                RemoteDisconnected,
                ConnectionResetError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(8.0, 2**attempt))
        raise RuntimeError(f"model_request_failed:{last_error}")

    @staticmethod
    def _assistant_requests_reply(kind: str, content: str) -> bool:
        normalized = content.replace(" ", "")
        if kind == "identity":
            fields = any(marker in normalized for marker in ("姓名", "名字", "手机号", "手机号码", "后四位"))
            request = any(marker in normalized for marker in ("请提供", "请告诉", "需要您提供", "麻烦提供"))
            return fields and request
        if kind == "confirmation":
            confirmation = any(marker in normalized for marker in ("确认", "确定"))
            request = any(
                marker in normalized
                for marker in (
                    "请确认",
                    "请您确认",
                    "您确认",
                    "确认要",
                    "确认是否",
                    "是否确定",
                    "是否确认",
                    "需要您确认",
                    "再次确认",
                    "回复",
                )
            )
            return confirmation and request
        return False

    @staticmethod
    def _correction_for_state(
        tools: CustomerServiceTools,
        scripted_replies: list[dict[str, Any]],
    ) -> tuple[str, str] | None:
        successful = {item.name for item in tools.executions if item.ok}
        if tools.verified_customer_id is None:
            return (
                "identity",
                "你尚未通过工具核验身份。若用户已提供姓名和手机号后四位，请立即调用当前可用的核验工具；"
                "若信息缺失，请明确要求用户提供。不要只说请稍候。",
            )
        if "escalate_to_human" in successful:
            return None
        available = {item["function"]["name"] for item in tools.available_schemas()}
        required_reads = {
            name
            for name in tools.task.required_tools
            if name not in {"cancel_order", "update_shipping_address", "create_refund", "escalate_to_human"}
        }
        missing_required_reads = sorted((required_reads - successful) & available)
        if missing_required_reads:
            return (
                "required_read:" + ",".join(missing_required_reads),
                "任务尚未完成。请立即调用这些当前可用的必要读取工具："
                + "、".join(missing_required_reads)
                + "。用户若已在原请求中明确确认操作，不要重复询问确认，也不要升级人工。",
            )
        successful_reads = successful & {"list_orders", "get_order", "get_shipment", "search_refund_policy"}
        if not successful_reads:
            return (
                "read",
                "身份已经通过工具核验。用户请求尚未查询，请立即调用当前可用的读取工具获取真实订单、物流或政策信息，"
                "不要只描述计划。",
            )
        if scripted_replies and scripted_replies[0].get("kind") == "confirmation":
            requested_actions = {
                "cancel_order": "取消订单",
                "update_shipping_address": "修改收货地址",
                "create_refund": "执行退货退款",
            }
            action_text = "或".join(
                requested_actions[name]
                for name in tools.task.allowed_write_tools
                if name in requested_actions
            ) or "原业务操作"
            return (
                "confirmation",
                f"写工具会在用户确认前暂时隐藏，这不表示系统缺少工具。现在只需明确询问用户是否确认{action_text}；"
                "不要改问是否升级人工，不要继续查询其他无关工具，也不得声称已经执行。",
            )
        successful_writes = successful & {"cancel_order", "update_shipping_address", "create_refund"}
        if available & {"cancel_order", "update_shipping_address", "create_refund"} and not successful_writes:
            return (
                "write",
                "用户已经明确确认且前置查询已完成。请调用当前可用的写工具实际执行操作；"
                "没有成功的工具结果时不得声称完成。",
            )
        return None

    def run(self, task: TaskSpec, tools: CustomerServiceTools) -> AgentRun:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.user_prompt},
        ]
        scripted_replies = [dict(item) for item in task.metadata.get("scripted_user_replies") or []]
        final = ""
        corrections_used: set[str] = set()
        for step in range(self.max_steps):
            assistant = self._complete(messages, tools.available_schemas())
            assistant["role"] = "assistant"
            messages.append(assistant)
            calls = assistant.get("tool_calls") or []
            if not calls:
                final = str(assistant.get("content") or "")
                reply_index = next(
                    (
                        index
                        for index, reply in enumerate(scripted_replies)
                        if self._assistant_requests_reply(str(reply.get("kind") or ""), final)
                    ),
                    None,
                )
                if reply_index is not None:
                    reply = scripted_replies.pop(reply_index)
                    content = str(reply.get("content") or "").strip()
                    messages.append({"role": "user", "content": content})
                    tools.observe_user_message(content)
                    continue
                correction = self._correction_for_state(tools, scripted_replies)
                if correction and correction[0] not in corrections_used and len(corrections_used) < self.correction_retries:
                    corrections_used.add(correction[0])
                    messages.append({"role": "system", "content": correction[1]})
                    continue
                return AgentRun(tuple(messages), final, "model_finished")
            for index, call in enumerate(calls):
                function = call.get("function") or {}
                name = str(function.get("name") or call.get("name") or "")
                arguments = function.get("arguments") or call.get("arguments") or "{}"
                execution = tools.execute(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or f"call_{step}_{index}"),
                        "name": name,
                        "content": json.dumps(execution.output, ensure_ascii=False),
                    }
                )
        return AgentRun(tuple(messages), final, "max_steps")


def run_task(task: TaskSpec, agent: Agent, *, trial: int = 0) -> RolloutRecord:
    with tempfile.TemporaryDirectory(prefix="ecommerce-cs-") as temporary:
        trial_db = Path(temporary) / "state.sqlite"
        shutil.copy2(task.database_path, trial_db)
        isolated_task = replace(task, database_path=trial_db)
        tools = CustomerServiceTools(trial_db, isolated_task)
        tools.observe_user_message(task.user_prompt)
        try:
            run = agent.run(isolated_task, tools)
            execution_rows = tuple(item.to_dict() for item in tools.executions)
            verification = verify_task(isolated_task, run.final_response, execution_rows)
            reward_components = score_verification(verification)
            return RolloutRecord(
                task_id=task.task_id,
                split=task.split,
                scenario=task.scenario,
                trial=trial,
                reward=reward_components.strict_binary,
                messages=run.messages,
                tool_executions=execution_rows,
                final_response=run.final_response,
                verifier=verification,
                verifier_version=VERIFIER_VERSION,
                reward_components=reward_components.to_dict(),
                stopped_reason=run.stopped_reason,
            )
        except Exception as exc:
            verification = VerificationResult(False, False, False, False, False, (f"worker_error:{type(exc).__name__}",))
            reward_components = score_verification(verification)
            return RolloutRecord(
                task_id=task.task_id,
                split=task.split,
                scenario=task.scenario,
                trial=trial,
                reward=0.0,
                messages=(),
                tool_executions=tuple(item.to_dict() for item in tools.executions),
                final_response="",
                verifier=verification,
                verifier_version=VERIFIER_VERSION,
                reward_components=reward_components.to_dict(),
                stopped_reason="worker_error",
                error=f"{type(exc).__name__}: {exc}",
            )


def collect_rollouts(
    task_paths: list[Path],
    agent_factory: Any,
    output_path: Path,
    *,
    trials_per_task: int = 1,
    verbose: bool = False,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempted = 0
    successes = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for task_path in task_paths:
            task = TaskSpec.from_file(task_path)
            for trial in range(trials_per_task):
                attempted += 1
                agent = agent_factory(task, trial)
                rollout = run_task(task, agent, trial=trial)
                successes += int(rollout.reward >= 1.0)
                handle.write(json.dumps(rollout.to_dict(), ensure_ascii=False) + "\n")
                if verbose:
                    print(f"\n=== {rollout.task_id} | {rollout.scenario} | trial={trial} ===")
                    for message in rollout.messages:
                        role = str(message.get("role") or "unknown")
                        calls = message.get("tool_calls") or []
                        if calls:
                            for call in calls:
                                function = call.get("function") or {}
                                print(f"[{role}/tool-call] {function.get('name')} {function.get('arguments')}")
                        elif message.get("content"):
                            print(f"[{role}] {message['content']}")
                    print(f"[verifier] reward={rollout.reward} failures={list(rollout.verifier.failures)}")
    return {"attempted": attempted, "successes": successes}
