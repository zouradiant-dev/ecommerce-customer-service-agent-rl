"""Explicit model identity/protocol checks, separate from online task evaluation."""
from __future__ import annotations

import json
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .artifacts import BASE_REVISION, FILE_HASHES, MODEL_ID
from .model import OpenAIChatModel, SYSTEM_PROMPT, online_protocol_metadata, terminal_content, tool_schemas


def check_endpoint(base_url: str, api_key: str = "", *, generate: bool = False) -> dict:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("invalid_api_base")
    if parsed.path.rstrip("/") != "/v1":
        raise ValueError("api_base_requires_v1_path")
    health_url = base_url.rstrip("/")[:-3] + "/healthz"
    request = Request(health_url, headers={"Authorization": "Bearer " + api_key})
    with urlopen(request, timeout=15) as response:
        raw = response.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError("health_response_too_large")
    health = json.loads(raw)
    expected = {"model_id": MODEL_ID, "base_revision": BASE_REVISION,
                "adapter_sha256": FILE_HASHES["adapter_model.safetensors"],
                "tokenizer_sha256": FILE_HASHES["tokenizer.json"],
                "tokenizer_config_sha256": FILE_HASHES["tokenizer_config.json"],
                "tool_parser": "strict_qwen_native_tool_tags_v1", **online_protocol_metadata()}
    if not isinstance(health, dict) or health.get("ready") is not True:
        raise ValueError("model_service_not_ready")
    if any(health.get(key) != value for key, value in expected.items()):
        raise ValueError("model_identity_or_online_protocol_mismatch")
    report = {"schema_version": "v18_connection_check_v1", "service_identity_checked": True,
              "generation_checked": False, "online_quality_verified": False,
              "precision": health.get("precision"), **expected}
    if generate:
        result = OpenAIChatModel(base_url, MODEL_ID, api_key).complete(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": "请查询我的订单。"}], tool_schemas())
        calls = result.get("tool_calls") or []
        if calls:
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError("probe_requires_one_list_orders_call")
            function = calls[0].get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if function.get("name") != "list_orders" or arguments != {}:
                raise ValueError("probe_tool_protocol_mismatch")
        else:
            # A polite text answer proves generation but does not pass the tool probe.
            terminal_content(result.get("content") or "")
            raise ValueError("probe_did_not_call_list_orders")
        report.update(generation_checked=True, tool_probe_passed=True,
                      usage=result.get("_provider_metadata", {}))
    return report
