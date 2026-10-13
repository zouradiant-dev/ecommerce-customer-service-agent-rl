"""Single-GPU Qwen/PEFT inference transport. Contains no business tool executor."""
from __future__ import annotations

from copy import deepcopy
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any
from uuid import uuid4

from ..training import normalize_messages
from .artifacts import BASE_MODEL, BASE_REVISION, FILE_HASHES, MODEL_ID
from .model import online_protocol_metadata


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address: tuple[str, int], generator: Any, *, api_key: str = "", precision: str = "bf16"):
        self.generator, self.api_key, self.precision = generator, api_key, precision
        self.inference_lock = threading.Lock()
        super().__init__(address, InferenceHandler)

    def metadata(self) -> dict:
        return {"ready": True, "model_id": MODEL_ID, "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION, "adapter_sha256": FILE_HASHES["adapter_model.safetensors"],
                "tokenizer_sha256": FILE_HASHES["tokenizer.json"], "precision": self.precision,
                "tokenizer_config_sha256": FILE_HASHES["tokenizer_config.json"],
                "transport": "openai_chat_completions", "tool_parser": "strict_qwen_native_tool_tags_v1",
                "training_protocol": "v18_offline_nine_tools_terminal_json",
                "online_quality_verified": False, **online_protocol_metadata()}


class InferenceHandler(BaseHTTPRequestHandler):
    server: InferenceServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *args: Any) -> None:
        pass

    def reply(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def authorized(self) -> bool:
        return not self.server.api_key or hmac.compare_digest(
            self.headers.get("Authorization", "").encode("utf-8"),
            ("Bearer " + self.server.api_key).encode("utf-8"))

    def do_GET(self) -> None:
        if not self.authorized():
            self.reply(401, {"error": "authentication_required"})
        elif self.path == "/healthz":
            self.reply(200, self.server.metadata())
        elif self.path == "/v1/models":
            self.reply(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
        else:
            self.reply(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if not self.authorized():
            self.reply(401, {"error": "authentication_required"})
            return
        if self.path != "/v1/chat/completions":
            self.reply(404, {"error": "not_found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 262144:
                raise ValueError("invalid_request_size")
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("incomplete_request")
            payload = json.loads(raw)
            allowed = {"model", "messages", "tools", "tool_choice", "temperature", "max_tokens", "stream"}
            if not isinstance(payload, dict) or set(payload) - allowed:
                raise ValueError("unsupported_parameters")
            if payload.get("model") != MODEL_ID:
                raise ValueError("model_id_mismatch")
            if payload.get("stream", False) is not False or payload.get("temperature", 0) != 0:
                raise ValueError("only_nonstreaming_greedy_generation_supported")
            if payload.get("tool_choice", "auto") != "auto":
                raise ValueError("only_auto_tool_choice_supported")
            max_tokens = payload.get("max_tokens", 512)
            if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
                raise ValueError("invalid_max_tokens")
            messages, tools = payload.get("messages"), payload.get("tools", [])
            if not isinstance(messages, list) or not 1 <= len(messages) <= 200:
                raise ValueError("invalid_messages")
            if any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant", "tool"}
                   or not isinstance(m.get("content", "") or "", str)
                   for m in messages):
                raise ValueError("invalid_message_role")
            messages = normalize_messages(messages)
            if not isinstance(tools, list) or len(tools) > 16:
                raise ValueError("invalid_tools")
            for tool in tools:
                if (not isinstance(tool, dict) or tool.get("type") != "function"
                        or not isinstance(tool.get("function"), dict)
                        or not isinstance(tool["function"].get("name"), str)
                        or not isinstance(tool["function"].get("parameters"), dict)):
                    raise ValueError("invalid_tool_schema")
        except (ValueError, TypeError, KeyError, AttributeError):
            self.reply(400, {"error": "invalid_inference_request"})
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.reply(429, {"error": "inference_busy; retry as a new user request"})
            return
        try:
            # Completer verifies exact template length before generation, without truncation.
            result = self.server.generator(messages, tools, max_tokens)
            finish = result.get("local_finish_reason", "stop")
            if finish == "length":
                self.reply(422, {"error": "generation_truncated; output_not_executed"})
                return
            message = {k: result[k] for k in ("role", "content", "tool_calls") if k in result}
            prompt = result["local_input_tokens"]
            completion = result["local_generated_tokens"]
            self.reply(200, {"id": "chatcmpl_" + uuid4().hex, "object": "chat.completion", "model": MODEL_ID,
                            "choices": [{"index": 0, "message": message,
                                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                      "total_tokens": prompt + completion}})
        except Exception as exc:
            # Return no prompt, raw generation, internal paths or credentials in errors.
            print("Inference request failed:", type(exc).__name__, flush=True)
            self.reply(422, {"error": "inference_or_tool_parse_failed"})
        finally:
            self.server.inference_lock.release()


class BoundedGenerator:
    def __init__(self, completer: Any):
        self.completer = completer

    def __call__(self, messages: list, tools: list, max_tokens: int) -> dict:
        config = deepcopy(self.completer.generation_config)
        config.max_new_tokens = max_tokens
        previous = self.completer.generation_config
        self.completer.generation_config = config
        try:
            return self.completer(messages, tools)
        finally:
            self.completer.generation_config = previous
