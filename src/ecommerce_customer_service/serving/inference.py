"""Single-GPU Qwen/PEFT inference transport. Contains no business tool executor."""
from __future__ import annotations

from copy import deepcopy
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..training import normalize_messages
from .artifacts import BASE_MODEL, BASE_REVISION, FILE_HASHES, MODEL_ID
from .model import online_protocol_metadata
from .diagnostics import DIAGNOSTICS_VERSION, DiagnosticStore, classify_failure


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address: tuple[str, int], generator: Any, *, api_key: str = "", precision: str = "bf16",
                 diagnostics_dir: Path | None = None, capture_invalid_output: bool = False,
                 repair_experiment_report: dict | None = None):
        self.generator, self.api_key, self.precision = generator, api_key, precision
        self.repair_experiment_report = repair_experiment_report
        if capture_invalid_output and diagnostics_dir is None:
            raise ValueError("raw_capture_requires_private_diagnostics_directory")
        self.diagnostics = DiagnosticStore(diagnostics_dir, capture_raw=capture_invalid_output) if diagnostics_dir else None
        self.inference_lock = threading.Lock()
        super().__init__(address, InferenceHandler)

    def metadata(self) -> dict:
        return {"ready": True, "model_id": MODEL_ID, "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION, "adapter_sha256": FILE_HASHES["adapter_model.safetensors"],
                "tokenizer_sha256": FILE_HASHES["tokenizer.json"], "precision": self.precision,
                "tokenizer_config_sha256": FILE_HASHES["tokenizer_config.json"],
                "transport": "openai_chat_completions", "tool_parser": "strict_qwen_native_tool_tags_v1",
                "training_protocol": "v18_offline_nine_tools_terminal_json",
                "diagnostics_version": DIAGNOSTICS_VERSION,
                "diagnostics_enabled": self.diagnostics is not None,
                "raw_failure_capture_enabled": bool(self.diagnostics and self.diagnostics.capture_raw),
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
        elif self.path == "/v1/repair-experiment" and self.server.repair_experiment_report is not None:
            self.reply(200, self.server.repair_experiment_report)
        elif self.path.startswith("/v1/diagnostics/") and self.server.diagnostics:
            try:
                value = self.server.diagnostics.read(self.path.removeprefix("/v1/diagnostics/"))
                self.reply(200, value)
            except (ValueError, FileNotFoundError):
                self.reply(404, {"error": "diagnostic_not_found"})
        else:
            self.reply(404, {"error": "not_found"})

    def failure(self, status: int, code: str, exception_type: str,
                metadata: dict | None = None, raw: str | None = None, reason: str | None = None) -> None:
        identifier = None
        if self.server.diagnostics:
            try:
                identifier = self.server.diagnostics.record(code=code, exception_type=exception_type,
                    metadata=metadata, raw=raw, parse_reason=reason)
            except Exception as exc:
                print("Diagnostic persistence failed:", type(exc).__name__, flush=True)
        # Raw generation is available only through authenticated diagnostic retrieval.
        detail = {"code": code, "diagnostic_id": identifier, "parse_reason": reason}
        for source, target in (("local_input_tokens", "input_tokens"), ("local_generated_tokens", "output_tokens")):
            value = (metadata or {}).get(source)
            if type(value) is int and value >= 0:
                detail[target] = value
        print("Inference request failed:", code, identifier or "not_recorded", flush=True)
        self.reply(status, {"error": detail})

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
                self.failure(422, "generation_truncated", "GenerationLimit", result,
                             result.get("local_raw_generation"))
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
            status, code, reason, metadata, raw = classify_failure(exc)
            self.failure(status, code, type(exc).__name__, metadata, raw, reason)
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
