"""Private, bounded inference failure records. Raw capture is explicit opt-in."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

OUTPUT_ERRORS = frozenset({"invalid_tool_call_output", "generation_truncated"})
DIAGNOSTICS_VERSION = "inference_diagnostics_v1"


class DiagnosticStore:
    def __init__(self, directory: Path, *, capture_raw: bool = False):
        self.directory, self.capture_raw = Path(directory), capture_raw
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            self.directory.chmod(0o700)

    def record(self, *, code: str, exception_type: str, metadata: dict[str, Any] | None = None,
               raw: str | None = None, parse_reason: str | None = None) -> str:
        identifier = uuid4().hex
        result: dict[str, Any] = {"schema_version": DIAGNOSTICS_VERSION, "id": identifier,
            "code": code, "exception_type": exception_type, "parse_reason": parse_reason,
            "raw_capture_enabled": self.capture_raw, "input_tokens": None, "output_tokens": None,
            "finish_reason": None, "raw_generation_captured": False}
        metadata = metadata or {}
        for source, target in (("local_input_tokens", "input_tokens"), ("local_generated_tokens", "output_tokens")):
            value = metadata.get(source)
            if type(value) is int and value >= 0:
                result[target] = value
        if metadata.get("local_finish_reason") in {"stop", "length"}:
            result["finish_reason"] = metadata["local_finish_reason"]
        if isinstance(raw, str) and self.capture_raw:
            result.update(raw_generation=raw[:16384], raw_generation_captured=True,
                raw_generation_sha256=sha256(raw.encode("utf-8")).hexdigest(),
                raw_generation_chars=len(raw), raw_capture_truncated=len(raw) > 16384)
        path = self.directory / (identifier + ".json")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, allow_nan=False, indent=2)
            output.write("\n")
        return identifier

    def read(self, identifier: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("invalid_diagnostic_id")
        path = self.directory / (identifier + ".json")
        if path.is_symlink():
            raise ValueError("linked_diagnostic_not_allowed")
        return json.loads(path.read_text(encoding="utf-8"))


def classify_failure(error: Exception) -> tuple[int, str, str | None, dict[str, Any], str | None]:
    from ..local_hf import ContextBudgetExceeded, ToolCallParseError
    if isinstance(error, ContextBudgetExceeded):
        return 413, "context_budget_exceeded", None, {"local_input_tokens": error.input_tokens}, None
    if isinstance(error, ToolCallParseError):
        metadata = error.generation_metadata
        if metadata.get("local_finish_reason") == "length":
            return 422, "generation_truncated", None, metadata, error.raw_text
        # Do not include arbitrary generated JSON keys/values in the public reason.
        reason = str(error).split(":", 1)[0]
        allowed = {"unbalanced_tool_call_tags", "invalid_tool_call_json", "missing_tool_call_closing_tag",
                   "tool_call_requires_name_and_arguments", "tool_name_must_be_nonempty_string", "tool_arguments_must_be_object"}
        return 422, "invalid_tool_call_output", reason if reason in allowed else "invalid_tool_format", metadata, error.raw_text
    if type(error).__name__ == "OutOfMemoryError" or "cuda out of memory" in str(error).lower():
        return 503, "cuda_out_of_memory", None, {}, None
    return 502, "inference_failed", None, {}, None
