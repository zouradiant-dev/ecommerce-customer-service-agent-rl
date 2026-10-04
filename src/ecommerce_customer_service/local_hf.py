"""Local Qwen generation using the existing agent loop and isolated tool environment.

Heavy dependencies are imported by the executable, not this module. Invalid tool
JSON is a policy-output failure, not an infrastructure/worker error. No JSON
repair, oracle response, or ReferenceAgent fallback is used.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Callable

from .models import TaskSpec
from .simulator import AgentRun, OpenAICompatibleAgent
from .training import canonicalize_tool_schemas, normalize_messages


class ToolCallParseError(ValueError):
    def __init__(self, reason: str, raw_text: str) -> None:
        super().__init__(reason)
        self.raw_text = raw_text
        self.messages: list[dict[str, Any]] = []
        self.generation_metadata: dict[str, Any] = {}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non_json_constant:{value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non_finite_json_number:{value}")
    return result


def parse_qwen_response(text: str, *, call_prefix: str = "call_local") -> dict[str, Any]:
    """Parse native <tool_call> JSON without treating tags inside JSON as delimiters."""
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_float=_finite_float)
    opening, closing = "<tool_call>", "</tool_call>"
    position = 0
    content: list[str] = []
    calls: list[dict[str, Any]] = []
    while (start := text.find(opening, position)) >= 0:
        prefix = text[position:start]
        if "<tool_call" in prefix or "</tool_call" in prefix:
            raise ToolCallParseError("unbalanced_tool_call_tags", text)
        content.append(prefix)
        body_start = start + len(opening)
        while body_start < len(text) and text[body_start].isspace():
            body_start += 1
        try:
            value, consumed = decoder.raw_decode(text[body_start:])
        except ValueError as exc:
            raise ToolCallParseError(f"invalid_tool_call_json:{exc}", text) from exc
        end = body_start + consumed
        while end < len(text) and text[end].isspace():
            end += 1
        if not text.startswith(closing, end):
            raise ToolCallParseError("missing_tool_call_closing_tag", text)
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ToolCallParseError("tool_call_requires_name_and_arguments", text)
        if not isinstance(value["name"], str) or not value["name"].strip():
            raise ToolCallParseError("tool_name_must_be_nonempty_string", text)
        if not isinstance(value["arguments"], dict):
            raise ToolCallParseError("tool_arguments_must_be_object", text)
        calls.append({
            "id": f"{call_prefix}_{len(calls)}",
            "type": "function",
            "function": {
                "name": value["name"],
                "arguments": json.dumps(value["arguments"], ensure_ascii=False),
            },
        })
        position = end + len(closing)
    tail = text[position:]
    if "<tool_call" in tail or "</tool_call" in tail:
        raise ToolCallParseError("unbalanced_tool_call_tags", text)
    content.append(tail)
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content).strip()}
    if calls:
        message["tool_calls"] = calls
    return message


class LocalTransformersAgent(OpenAICompatibleAgent):
    """Reuse the existing native loop; never make an HTTP request."""

    def __init__(self, completer: Callable[..., dict[str, Any]], *, max_steps: int = 20,
                 system_prompt: str | None = None) -> None:
        super().__init__("local-transformers", api_key="unused", max_steps=max_steps,
                         correction_retries=0, system_prompt=system_prompt)
        self.completer = completer

    def _complete(self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            return self.completer(messages, tool_schemas)
        except ToolCallParseError as exc:
            exc.messages = [*messages, {
                "role": "assistant", "content": exc.raw_text,
                "local_parse_error": str(exc), **exc.generation_metadata,
            }]
            raise

    def run(self, task: TaskSpec, tools: Any) -> AgentRun:
        try:
            return super().run(task, tools)
        except ToolCallParseError as exc:
            # Unknown-tool execution is guaranteed to fail verification. This
            # sentinel is NOT a callable business operation and ends the run.
            tools.execute("__invalid_tool_call__", {})
            return AgentRun(tuple(exc.messages), exc.raw_text, "invalid_tool_call")


class TransformersCompleter:
    def __init__(
        self, model: Any, tokenizer: Any, torch_module: Any, generation_config: Any,
        *, device: str, max_input_tokens: int = 8192,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.generation_config = generation_config
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.call_counter = 0

    def seed(self, value: int) -> None:
        random.seed(value)
        self.torch.manual_seed(value)
        self.torch.cuda.manual_seed_all(value)
        self.call_counter = 0

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self.tokenizer.apply_chat_template(
            normalize_messages(messages), tools=canonicalize_tool_schemas(tools),
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(inputs["input_ids"].shape[1])
        if input_tokens > self.max_input_tokens:
            raise RuntimeError(f"context_budget_exceeded:{input_tokens}>{self.max_input_tokens}; no truncation applied")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, generation_config=self.generation_config)
        generated = result[0, input_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        truncated = bool(
            len(generated) >= self.generation_config.max_new_tokens
            and int(generated[-1]) != self.tokenizer.eos_token_id
        )
        metadata = {
            "local_raw_generation": text,
            "local_input_tokens": input_tokens,
            "local_generated_tokens": len(generated),
            "local_finish_reason": "length" if truncated else "stop",
        }
        self.call_counter += 1
        try:
            message = parse_qwen_response(text, call_prefix=f"call_local_{self.call_counter}")
        except ToolCallParseError as exc:
            exc.generation_metadata = metadata
            raise
        return {**message, **metadata}


def paired_seed(seed: int, task_id: str, trial: int) -> int:
    offset = int.from_bytes(sha256(f"{task_id}:{trial}".encode()).digest()[:4], "big")
    return (seed + offset) % (2**31)


def select_dev_tasks(manifest: Path, *, per_template: int | None = 1) -> list[tuple[Path, TaskSpec]]:
    if per_template is not None and per_template < 1:
        raise ValueError("per_template must be positive or None")
    manifest = manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[tuple[Path, TaskSpec]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row.get("split") != "dev":
            continue
        path = (manifest.parent / str(row["task_file"])).resolve()
        task = TaskSpec.from_file(path)
        if task.split != "dev" or task.task_id != str(row["task_id"]):
            raise ValueError(f"manifest_task_mismatch:{path}")
        if task.task_id in seen:
            raise ValueError(f"duplicate_task_id:{task.task_id}")
        seen.add(task.task_id)
        if not task.database_path.is_file():
            raise FileNotFoundError(task.database_path)
        template = str(task.metadata.get("template_id") or task.scenario)
        groups[template].append((path, task))
    if not groups:
        raise ValueError("No dev tasks found; train/test tasks are never selected")
    selected = []
    for template in sorted(groups):
        values = sorted(groups[template], key=lambda item: item[1].task_id)
        if per_template is not None and len(values) < per_template:
            raise ValueError(f"not_enough_tasks:{template}:{len(values)}<{per_template}")
        selected.extend(values if per_template is None else values[:per_template])
    return selected


def validate_lora_adapter(path: Path, base_model: str) -> dict[str, Any]:
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("base_model_name_or_path") != base_model:
        raise ValueError("adapter_type_or_base_model_mismatch")
    if config.get("bias") != "none" or config.get("modules_to_save") or config.get("trainable_token_indices"):
        raise ValueError("paired_base_requires_lora_only_adapter_without_modified_bias_or_other_modules")
    weights = path / "adapter_model.safetensors"
    with weights.open("rb") as handle:
        raw_length = handle.read(8)
        header_length = int.from_bytes(raw_length, "little")
        if len(raw_length) != 8 or not 1 <= header_length <= min(weights.stat().st_size - 8, 16_000_000):
            raise ValueError("invalid_safetensors_header")
        header = json.loads(handle.read(header_length))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors or any(".lora_A." not in key and ".lora_B." not in key for key in tensors):
        raise ValueError("adapter_contains_non_lora_tensors; cannot claim disabled_adapter_is_original_base")
    data_length = weights.stat().st_size - 8 - header_length
    spans = sorted(tuple(value["data_offsets"]) for value in tensors.values())
    offset = 0
    for start, end in spans:
        if start != offset or end <= start or end > data_length:
            raise ValueError("invalid_safetensors_data_offsets")
        offset = end
    if offset != data_length:
        raise ValueError("truncated_or_extra_safetensors_data")
    return config


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
