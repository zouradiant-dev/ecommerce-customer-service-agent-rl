from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from statistics import mean, pstdev
from typing import Any, Iterable


def canonicalize_tool_schemas(tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use one recursive key order so training and remote serving render identical prompts."""

    return json.loads(json.dumps(list(tools), ensure_ascii=False, sort_keys=True))


def normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Return tool arguments in the object form expected by Qwen chat templates."""

    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return deepcopy(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("tool_arguments_invalid_json") from exc
        if not isinstance(parsed, dict):
            raise ValueError("tool_arguments_must_be_object")
        return parsed
    raise ValueError("tool_arguments_must_be_object_or_json_string")


def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the OpenAI/Qwen fields needed by the chat template."""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        item: dict[str, Any] = {"role": role, "content": message.get("content") or ""}
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message.get("tool_calls") or []:
                function = dict(call.get("function") or call)
                arguments = normalize_tool_arguments(function.get("arguments"))
                calls.append(
                    {
                        "id": str(call.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(function.get("name") or ""),
                            "arguments": arguments,
                        },
                    }
                )
            item["tool_calls"] = calls
        if role == "tool":
            item["name"] = str(message.get("name") or "")
            item["tool_call_id"] = str(message.get("tool_call_id") or "")
        normalized.append(item)
    return normalized


def validate_training_row(row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    messages = list(row.get("messages") or [])
    loss_mask = list(row.get("loss_mask") or [])
    if not messages:
        failures.append("empty_messages")
    if loss_mask and len(loss_mask) != len(messages):
        failures.append("messages_loss_mask_length_mismatch")

    calls: set[str] = set()
    responses: set[str] = set()
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        if role not in {"system", "user", "assistant", "tool"}:
            failures.append(f"invalid_role:{index}:{role}")
        if loss_mask and loss_mask[index] and role != "assistant":
            failures.append(f"non_assistant_loss:{index}:{role}")
        for call in message.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            function = call.get("function") or call
            name = str(function.get("name") or "")
            if not call_id or not name:
                failures.append(f"invalid_tool_call:{index}")
            try:
                normalize_tool_arguments(function.get("arguments"))
            except ValueError as exc:
                failures.append(f"invalid_tool_arguments:{index}:{exc}")
            calls.add(call_id)
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id or call_id not in calls:
                failures.append(f"orphan_tool_response:{index}:{call_id}")
            responses.add(call_id)
    missing = sorted(calls - responses)
    if missing:
        failures.append("tool_calls_without_response:" + ",".join(missing))
    return failures


def render_with_assistant_mask(
    tokenizer: Any,
    messages: Iterable[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    loss_mask: list[bool] | None = None,
) -> tuple[list[int], list[int]]:
    """Render once and identify assistant tokens by prefix-diffing the chat template."""

    normalized = normalize_messages(messages)
    template_kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": False}
    normalized_tools = canonicalize_tool_schemas(tools or [])
    if normalized_tools:
        template_kwargs["tools"] = normalized_tools
    full = list(tokenizer.apply_chat_template(normalized, **template_kwargs))
    mask = [0] * len(full)
    targets = loss_mask or [message.get("role") == "assistant" for message in normalized]
    if len(targets) != len(normalized):
        raise ValueError("messages_loss_mask_length_mismatch")

    for index, (message, target) in enumerate(zip(normalized, targets)):
        if message.get("role") != "assistant" or not target:
            continue
        prefix_kwargs: dict[str, Any] = {"tokenize": True}
        if normalized_tools:
            prefix_kwargs["tools"] = normalized_tools
        before = list(
            tokenizer.apply_chat_template(
                normalized[:index],
                add_generation_prompt=True,
                **prefix_kwargs,
            )
        )
        through = list(
            tokenizer.apply_chat_template(
                normalized[: index + 1],
                add_generation_prompt=False,
                **prefix_kwargs,
            )
        )
        lower, upper = len(before), len(through)
        if not (lower < upper <= len(full)) or full[:lower] != before or full[:upper] != through:
            raise ValueError(f"assistant_mask_alignment_failed:{index}")
        for position in range(lower, upper):
            mask[position] = 1
    if not any(mask):
        raise ValueError("empty_assistant_token_mask")
    return full, mask


def save_base_model_configs(model: Any, path: Any) -> None:
    """Save base-model config files next to a PEFT adapter checkpoint."""

    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if callable(get_base_model) else model
    config = getattr(base_model, "config", None)
    if config is None or not hasattr(config, "save_pretrained"):
        raise ValueError("base_model_config_unavailable")
    config.save_pretrained(path)

    generation_config = getattr(base_model, "generation_config", None)
    if generation_config is None:
        generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(path)


def compute_group_advantages(
    rollouts: Iterable[dict[str, Any]],
    *,
    denominator_epsilon: float = 0.25,
    advantage_clip: float = 3.0,
    require_mixed_outcomes: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute strict binary group-relative advantages without mutating input rows."""

    rows = [deepcopy(row) for row in rollouts]
    if any(row.get("split") == "test" for row in rows):
        raise ValueError("test rollouts must not be used for GRPO updates")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("task_id") or "")].append(row)

    mixed_groups = 0
    selected: list[dict[str, Any]] = []
    for group in groups.values():
        rewards = [1.0 if float(row.get("reward") or 0.0) >= 1.0 - 1e-6 else 0.0 for row in group]
        mixed = len(group) >= 2 and len(set(rewards)) == 2
        mixed_groups += int(mixed)
        if require_mixed_outcomes and not mixed:
            continue
        center = mean(rewards)
        deviation = pstdev(rewards) if len(rewards) > 1 else 0.0
        if deviation == 0.0:
            continue
        for row, reward in zip(group, rewards):
            advantage = (reward - center) / (deviation + denominator_epsilon)
            row["binary_reward"] = reward
            row["advantage"] = max(-advantage_clip, min(advantage_clip, advantage))
            selected.append(row)

    summary = {
        "groups": len(groups),
        "mixed_groups": mixed_groups,
        "mixed_group_rate": mixed_groups / max(1, len(groups)),
        "selected_rollouts": len(selected),
        "dropped_rollouts": len(rows) - len(selected),
        "denominator_epsilon": denominator_epsilon,
        "advantage_clip": advantage_clip,
    }
    return selected, summary
