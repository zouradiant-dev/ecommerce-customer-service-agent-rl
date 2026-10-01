from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .data import write_jsonl


ABCD_SOURCE_URL = "https://github.com/asappresearch/abcd"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    opener = gzip.open if path.suffix.casefold() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _turns(conversation: dict[str, Any], *, use_delexed: bool) -> Iterable[tuple[str, str, str | None]]:
    if use_delexed:
        for turn in conversation.get("delexed") or []:
            speaker = str(turn.get("speaker") or "").casefold()
            text = str(turn.get("text") or "").strip()
            targets = list(turn.get("targets") or [])
            action_name = str(targets[2]) if len(targets) > 2 and targets[2] else None
            yield speaker, text, action_name
        return

    for turn in conversation.get("original") or []:
        if isinstance(turn, dict):
            yield str(turn.get("speaker") or "").casefold(), str(turn.get("text") or "").strip(), None
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            yield str(turn[0]).casefold(), str(turn[1]).strip(), None


def convert_abcd_dataset(
    source_path: Path,
    output_dir: Path,
    *,
    splits: tuple[str, ...] = ("train", "dev"),
    use_delexed: bool = True,
) -> dict[str, Any]:
    """Convert a local ABCD JSON/JSON.GZ file into language-only SFT rows.

    ABCD action turns are retained as metadata, not represented as successful ecommerce
    tool calls. This prevents human-dialogue data from receiving an unearned tool reward.
    The test split is rejected by default and must never be mixed into training output.
    """

    requested = tuple(dict.fromkeys(str(split) for split in splits))
    if not requested or any(split not in {"train", "dev"} for split in requested):
        raise ValueError("ABCD conversion only accepts the train and dev splits")

    source_path = source_path.resolve()
    source_sha256 = _sha256(source_path)
    payload = _load_json(source_path)
    if isinstance(payload, list):
        source_splits: dict[str, list[dict[str, Any]]] = {"train": payload}
    elif isinstance(payload, dict):
        source_splits = {str(key): list(value) for key, value in payload.items() if isinstance(value, list)}
    else:
        raise ValueError("ABCD source must be a split dictionary or a conversation list")

    output_dir = output_dir.resolve()
    counts: dict[str, int] = {}
    skipped = 0
    action_turns = 0
    for split in requested:
        rows: list[dict[str, Any]] = []
        for index, conversation in enumerate(source_splits.get(split) or []):
            messages: list[dict[str, str]] = []
            actions: list[dict[str, Any]] = []
            for turn_index, (speaker, content, action_name) in enumerate(
                _turns(conversation, use_delexed=use_delexed),
                1,
            ):
                if not content:
                    continue
                if speaker == "customer":
                    messages.append({"role": "user", "content": content})
                elif speaker == "agent":
                    messages.append({"role": "assistant", "content": content})
                elif speaker == "action":
                    actions.append(
                        {
                            "turn_index": turn_index,
                            "action": action_name,
                            "text": content,
                        }
                    )
            roles = {message["role"] for message in messages}
            if not {"user", "assistant"}.issubset(roles):
                skipped += 1
                continue
            conversation_id = str(conversation.get("convo_id", index))
            scenario = dict(conversation.get("scenario") or {})
            rows.append(
                {
                    "id": f"abcd_{split}_{conversation_id}",
                    "format_version": "external_dialogue_sft_v1",
                    "messages": messages,
                    "loss_mask": [message["role"] == "assistant" for message in messages],
                    "metadata": {
                        "source": "ABCD",
                        "source_url": ABCD_SOURCE_URL,
                        "source_sha256": source_sha256,
                        "license": "MIT",
                        "split": split,
                        "conversation_id": conversation_id,
                        "flow": scenario.get("flow"),
                        "subflow": scenario.get("subflow"),
                        "text_variant": "delexed" if use_delexed else "original",
                        "excluded_action_turns": actions,
                        "training_scope": "language_sft_only",
                    },
                }
            )
            action_turns += len(actions)
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)

    summary = {
        "format_version": "external_dialogue_sft_v1",
        "source": "ABCD",
        "source_url": ABCD_SOURCE_URL,
        "source_file": source_path.name,
        "source_sha256": source_sha256,
        "license": "MIT",
        "text_variant": "delexed" if use_delexed else "original",
        "splits": counts,
        "skipped_conversations": skipped,
        "excluded_action_turns": action_turns,
        "note": "Use for language SFT only; ecommerce tool correctness still comes from verified local trajectories.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
