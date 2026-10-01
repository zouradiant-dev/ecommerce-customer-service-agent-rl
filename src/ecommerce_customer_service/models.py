from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StateAssertion:
    table: str
    where: dict[str, Any]
    equals: dict[str, Any] = field(default_factory=dict)
    count: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateAssertion":
        return cls(
            table=str(value["table"]),
            where=dict(value.get("where") or {}),
            equals=dict(value.get("equals") or {}),
            count=value.get("count"),
        )


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    scenario: str
    user_id: str
    user_prompt: str
    reference_time: str
    database_path: Path
    assertions: tuple[StateAssertion, ...]
    required_values: tuple[str, ...] = ()
    required_intents: tuple[str, ...] = ()
    # Legacy v1 compatibility. New datasets use required_values/required_intents.
    required_facts: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    allowed_write_tools: tuple[str, ...] = ()
    allowed_policy_violations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "TaskSpec":
        path = path.resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        db_value = Path(str(value.get("database") or "state.sqlite"))
        database_path = db_value if db_value.is_absolute() else path.parent / db_value
        return cls(
            task_id=str(value["task_id"]),
            split=str(value["split"]),
            scenario=str(value["scenario"]),
            user_id=str(value["user_id"]),
            user_prompt=str(value["user_prompt"]),
            reference_time=str(value["reference_time"]),
            database_path=database_path.resolve(),
            assertions=tuple(StateAssertion.from_dict(item) for item in value.get("assertions") or []),
            required_values=tuple(str(item) for item in value.get("required_values") or []),
            required_intents=tuple(str(item) for item in value.get("required_intents") or []),
            required_facts=tuple(str(item) for item in value.get("required_facts") or []),
            required_tools=tuple(str(item) for item in value.get("required_tools") or []),
            allowed_write_tools=tuple(str(item) for item in value.get("allowed_write_tools") or []),
            allowed_policy_violations=tuple(str(item) for item in value.get("allowed_policy_violations") or []),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self, *, relative_database: str = "state.sqlite") -> dict[str, Any]:
        payload = {
            "schema_version": "ecommerce_cs_task_v2",
            "task_id": self.task_id,
            "split": self.split,
            "scenario": self.scenario,
            "user_id": self.user_id,
            "user_prompt": self.user_prompt,
            "reference_time": self.reference_time,
            "database": relative_database,
            "assertions": [asdict(item) for item in self.assertions],
            "required_values": list(self.required_values),
            "required_intents": list(self.required_intents),
            "required_tools": list(self.required_tools),
            "allowed_write_tools": list(self.allowed_write_tools),
            "allowed_policy_violations": list(self.allowed_policy_violations),
            "metadata": dict(self.metadata),
        }
        if self.required_facts:
            payload["required_facts"] = list(self.required_facts)
        return payload


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    state_passed: bool
    communication_passed: bool
    policy_passed: bool
    required_tools_passed: bool
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutRecord:
    task_id: str
    split: str
    scenario: str
    trial: int
    reward: float
    messages: tuple[dict[str, Any], ...]
    tool_executions: tuple[dict[str, Any], ...]
    final_response: str
    verifier: VerificationResult
    verifier_version: str = "deterministic_semantic_v2"
    reward_components: dict[str, float] = field(default_factory=dict)
    stopped_reason: str = "completed"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ecommerce_cs_rollout_v2",
            "task_id": self.task_id,
            "split": self.split,
            "scenario": self.scenario,
            "trial": self.trial,
            "reward": self.reward,
            "messages": list(self.messages),
            "tool_executions": list(self.tool_executions),
            "final_response": self.final_response,
            "verifier": self.verifier.to_dict(),
            "verifier_version": self.verifier_version,
            "reward_components": dict(self.reward_components),
            "stopped_reason": self.stopped_reason,
            "error": self.error,
        }
