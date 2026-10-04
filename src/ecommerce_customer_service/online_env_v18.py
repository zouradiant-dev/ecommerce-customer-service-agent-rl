"""Explicit v18 online environment; historical GRPO routes remain unchanged."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import TaskSpec
from .online_env import EcommerceCustomerServiceEnv, _final_response
from .tools_v18 import CustomerServiceToolsV18
from .verifier import VERIFIER_VERSION
from .verifier_v18 import (
    STRICT_REWARD_V18,
    TRAIN_REWARD_V18,
    V18RewardBreakdown,
    V18VerificationResult,
    score_v18,
    select_reward_v18,
    verify_task_v18,
)


class EcommerceCustomerServiceEnvV18(EcommerceCustomerServiceEnv):
    def __init__(self, *, task_path: Path, reward_mode: str = TRAIN_REWARD_V18) -> None:
        if reward_mode not in {STRICT_REWARD_V18, TRAIN_REWARD_V18}:
            raise ValueError(f"unsupported_v18_reward_mode:{reward_mode}")
        super().__init__(
            task_path=task_path,
            reward_mode="strict_binary",
            verifier_version=VERIFIER_VERSION,
        )
        self._reward_mode_v18 = reward_mode
        self._last_v18_verification: V18VerificationResult | None = None
        self._last_v18_breakdown: V18RewardBreakdown | None = None

    def reset(self, **_: Any) -> None:
        self._cleanup()
        source = TaskSpec.from_file(self._task_path)
        if source.split != "train" or source.metadata.get("scripted_user_replies"):
            raise ValueError("v18 online environment requires inline train tasks")
        if not isinstance(source.metadata.get("terminal_v18"), dict):
            raise ValueError("v18 task requires terminal_v18 metadata")
        self._temporary = tempfile.TemporaryDirectory(prefix="ecommerce-cs-v18-")
        database_path = Path(self._temporary.name) / "state.sqlite"
        shutil.copy2(source.database_path, database_path)
        self._task = replace(source, database_path=database_path)
        self._tools = CustomerServiceToolsV18(database_path, self._task)
        self._tools.observe_user_message(self._task.user_prompt)
        self._last_verification = None
        self._last_breakdown = None
        self._last_v18_verification = None
        self._last_v18_breakdown = None

    def _evaluate(self, final_response: str, *, stopped_reason: str = "completed") -> float:
        if self._task is None or self._tools is None:
            raise RuntimeError("environment_not_reset")
        calls = tuple(item.to_dict() for item in self._tools.executions)
        self._last_v18_verification = verify_task_v18(self._task, final_response, calls)
        self._last_v18_breakdown = score_v18(
            self._task,
            self._last_v18_verification,
            calls,
            stopped_reason=stopped_reason,
        )
        return select_reward_v18(self._last_v18_breakdown, self._reward_mode_v18)


def build_environment_factories_v18(
    task_paths: Iterable[Path],
    *,
    reward_mode: str = TRAIN_REWARD_V18,
) -> dict[str, Callable[[], EcommerceCustomerServiceEnvV18]]:
    factories: dict[str, Callable[[], EcommerceCustomerServiceEnvV18]] = {}
    for path in task_paths:
        task = TaskSpec.from_file(path)
        if not isinstance(task.metadata.get("terminal_v18"), dict):
            raise ValueError(f"v18 task missing terminal_v18 metadata:{task.task_id}")
        factories[task.task_id] = partial(
            EcommerceCustomerServiceEnvV18,
            task_path=Path(path).resolve(),
            reward_mode=reward_mode,
        )
    return factories


def environment_reward_v18(
    completions: list[Any],
    environments: list[EcommerceCustomerServiceEnvV18],
    **_: Any,
) -> list[float]:
    if len(completions) != len(environments):
        raise ValueError("completion_environment_length_mismatch")
    return [
        environment._evaluate(_final_response(completion))
        for completion, environment in zip(completions, environments)
    ]
