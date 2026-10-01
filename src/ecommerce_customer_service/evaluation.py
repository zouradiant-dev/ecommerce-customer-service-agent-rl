from __future__ import annotations

from collections import Counter, defaultdict
from math import comb
from statistics import mean
from typing import Any, Iterable

from .diagnostics import classify_bad_case


def _pass_at_k(successes: int, trials: int, k: int) -> float | None:
    if trials < k:
        return None
    if successes == 0:
        return 0.0
    if trials - successes < k:
        return 1.0
    return 1.0 - comb(trials - successes, k) / comb(trials, k)


def evaluate_rollouts(rows: Iterable[dict[str, Any]], *, pass_k: tuple[int, ...] = (1, 2, 4)) -> dict[str, Any]:
    materialized = list(rows)
    by_task: dict[str, list[float]] = defaultdict(list)
    scenario_values: dict[str, list[float]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    for row in materialized:
        value = 1.0 if float(row.get("reward") or 0.0) >= 1.0 - 1e-6 else 0.0
        by_task[str(row.get("task_id"))].append(value)
        scenario_values[str(row.get("scenario") or "unknown")].append(value)
        failure_counts.update(classify_bad_case(row))
    task_scores = {task_id: mean(values) for task_id, values in by_task.items()}
    pass_at: dict[str, float | None] = {}
    for k in pass_k:
        estimates = [
            value
            for values in by_task.values()
            if (value := _pass_at_k(int(sum(values)), len(values), k)) is not None
        ]
        pass_at[str(k)] = mean(estimates) if estimates else None
    return {
        "rollouts": len(materialized),
        "tasks": len(by_task),
        "pass_1": mean(task_scores.values()) if task_scores else 0.0,
        "pass_at_k": pass_at,
        "scenario_pass_1": {key: mean(values) for key, values in sorted(scenario_values.items())},
        "failure_taxonomy": dict(sorted(failure_counts.items())),
        "task_scores": dict(sorted(task_scores.items())),
    }


def paired_bootstrap_ci(deltas: dict[str, float], *, samples: int = 10_000, seed: int = 12_345) -> tuple[float, float]:
    values = [deltas[key] for key in sorted(deltas)]
    if not values:
        return 0.0, 0.0
    means: list[float] = []
    state = seed
    for _ in range(samples):
        total = 0.0
        for _ in values:
            state = (1_103_515_245 * state + 12_345) & 0x7FFFFFFF
            total += values[state % len(values)]
        means.append(total / len(values))
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def compare_rollouts(base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = evaluate_rollouts(base_rows)
    candidate = evaluate_rollouts(candidate_rows)
    common = sorted(set(base["task_scores"]) & set(candidate["task_scores"]))
    deltas = {task: candidate["task_scores"][task] - base["task_scores"][task] for task in common}
    low, high = paired_bootstrap_ci(deltas)
    return {
        "common_tasks": len(common),
        "base_pass_1": mean(base["task_scores"][task] for task in common) if common else 0.0,
        "candidate_pass_1": mean(candidate["task_scores"][task] for task in common) if common else 0.0,
        "delta": mean(deltas.values()) if deltas else 0.0,
        "paired_bootstrap_95_ci": [low, high],
        "significant": low > 0 or high < 0,
    }

