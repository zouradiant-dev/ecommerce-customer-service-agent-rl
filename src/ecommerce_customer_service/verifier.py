from __future__ import annotations

import re
import sqlite3
from typing import Any, Iterable
import unicodedata
from decimal import Decimal, InvalidOperation

from .communication import missing_communication_intents
from .database import connect
from .models import StateAssertion, TaskSpec, VerificationResult


VERIFIER_VERSION = "deterministic_semantic_v8"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
WRITE_TOOLS = {"cancel_order", "update_shipping_address", "create_refund", "escalate_to_human"}
SUCCESS_CLAIMS = {
    "cancel_order": ("订单已取消", "取消成功"),
    "update_shipping_address": ("地址已更新", "收货地址已更新", "修改地址成功"),
    "create_refund": ("已创建退货", "创建退货并原路退款", "已原路退款", "退款成功"),
    "escalate_to_human": ("已创建人工工单", "已经为您创建人工工单", "已升级人工"),
}
LEGACY_REQUIRED_FACT_ALIASES = {
    "当前地址已经是": (
        "当前地址已经是",
        "当前收货地址已经是",
        "目前地址已经是",
        "目前收货地址已经是",
        "当前地址为",
        "当前收货地址为",
    ),
    "无需修改": (
        "无需修改",
        "不需要修改",
        "不用修改",
        "不必修改",
        "无需变更",
        "无需重复写入",
        "不需要重复写入",
        "不用重复写入",
    ),
}


def _fold_text(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold())


def _required_fact_present(fact: str, folded_response: str) -> bool:
    candidates = LEGACY_REQUIRED_FACT_ALIASES.get(fact, (fact,))
    return any(_fold_text(candidate) in folded_response for candidate in candidates)


def _required_value_present(value: str, response: str, folded_response: str) -> bool:
    money = re.fullmatch(r"(\d+)\.(\d{2})", value)
    if money:
        normalized_response = unicodedata.normalize("NFKC", response)
        try:
            expected = Decimal(value)
        except InvalidOperation:
            return False
        candidates = re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", normalized_response)
        return any(Decimal(candidate) == expected for candidate in candidates)
    return _fold_text(value) in folded_response


def communication_failures(task: TaskSpec, final_response: str) -> tuple[str, ...]:
    """Return versioned deterministic communication failures without inspecting state."""

    folded_response = _fold_text(final_response)
    missing_values = [
        value
        for value in task.required_values
        if not _required_value_present(value, final_response, folded_response)
    ]
    missing_facts = [fact for fact in task.required_facts if not _required_fact_present(fact, folded_response)]
    missing_intents = missing_communication_intents(task.required_intents, final_response)
    return tuple(
        [f"missing_value:{value}" for value in missing_values]
        + [f"missing_communication:{fact}" for fact in missing_facts]
        + [f"missing_intent:{intent}" for intent in missing_intents]
    )


def required_tool_failures(
    task: TaskSpec,
    tool_executions: Iterable[dict[str, Any]],
) -> tuple[str, ...]:
    """Return missing required-tool failures from a recorded trajectory."""

    executions = list(tool_executions)
    allowed_violations = set(task.allowed_policy_violations)
    successful_names = {str(item.get("name")) for item in executions if item.get("ok")}
    missing: list[str] = []
    for name in task.required_tools:
        acceptable_names = {name}
        if task.scenario == "latest_order_lookup" and name == "get_order":
            # list_orders already supplies the amount and order state; a successful
            # shipment lookup is equally strong evidence for the requested ship status.
            acceptable_names.add("get_shipment")
        if successful_names & acceptable_names:
            continue
        expected_failure_seen = any(
            item.get("name") == name and item.get("policy_violation") in allowed_violations
            for item in executions
        )
        if not expected_failure_seen:
            missing.append(f"required_tool_missing_or_failed:{name}")
    return tuple(missing)


def policy_failures(
    task: TaskSpec,
    final_response: str,
    tool_executions: Iterable[dict[str, Any]],
) -> tuple[str, ...]:
    """Recompute policy and tool-recovery failures from a recorded trajectory."""

    executions = list(tool_executions)
    failures: list[str] = []
    folded_response = _fold_text(final_response)
    allowed_violations = set(task.allowed_policy_violations)
    unexpected_violations = [
        str(item.get("policy_violation"))
        for item in executions
        if item.get("policy_violation") and str(item.get("policy_violation")) not in allowed_violations
    ]
    failures.extend(f"policy_violation:{value}" for value in unexpected_violations)

    allowed_writes = set(task.allowed_write_tools)
    unexpected_writes = [
        str(item.get("name"))
        for item in executions
        if item.get("is_write") and item.get("name") not in allowed_writes
    ]
    failures.extend(f"unexpected_write:{name}" for name in unexpected_writes)

    successful_names = {str(item.get("name")) for item in executions if item.get("ok")}
    for tool_name, markers in SUCCESS_CLAIMS.items():
        if tool_name in successful_names:
            continue
        if any(_fold_text(marker) in folded_response for marker in markers):
            failures.append(f"unsupported_success_claim:{tool_name}")

    for index, item in enumerate(executions):
        name = str(item.get("name") or "")
        if not item.get("is_write") or name == "escalate_to_human":
            continue
        prior_success = {str(row.get("name")) for row in executions[:index] if row.get("ok")}
        evidence_tools = {value for value in task.required_tools if value not in WRITE_TOOLS}
        missing_evidence = sorted(evidence_tools - prior_success)
        if missing_evidence:
            failures.append(f"premature_write:{name}:missing={','.join(missing_evidence)}")

    for index, item in enumerate(executions):
        if item.get("ok") or item.get("policy_violation"):
            continue
        name = str(item.get("name") or "")
        recovered = any(
            later.get("ok")
            and (later.get("name") == name or later.get("name") == "escalate_to_human")
            for later in executions[index + 1 :]
        )
        if not recovered:
            failures.append(f"unrecovered_tool_error:{name}:{(item.get('output') or {}).get('error', 'unknown')}")

    minimum_attempts = {
        str(name): int(count)
        for name, count in dict(task.metadata.get("min_attempts_before_escalation") or {}).items()
    }
    for index, item in enumerate(executions):
        if item.get("name") != "escalate_to_human" or not item.get("ok"):
            continue
        for tool_name, required_count in minimum_attempts.items():
            observed = sum(row.get("name") == tool_name for row in executions[:index])
            if observed < required_count:
                failures.append(
                    f"premature_escalation:{tool_name}:required={required_count}:actual={observed}"
                )
    return tuple(failures)


def _assert_identifier(value: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Unsafe SQL identifier in trusted task spec: {value!r}")
    return value


def _check_assertion(connection: sqlite3.Connection, assertion: StateAssertion) -> list[str]:
    table = _assert_identifier(assertion.table)
    where_items = list(assertion.where.items())
    where_sql = " AND ".join(f"{_assert_identifier(key)} = ?" for key, _ in where_items) or "1 = 1"
    parameters = [value for _, value in where_items]
    rows = list(connection.execute(f"SELECT * FROM {table} WHERE {where_sql}", parameters))
    failures: list[str] = []
    if assertion.count is not None and len(rows) != assertion.count:
        failures.append(f"state_count:{table}:expected={assertion.count}:actual={len(rows)}")
    if assertion.equals:
        if not rows:
            failures.append(f"state_missing:{table}:{assertion.where}")
        else:
            for key, expected in assertion.equals.items():
                column = _assert_identifier(key)
                if rows[0][column] != expected:
                    failures.append(
                        f"state_value:{table}.{column}:expected={expected!r}:actual={rows[0][column]!r}"
                    )
    return failures


def verify_task(
    task: TaskSpec,
    final_response: str,
    tool_executions: Iterable[dict[str, Any]],
) -> VerificationResult:
    executions = list(tool_executions)
    failures: list[str] = []
    with connect(task.database_path) as connection:
        for assertion in task.assertions:
            failures.extend(_check_assertion(connection, assertion))
    state_passed = not any(item.startswith("state_") for item in failures)

    communication_failure_rows = communication_failures(task, final_response)
    failures.extend(communication_failure_rows)
    communication_passed = not communication_failure_rows

    required_failure_rows = required_tool_failures(task, executions)
    failures.extend(required_failure_rows)
    policy_failure_rows = policy_failures(task, final_response, executions)
    failures.extend(policy_failure_rows)
    policy_passed = not policy_failure_rows
    required_tools_passed = not required_failure_rows
    passed = state_passed and communication_passed and policy_passed and required_tools_passed
    return VerificationResult(
        passed=passed,
        state_passed=state_passed,
        communication_passed=communication_passed,
        policy_passed=policy_passed,
        required_tools_passed=required_tools_passed,
        failures=tuple(failures),
    )
