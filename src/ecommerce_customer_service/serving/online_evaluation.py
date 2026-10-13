"""Online runtime regression runner. Gold labels never enter model context."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Any

from .backend import MockBusinessBackend
from .context_builder import CONTEXT_VERSION, ContextBuilder
from .diagnostics import OUTPUT_ERRORS
from .model import ChatModel
from .policy import get_policy
from .service import CustomerService
from .storage import ServiceError, Store, encode
from .telemetry import summarize


class ScriptedSmokeModel:
    """Test fixture, explicitly not a trained model or algorithm baseline."""
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.model = "engineering_fixture_not_llm"

    def complete(self, messages, tools):
        if not self.responses:
            raise RuntimeError("smoke_fixture_exhausted")
        return self.responses.pop(0)


class FaultBackend:
    def __init__(self, target: MockBusinessBackend) -> None:
        self.target = target
        self.lose_ack = self.unavailable = self.read_timeout = False
        self.cancel_attempts = 0

    def __getattr__(self, name):
        return getattr(self.target, name)

    def get_order(self, customer, order):
        if self.read_timeout:
            self.read_timeout = False
            raise TimeoutError("injected_read_timeout")
        return self.target.get_order(customer, order)

    def cancel(self, *args):
        self.cancel_attempts += 1
        if self.unavailable:
            raise TimeoutError("injected_backend_unavailable")
        result = self.target.cancel(*args)
        if self.lose_ack:
            self.lose_ack = False
            raise TimeoutError("injected_lost_response_after_commit")
        return result

    def lookup(self, *args):
        return None if self.unavailable else self.target.lookup(*args)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def initial_orders(case: dict[str, Any], backend: MockBusinessBackend) -> None:
    if "initial_orders" not in case:
        return
    rows = case["initial_orders"]
    fields = {"id", "customer_id", "product", "status", "paid_cents", "version"}
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20:
        raise ValueError("invalid_initial_orders")
    for row in rows:
        if (not isinstance(row, dict) or set(row) != fields
                or not isinstance(row["id"], str) or not 1 <= len(row["id"]) <= 128
                or row["customer_id"] not in {"customer_alice", "customer_bob"}
                or not isinstance(row["product"], str) or not 1 <= len(row["product"]) <= 300
                or row["status"] not in {"pending", "paid", "shipped", "cancelled"}
                or type(row["paid_cents"]) is not int or row["paid_cents"] < 0
                or type(row["version"]) is not int or row["version"] < 1):
            raise ValueError("invalid_initial_order")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate_initial_order")
    # Only the newly created isolated case database is changed, never the web database.
    with backend.transaction() as db:
        db.execute("DELETE FROM orders")
        db.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", [
            tuple(row[name] for name in ("id", "customer_id", "product", "status", "paid_cents", "version")) for row in rows])


def run_case(case: dict[str, Any], directory: Path, model: ChatModel,
             context: ContextBuilder, *, policy_profile: str = "baseline_v1") -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    store = Store(directory / "sessions.sqlite")
    raw_backend = MockBusinessBackend(directory / "business.sqlite")
    initial_orders(case, raw_backend)
    backend = FaultBackend(raw_backend)
    service = CustomerService(store, backend, model, context_builder=context, policy_profile=policy_profile)
    customer = case.get("customer_id", "customer_alice")
    conversation = service.create_conversation(customer)["conversation_id"]
    aliases: dict[str, str] = {}
    failures, events = [], []
    started = time.perf_counter()
    for index, step in enumerate(case["steps"]):
        try:
            kind = step["kind"]
            if kind == "message":
                result = service.message(customer, conversation, "turn_" + str(index), step["text"])
                if result.get("error"):
                    failures.append("turn_failed:" + str(result["error"]))
                operation = result.get("operation")
                if operation:
                    aliases[step.get("save_as", "latest")] = operation["id"]
                    aliases["latest"] = operation["id"]
            elif kind in {"confirm", "query"}:
                operation = aliases.get(step.get("ref", "latest"))
                if operation is None:
                    raise ValueError("agent_did_not_create_requested_preview")
                result = (service.confirm(customer, operation) if kind == "confirm" else service.get_operation(customer, operation))
            elif kind == "event":
                name = step["name"]
                if name == "change_order":
                    if step["status"] not in {"paid", "shipped", "cancelled"}:
                        raise ValueError("unsupported_order_event")
                    with raw_backend.transaction() as db:
                        db.execute("UPDATE orders SET status=?,version=version+1 WHERE id=?", (step["status"], step["order_id"]))
                elif name == "expire_confirmation":
                    with store.transaction() as db:
                        db.execute("UPDATE operations SET expires_at=0 WHERE conversation_id=? AND status='pending_confirmation'", (conversation,))
                elif name == "restart":
                    service = CustomerService(Store(directory / "sessions.sqlite"), backend, model, context_builder=context, policy_profile=policy_profile)
                    service.recover()
                elif name in {"lose_ack", "backend_unavailable", "read_timeout_once"}:
                    setattr(backend, {"lose_ack": "lose_ack", "backend_unavailable": "unavailable",
                                      "read_timeout_once": "read_timeout"}[name], True)
                else:
                    raise ValueError("unsupported_event")
                result = {"event_applied": name}
            elif kind == "checkpoint":
                assertions = step["assertions"]
                if (not isinstance(assertions, dict) or set(assertions) - {"pending_count", "cancellation_count"}
                        or any(type(value) is not int or value < 0 for value in assertions.values())):
                    raise ValueError("invalid_checkpoint")
                with store.transaction() as db:
                    pending = db.execute("SELECT COUNT(*) FROM operations WHERE conversation_id=? AND status='pending_confirmation' AND expires_at>?",
                                         (conversation, time.time())).fetchone()[0]
                with raw_backend.transaction() as db:
                    cancelled = db.execute("SELECT COUNT(*) FROM cancellations").fetchone()[0]
                result = {"pending_count": pending, "cancellation_count": cancelled}
                for key, expected_value in assertions.items():
                    if result[key] != expected_value:
                        failures.append("checkpoint:" + key)
            else:
                raise ValueError("unsupported_step")
            if step.get("expected_error"):
                failures.append("expected_error_not_raised:" + step["expected_error"])
            events.append({"step": index, "kind": kind, "result": result})
        except Exception as error:
            code = error.code if isinstance(error, ServiceError) else str(error)
            if code != step.get("expected_error"):
                failures.append("step_error:" + code)
            events.append({"step": index, "kind": step.get("kind"), "error": code})

    expected = case["expected"]
    with raw_backend.transaction() as db:
        statuses = {row["id"]: row["status"] for row in db.execute("SELECT id,status FROM orders")}
        refunds = [dict(row) for row in db.execute("SELECT * FROM refunds")]
        cancellation_count = db.execute("SELECT COUNT(*) FROM cancellations").fetchone()[0]
    for order, status in expected["order_statuses"].items():
        if statuses.get(order) != status:
            failures.append("order_state:" + order)
    if len(refunds) != expected["refund_count"]:
        failures.append("refund_count")
    if cancellation_count != expected["cancellation_count"]:
        failures.append("cancellation_count")
    if expected.get("refund_status") and any(row["status"] != expected["refund_status"] for row in refunds):
        failures.append("refund_status")
    history = service.history(customer, conversation)
    if expected.get("conversation_status") and history["status"] != expected["conversation_status"]:
        failures.append("conversation_status")
    operations = history["operations"]
    if expected.get("operation_status") and (not operations or operations[-1]["status"] != expected["operation_status"]):
        failures.append("operation_status")
    if "pending_count" in expected and sum(row["status"] == "pending_confirmation" and row["expires_at"] > time.time()
                                         for row in operations) != expected["pending_count"]:
        failures.append("pending_operation_count")
    replies = [str(row.get("content", "")) for row in history["messages"] if row.get("role") == "assistant"]
    for phrase in expected.get("forbidden_reply_phrases", []):
        if any(phrase in reply for reply in replies):
            failures.append("reply_phrase:" + phrase)
    if expected["cancellation_count"] == 0:
        for reply in replies:
            for clause in re.split(r"[。！？；\n]", reply):
                if re.search(r"已取消|已经取消|取消成功", clause) and not re.search(r"未|没有|不能|无法|不代表|不表示", clause):
                    failures.append("unsupported_cancel_success_phrase")
                    break
    spans = service.telemetry.rows()
    context_omitted = any(row["attributes"].get("omitted_groups", 0) > 0 for row in spans)
    invalid_tools = sum(row["kind"] == "tool" and row["name"] == "__invalid_tool__" for row in spans)
    if invalid_tools:
        failures.append("forbidden_tool_attempt")
    # A nominally completed turn must not claim success if business state is wrong.
    result = {"case_id": case["case_id"], "category": case["category"], "passed": not failures,
              "failures": failures, "elapsed_ms": (time.perf_counter()-started)*1000,
              "model_calls": sum(row["kind"] == "model" for row in spans),
              "tool_calls": sum(row["kind"] == "tool" for row in spans),
              "tool_errors": sum(row["kind"] == "tool" and row["status"] == "error" for row in spans),
              "context_omission_observed": context_omitted,
              "cancel_attempts": backend.cancel_attempts, "forbidden_tool_attempts": invalid_tools,
              "model_call_failures": sum(row["kind"] == "model" and row["status"] == "error" for row in spans),
              "model_output_failures": sum(row["kind"] == "model" and row["status"] == "error" and row["attributes"].get("error_code") in OUTPUT_ERRORS for row in spans),
              "model_infrastructure_failures": sum(row["kind"] == "model" and row["status"] == "error" and row["attributes"].get("error_code") not in OUTPUT_ERRORS for row in spans),
              "order_statuses": statuses, "refund_count": len(refunds), "trace_summary": summarize(spans)}
    write_json(directory / "events.json", events)
    write_json(directory / "history.json", history)
    write_json(directory / "spans.json", spans)
    write_json(directory / "result.json", result)
    return result


def run_suite(cases: list[dict[str, Any]], output: Path, *, mode: str,
              model: ChatModel | None = None, context: ContextBuilder | None = None,
              policy_profile: str = "baseline_v1") -> dict[str, Any]:
    if mode not in {"smoke", "model"} or (mode == "model" and model is None):
        raise ValueError("invalid_evaluation_mode")
    if not cases:
        raise ValueError("empty_suite")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value) for value in ids):
        raise ValueError("invalid_or_duplicate_case_id")
    if any(case.get("split") != "dev" for case in cases):
        raise ValueError("this_regression_runner_accepts_dev_only; use_a_separate_frozen_test_protocol")
    policy = get_policy(policy_profile)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = context or ContextBuilder()
    protocol = {"schema_version": "online_agent_evaluation_v3", "mode": mode,
                "context_version": CONTEXT_VERSION, **policy.metadata(),
                "evaluation_kind": "engineering_smoke" if mode == "smoke" else "served_model_on_simulated_business",
                "real_model_execution": mode == "model", "production_quality_claim": False,
                "independent_test_claim": False, "model": getattr(model, "model", None),
                "case_sha256": sha256(encode(cases).encode()).hexdigest(), "case_ids": ids,
                "system_prompt_sha256": sha256(policy.prompt.encode()).hexdigest(),
                "tools_sha256": sha256(encode(policy.tools).encode()).hexdigest(),
                "context_counter": context.counter.name, "context_budget": context.max_input_tokens,
                "context_policy": context.policy,
                "recent_turns": context.recent_turns, "user_simulation": "fixed_user_turns_and_explicit_confirmation_events",
                "evaluation_path": "service_runtime_on_isolated_simulated_business",
                "source_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("service.py", "context_builder.py", "telemetry.py", "model.py", "online_evaluation.py",
                                 "backend.py", "storage.py", "online_cases.py", "api.py", "policy.py")}}
    write_json(output / "protocol.json", protocol)
    results = []
    for case in cases:
        active_model = ScriptedSmokeModel(case["smoke_responses"]) if mode == "smoke" else model
        results.append(run_case(case, output / "cases" / case["case_id"], active_model, context, policy_profile=policy_profile))
        # Fail fast on a broken model endpoint, rather than buying many failing calls.
        if mode == "model" and results[-1]["model_infrastructure_failures"]:
            break
    passed = sum(row["passed"] for row in results)
    complete = len(results) == len(cases) and not (
        mode == "model" and any(row["model_infrastructure_failures"] for row in results))
    summary = {**protocol, "complete": complete, "expected_cases": len(cases), "evaluated_cases": len(results),
               "passed_cases": passed, "case_pass_rate": passed/len(results) if complete else None,
               "model_calls": sum(row["model_calls"] for row in results),
               "tool_calls": sum(row["tool_calls"] for row in results),
               "tool_errors": sum(row["tool_errors"] for row in results),
               "model_output_failures": sum(row["model_output_failures"] for row in results),
               "model_infrastructure_failures": sum(row["model_infrastructure_failures"] for row in results),
               "failures": dict(Counter(failure.split(":")[0] for row in results for failure in row["failures"])),
               "by_category": {category: {"cases": sum(row["category"] == category for row in results),
                    "passed": sum(row["category"] == category and row["passed"] for row in results)}
                    for category in sorted({row["category"] for row in results})},
               "limitations": ["synthetic_dev_regression_not_independent_test", "fixed_user_turns_not_adaptive_user",
                    "deterministic_state_checks_and_limited_phrase_checks", "no_real_payment_or_production_claim"]}
    write_json(output / "summary.json", summary)
    with (output / "bad_cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in results:
            if not row["passed"]:
                stream.write(encode(row) + "\n")
    report = ["# 在线 Agent 开发回归报告", "", "模式：" + protocol["evaluation_kind"], "",
              "Smoke 结果仅验证工程行为，不能证明 SFT 模型质量。" if mode == "smoke" else
              "模型结果来自模拟业务开发回归，不能替代独立测试或真实业务效果。", "",
              f"完整执行：{complete}；通过：{passed}/{len(results)}。", "",
              "| 用例 | 通过 | 模型调用 | 工具调用 | 失败 |", "|---|---|---:|---:|---|"]
    report += [f"| {row['case_id']} | {row['passed']} | {row['model_calls']} | {row['tool_calls']} | {'; '.join(row['failures'])} |" for row in results]
    (output / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return summary
