"""Local traces without prompt bodies, tool payloads, credentials or customer PII."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import inspect
import json
import time
from typing import Any
from uuid import uuid4
import warnings

from .storage import Store, encode

_current: ContextVar[Any] = ContextVar("online_agent_span", default=None)
ALLOWED = {"conversation_id", "request_id", "operation_id", "tool_call_id", "order_id", "model_id",
           "prompt_sha256", "tools_sha256", "counter", "context_version", "context_policy", "input_budget_units", "budget_limit",
           "kept_groups", "omitted_groups", "digest_entries", "input_tokens", "output_tokens", "finish_reason",
           "outcome", "error_code", "exception_type", "handoff", "attempt", "response_model", "http_status"}


def current_span() -> Any:
    return _current.get()


@dataclass
class Span:
    owner: "Telemetry"
    kind: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def set(self, **values: Any) -> None:
        for key, value in values.items():
            if key in {"input_tokens", "output_tokens"} and (type(value) is not int or value < 0):
                continue
            if key in ALLOWED and isinstance(value, (str, int, float, bool, type(None))):
                self.attributes[key] = value[:256] if isinstance(value, str) else value

    def fail(self, code: str) -> None:
        self.status = "error"
        self.set(error_code=code)

    def __enter__(self) -> "Span":
        parent = _current.get()
        self.id = uuid4().hex
        self.trace_id = parent.trace_id if parent else uuid4().hex
        self.parent_id = parent.id if parent else None
        self.started, self.clock = time.time(), time.perf_counter()
        self.status = "ok"
        if parent:
            inherited = {key: value for key, value in parent.attributes.items() if key in
                         {"conversation_id", "request_id", "operation_id"}}
            self.attributes = {**inherited, **self.attributes}
        self.token = _current.set(self)
        self.owner.persist(self, "running", None)
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        if error:
            self.fail(getattr(error, "code", type(error).__name__))
            self.set(exception_type=type(error).__name__)
        self.owner.persist(self, self.status, (time.perf_counter() - self.clock) * 1000)
        _current.reset(self.token)


class Telemetry:
    def __init__(self, store: Store) -> None:
        self.store, self.dropped = store, 0
        with store.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS spans (id TEXT PRIMARY KEY,trace_id TEXT NOT NULL,"
                       "parent_id TEXT,kind TEXT NOT NULL,name TEXT NOT NULL,status TEXT NOT NULL,"
                       "started_at REAL NOT NULL,duration_ms REAL,attributes TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS spans_trace ON spans(trace_id,started_at)")

    def span(self, kind: str, name: str, **attributes: Any) -> Span:
        span = Span(self, kind, name)
        span.set(**attributes)
        return span

    def persist(self, span: Span, status: str, duration: float | None) -> None:
        try:
            with self.store.transaction() as db:
                db.execute("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                           "status=excluded.status,duration_ms=excluded.duration_ms,attributes=excluded.attributes",
                           (span.id, span.trace_id, span.parent_id, span.kind, span.name, status,
                            span.started, duration, encode(span.attributes)))
        except Exception as error:
            self.dropped += 1
            warnings.warn("telemetry_persist_failed:" + type(error).__name__, RuntimeWarning)

    def recover(self) -> int:
        with self.store.transaction() as db:
            return db.execute("UPDATE spans SET status='interrupted' WHERE status='running'").rowcount

    def rows(self) -> list[dict[str, Any]]:
        with self.store.transaction() as db:
            return [{**dict(row), "attributes": json.loads(row["attributes"])}
                    for row in db.execute("SELECT * FROM spans ORDER BY started_at,id")]


def traced(kind: str):
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            values = signature.bind(self, *args, **kwargs).arguments
            attributes = {dest: values[source] for source, dest in
                          (("conversation", "conversation_id"), ("request", "request_id"), ("operation", "operation_id"))
                          if source in values}
            with self.telemetry.span(kind, function.__name__, **attributes) as span:
                result = function(self, *args, **kwargs)
                if isinstance(result, dict):
                    if result.get("error"):
                        span.fail(str(result["error"]))
                    if "status" in result:
                        span.set(outcome=result["status"])
                    if "id" in result:
                        span.set(operation_id=result["id"])
                    if result.get("handed_off"):
                        span.set(handoff=True)
                return result
        return wrapped
    return decorate


class ObservedBackend:
    def __init__(self, target: Any, telemetry: Telemetry) -> None:
        self.target, self.telemetry = target, telemetry

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.target, name)
        if name not in {"list_orders", "get_order", "preview_cancel", "cancel", "lookup"}:
            return method
        def invoke(*args, **kwargs):
            with self.telemetry.span("backend", name) as span:
                result = method(*args, **kwargs)
                if isinstance(result, dict) and result.get("error"):
                    span.fail(str(result["error"]))
                return result
        return invoke


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = sorted({row["kind"] for row in rows})
    groups = {}
    for kind in kinds:
        selected = [row for row in rows if row["kind"] == kind]
        durations = sorted(row["duration_ms"] for row in selected if row["duration_ms"] is not None)
        groups[kind] = {"calls": len(selected), "errors": sum(row["status"] == "error" for row in selected),
                       "unfinished": sum(row["status"] in {"running", "interrupted"} for row in selected),
                       "p50_ms": durations[(len(durations)-1)//2] if durations else None,
                       "p95_ms": durations[max(0, int(len(durations)*.95 + .999)-1)] if durations else None}
    model = [row for row in rows if row["kind"] == "model"]
    with_usage = [row for row in model if type(row["attributes"].get("input_tokens")) is int
                  and type(row["attributes"].get("output_tokens")) is int
                  and row["attributes"]["input_tokens"] >= 0 and row["attributes"]["output_tokens"] >= 0]
    errors: dict[str, int] = {}
    for row in rows:
        if row["status"] == "error":
            key = row["kind"] + ":" + str(row["attributes"].get("error_code", "unknown"))
            errors[key] = errors.get(key, 0) + 1
    return {"schema_version": "online_trace_summary_v1", "by_kind": groups, "errors": errors,
            "model_usage_coverage": len(with_usage)/len(model) if model else None,
            "observed_input_tokens": sum(row["attributes"]["input_tokens"] for row in with_usage) if with_usage else None,
            "observed_output_tokens": sum(row["attributes"]["output_tokens"] for row in with_usage) if with_usage else None,
            "task_success_rate": None, "note": "Trace errors are diagnostics, not gold task-success labels."}
