from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ServiceError(Exception):
    def __init__(self, code: str, status: int = 400) -> None:
        super().__init__(code)
        self.code, self.status = code, status


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, created_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS turns (
    conversation_id TEXT NOT NULL REFERENCES conversations(id), request_id TEXT NOT NULL,
    text TEXT NOT NULL, status TEXT NOT NULL, response TEXT, created_at REAL NOT NULL,
    PRIMARY KEY (conversation_id, request_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running_turn ON turns(conversation_id)
    WHERE status='running';
CREATE TABLE IF NOT EXISTS request_intents (
    conversation_id TEXT NOT NULL, request_id TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (conversation_id, request_id),
    FOREIGN KEY (conversation_id,request_id) REFERENCES turns(conversation_id,request_id)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
    request_id TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL,
    turn_message INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS operations (
    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
    customer_id TEXT NOT NULL, order_id TEXT NOT NULL, amount_cents INTEGER NOT NULL,
    order_version INTEGER NOT NULL, status TEXT NOT NULL, expires_at REAL NOT NULL,
    receipt TEXT, error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS operation_conversation ON operations(conversation_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS one_live_operation_per_order ON operations(customer_id,order_id)
    WHERE status IN ('pending_confirmation','executing','unknown','submitted');
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
    request_id TEXT NOT NULL, tool TEXT NOT NULL, arguments TEXT NOT NULL,
    result TEXT NOT NULL, created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def require_owner(db: sqlite3.Connection, customer: str, conversation: str) -> None:
        row = db.execute("SELECT customer_id FROM conversations WHERE id=?", (conversation,)).fetchone()
        if row is None or row["customer_id"] != customer:
            raise ServiceError("conversation_not_found", 404)

    def history(self, customer: str, conversation: str) -> list[dict[str, Any]]:
        with self.transaction() as db:
            self.require_owner(db, customer, conversation)
            return [json.loads(row[0]) for row in db.execute(
                "SELECT payload FROM messages WHERE conversation_id=? ORDER BY id", (conversation,)
            )]

    def model_history(self, customer: str, conversation: str) -> list[dict[str, Any]]:
        # Interrupted turns can contain unmatched tool calls. Keep them in the audit
        # UI, but do not replay their partial protocol into the next model request.
        with self.transaction() as db:
            self.require_owner(db, customer, conversation)
            return [json.loads(row[0]) for row in db.execute(
                "SELECT m.payload FROM messages m LEFT JOIN turns t "
                "ON t.conversation_id=m.conversation_id AND t.request_id=m.request_id AND m.turn_message=1 "
                "WHERE m.conversation_id=? AND (t.status IS NULL OR t.status!='failed') ORDER BY m.id",
                (conversation,)
            )]

    def append(self, conversation: str, request: str, message: dict[str, Any], *, turn_message: bool = True) -> None:
        with self.transaction() as db:
            db.execute("INSERT INTO messages(conversation_id,request_id,payload,created_at,turn_message) VALUES (?,?,?,?,?)",
                       (conversation, request, encode(message), time.time(), int(turn_message)))

    def context_state(self, customer: str, conversation: str) -> dict[str, Any]:
        from .context_builder import historical_observations
        with self.transaction() as db:
            self.require_owner(db, customer, conversation)
            status = db.execute("SELECT status FROM conversations WHERE id=?", (conversation,)).fetchone()[0]
            rows = db.execute("SELECT * FROM operations WHERE conversation_id=? AND (status IN "
                              "('pending_confirmation','executing','unknown','submitted') OR id IN "
                              "(SELECT id FROM operations WHERE conversation_id=? ORDER BY created_at DESC LIMIT 5)) "
                              "ORDER BY created_at", (conversation, conversation)).fetchall()
            operations = []
            for row in rows:
                value = {key: row[key] for key in ("id", "order_id", "status", "amount_cents", "order_version", "expires_at")}
                if value["status"] == "pending_confirmation" and value["expires_at"] <= time.time():
                    value["status"] = "expired"
                value["source"] = "operation_store"
                value["authorization_from_summary"] = False
                operations.append(value)
            audit = db.execute("SELECT tool,result,created_at FROM audit WHERE conversation_id=? "
                               "ORDER BY id DESC LIMIT 50", (conversation,)).fetchall()
        return {"conversation_status": status, "operations": operations,
                "order_observations": historical_observations(list(reversed(audit)))}

    def record(self, conversation: str, request: str, tool: str, arguments: Any, result: Any) -> None:
        with self.transaction() as db:
            db.execute("INSERT INTO audit(conversation_id,request_id,tool,arguments,result,created_at) "
                       "VALUES (?,?,?,?,?,?)", (conversation, request, tool, encode(arguments), encode(result), time.time()))

    def recover_interrupted_turns(self) -> int:
        """Call once at startup of the single service process, never per request."""
        with self.transaction() as db:
            rows = db.execute("SELECT conversation_id,request_id FROM turns WHERE status='running'").fetchall()
            for row in rows:
                response = {"error": "turn_interrupted", "message": "服务曾中断，请使用新的请求 ID 继续。"}
                db.execute("UPDATE turns SET status='failed',response=? WHERE conversation_id=? AND request_id=?",
                           (encode(response), row[0], row[1]))
                db.execute("UPDATE operations SET status='superseded',updated_at=? "
                           "WHERE conversation_id=? AND status='pending_confirmation'", (time.time(), row[0]))
            return len(rows)
