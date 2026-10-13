from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .storage import ServiceError, encode


class BackendRejected(Exception):
    """A definitive rejection, as opposed to an ambiguous network failure."""


class BusinessBackend(Protocol):
    def list_orders(self, customer: str) -> dict[str, Any]: ...
    def get_order(self, customer: str, order: str) -> dict[str, Any]: ...
    def preview_cancel(self, customer: str, order: str) -> dict[str, Any]: ...
    def cancel(self, customer: str, order: str, operation: str, version: int) -> dict[str, Any]: ...
    def lookup(self, customer: str, operation: str) -> dict[str, Any] | None: ...


class MockBusinessBackend:
    """Shared durable order service. No real payment or enterprise integration."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, product TEXT NOT NULL,
                    status TEXT NOT NULL, paid_cents INTEGER NOT NULL, version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cancellations (
                    operation_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
                    order_id TEXT NOT NULL, request TEXT NOT NULL, receipt TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS refunds (
                    id TEXT PRIMARY KEY, order_id TEXT NOT NULL UNIQUE REFERENCES orders(id),
                    amount_cents INTEGER NOT NULL, status TEXT NOT NULL
                );
            """)
            db.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?)", [
                ("order_A", "customer_alice", "无线耳机", "paid", 19900, 1),
                ("order_B", "customer_alice", "键盘", "paid", 29900, 1),
                ("order_C", "customer_alice", "鼠标", "shipped", 9900, 1),
                ("order_D", "customer_bob", "显示器", "paid", 129900, 1),
            ])

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
    def _owned(db: sqlite3.Connection, customer: str, order: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM orders WHERE id=? AND customer_id=?", (order, customer)).fetchone()
        if row is None:
            raise ServiceError("order_not_found", 404)
        return row

    def list_orders(self, customer: str) -> dict[str, Any]:
        with self.transaction() as db:
            return {"orders": [dict(row) for row in db.execute(
                "SELECT * FROM orders WHERE customer_id=? ORDER BY id", (customer,)
            )]}

    def get_order(self, customer: str, order: str) -> dict[str, Any]:
        with self.transaction() as db:
            row = dict(self._owned(db, customer, order))
            refund = db.execute("SELECT * FROM refunds WHERE order_id=?", (order,)).fetchone()
            return {"order": row, "refund": dict(refund) if refund else None}

    def preview_cancel(self, customer: str, order: str) -> dict[str, Any]:
        row = self.get_order(customer, order)["order"]
        if row["status"] not in {"pending", "paid"}:
            raise ServiceError("order_not_cancellable", 409)
        return {"order_id": order, "order_version": row["version"],
                "refund_cents": row["paid_cents"] if row["status"] == "paid" else 0}

    def cancel(self, customer: str, order: str, operation: str, version: int) -> dict[str, Any]:
        request = encode({"customer": customer, "order": order, "version": version})
        with self.transaction() as db:
            existing = db.execute("SELECT * FROM cancellations WHERE operation_id=?", (operation,)).fetchone()
            if existing:
                if existing["request"] != request:
                    raise BackendRejected("idempotency_parameter_mismatch")
                return json.loads(existing["receipt"])
            row = self._owned(db, customer, order)
            if row["version"] != version or row["status"] not in {"pending", "paid"}:
                raise BackendRejected("order_changed_reconfirm_required")
            amount = row["paid_cents"] if row["status"] == "paid" else 0
            refund_id = "refund_" + operation if amount else None
            db.execute("UPDATE orders SET status='cancelled',version=version+1 WHERE id=?", (order,))
            if refund_id:
                db.execute("INSERT INTO refunds VALUES (?,?,?,'pending')", (refund_id, order, amount))
            receipt = {"operation_id": operation, "order_id": order, "status": "succeeded",
                       "order_status": "cancelled", "refund_id": refund_id,
                       "refund_cents": amount, "refund_status": "pending" if refund_id else None}
            db.execute("INSERT INTO cancellations VALUES (?,?,?,?,?)",
                       (operation, customer, order, request, encode(receipt)))
            return receipt

    def lookup(self, customer: str, operation: str) -> dict[str, Any] | None:
        with self.transaction() as db:
            row = db.execute("SELECT receipt FROM cancellations WHERE operation_id=? AND customer_id=?",
                             (operation, customer)).fetchone()
            return json.loads(row[0]) if row else None


class HttpBusinessBackend:
    """Contract adapter; the enterprise service must implement these endpoints.

    Writes are never retried here. The remote service MUST durably deduplicate
    operation_id and expose lookup; missing lookup is not proof of no execution.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 15) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("invalid_business_url")
        self.base_url, self.token, self.timeout = base_url.rstrip("/"), token, timeout

    def _request(self, path: str, payload: Any = None, *, missing_ok: bool = False) -> Any:
        req = Request(self.base_url + path,
                      data=encode(payload).encode("utf-8") if payload is not None else None,
                      headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
                      method="POST" if payload is not None else "GET")
        try:
            with urlopen(req, timeout=self.timeout) as response:
                raw = response.read(1_048_577)
                if len(raw) > 1_048_576:
                    raise ValueError("business_response_too_large")
                return json.loads(raw)
        except HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            # Only the documented rejected response is a definitive write failure.
            raw = exc.read(4096)
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                body = {}
            if exc.code in {400, 403, 404, 409, 422} and body.get("status") == "rejected":
                raise BackendRejected(str(body.get("code", "business_rejected"))) from exc
            raise

    def list_orders(self, customer: str) -> dict[str, Any]:
        result = self._request("/customers/" + quote(customer, safe="") + "/orders")
        if (not isinstance(result, dict) or not isinstance(result.get("orders"), list)
                or any(not isinstance(row, dict) or row.get("customer_id") != customer for row in result["orders"])):
            raise ValueError("invalid_order_ownership_response")
        return result

    def get_order(self, customer: str, order: str) -> dict[str, Any]:
        result = self._request("/customers/" + quote(customer, safe="") + "/orders/" + quote(order, safe=""))
        row = result.get("order") if isinstance(result, dict) else None
        if not isinstance(row, dict) or row.get("id") != order or row.get("customer_id") != customer:
            raise ValueError("invalid_order_ownership_response")
        return result

    def preview_cancel(self, customer: str, order: str) -> dict[str, Any]:
        self.get_order(customer, order)
        return self._request("/cancellations/preview", {"customer_id": customer, "order_id": order})

    def cancel(self, customer: str, order: str, operation: str, version: int) -> dict[str, Any]:
        return self._request("/cancellations", {"customer_id": customer, "order_id": order,
                                             "operation_id": operation, "expected_version": version})

    def lookup(self, customer: str, operation: str) -> dict[str, Any] | None:
        return self._request("/cancellations/" + quote(operation, safe="") + "?" +
                             urlencode({"customer_id": customer}), missing_ok=True)
