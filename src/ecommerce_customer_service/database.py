from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phone_last4 TEXT NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN ('standard', 'plus', 'vip'))
);

CREATE TABLE products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price_cents INTEGER NOT NULL CHECK (unit_price_cents >= 0),
    returnable INTEGER NOT NULL CHECK (returnable IN (0, 1)),
    return_window_days INTEGER NOT NULL CHECK (return_window_days >= 0)
);

CREATE TABLE inventory (
    product_id TEXT PRIMARY KEY REFERENCES products(id),
    quantity INTEGER NOT NULL CHECK (quantity >= 0)
);

CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    shipping_address TEXT NOT NULL,
    total_paid_cents INTEGER NOT NULL CHECK (total_paid_cents >= 0),
    coupon_code TEXT,
    discount_cents INTEGER NOT NULL DEFAULT 0 CHECK (discount_cents >= 0),
    currency TEXT NOT NULL DEFAULT 'CNY'
);

CREATE TABLE order_items (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    product_id TEXT NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_cents INTEGER NOT NULL CHECK (unit_price_cents >= 0),
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE payments (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    method TEXT NOT NULL,
    status TEXT NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents >= 0)
);

CREATE TABLE shipments (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    status TEXT NOT NULL,
    tracking_number TEXT,
    carrier TEXT,
    shipped_at TEXT,
    delivered_at TEXT
);

CREATE TABLE returns (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    order_item_id TEXT NOT NULL REFERENCES order_items(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE refunds (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    return_id TEXT REFERENCES returns(id),
    payment_id TEXT REFERENCES payments(id),
    amount_cents INTEGER NOT NULL CHECK (amount_cents >= 0),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE support_tickets (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    issue TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open one transaction-scoped connection and always release its file handle.

    ``sqlite3.Connection`` commits or rolls back when used directly as a context
    manager, but its ``__exit__`` method does not close the connection.  That can
    leave temporary databases locked on Windows until garbage collection runs.
    Keep the transaction semantics while making connection lifetime explicit.
    """

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()


def create_database(path: Path, *, overwrite: bool = False) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Database already exists: {path}")
        path.unlink()
    with connect(path) as connection:
        connection.executescript(SCHEMA)
    return path
