from __future__ import annotations

import hashlib
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from typing import Any
from urllib.parse import urlparse

from .service import CustomerService
from .storage import ServiceError, encode
from .telemetry import current_span


def load_auth(path: Path, *, create_demo: bool = False) -> dict[str, str]:
    """Demo bearer-token authentication; replace issuance with enterprise identity."""
    path = Path(path)
    if not path.exists():
        if not create_demo:
            raise ValueError("auth_file_missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens = {secrets.token_urlsafe(32): "customer_alice", secrets.token_urlsafe(32): "customer_bob"}
        # Exclusive create avoids overwriting credentials from another launch.
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"tokens": tokens}, stream, ensure_ascii=False, indent=2)
        path.chmod(0o600)
    value = json.loads(path.read_text(encoding="utf-8"))
    tokens = value.get("tokens")
    if not isinstance(tokens, dict) or not tokens:
        raise ValueError("invalid_auth_file")
    if any(not isinstance(token, str) or len(token) < 16 or not isinstance(customer, str) or not customer
           for token, customer in tokens.items()):
        raise ValueError("invalid_auth_entry")
    return {hashlib.sha256(token.encode()).hexdigest(): customer for token, customer in tokens.items()}


class OnlineServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: CustomerService, auth: dict[str, str],
                 *, demo_accounts: dict[str, str] | None = None) -> None:
        self.service, self.auth = service, auth
        if demo_accounts and address[0] not in {"127.0.0.1", "localhost"}:
            raise ValueError("demo_login_requires_loopback_listener")
        self.demo_accounts = demo_accounts or {}
        self.demo_sessions: dict[str, tuple[str, float]] = {}
        self.session_lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(address, OnlineHandler)


class OnlineHandler(BaseHTTPRequestHandler):
    server: OnlineServer
    protocol_version = "HTTP/1.0"

    def send_response(self, code: int, message: str | None = None) -> None:
        span = current_span()
        if span and span.kind == "http":
            span.set(http_status=code)
            if code >= 400:
                span.fail("http_" + str(code))
        super().send_response(code, message)

    def _send(self, status: int, body: Any) -> None:
        raw = encode(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _customer(self) -> str:
        auth = self.headers.get("Authorization", "")
        if not auth:
            cookies = SimpleCookie()
            try:
                cookies.load(self.headers.get("Cookie", ""))
                token = cookies["demo_session"].value if "demo_session" in cookies else ""
            except Exception:
                token = ""
            digest = hashlib.sha256(token.encode()).hexdigest()
            with self.server.session_lock:
                session = self.server.demo_sessions.get(digest)
            if session and session[1] > time.time():
                return session[0]
        if not auth.startswith("Bearer "):
            raise ServiceError("authentication_required", 401)
        digest = hashlib.sha256(auth[7:].encode()).hexdigest()
        customer = self.server.auth.get(digest)
        if customer is None:
            raise ServiceError("invalid_token", 401)
        return customer

    def _demo_login(self, body: dict[str, Any]) -> None:
        self._fields(body, {"account"})
        host = urlparse("http://" + self.headers.get("Host", "")).hostname
        if (not self.server.demo_accounts or self.client_address[0] != "127.0.0.1"
                or host not in {"127.0.0.1", "localhost"}):
            raise ServiceError("demo_login_not_available", 404)
        account = body["account"]
        if not isinstance(account, str) or account not in self.server.demo_accounts:
            raise ServiceError("invalid_demo_account")
        customer = self.server.demo_accounts[account]
        token = secrets.token_urlsafe(32)
        with self.server.session_lock:
            now = time.time()
            self.server.demo_sessions = {key: value for key, value in self.server.demo_sessions.items() if value[1] > now}
            if len(self.server.demo_sessions) >= 1000:
                raise ServiceError("too_many_demo_sessions", 429)
            self.server.demo_sessions[hashlib.sha256(token.encode()).hexdigest()] = (customer, now + 3600)
        with self.server.service.store.transaction() as db:
            latest = db.execute("SELECT id FROM conversations WHERE customer_id=? AND status='active' "
                                "ORDER BY created_at DESC LIMIT 1", (customer,)).fetchone()
        raw = encode({"account": account, "conversation_id": latest[0] if latest else None}).encode("utf-8")
        self.send_response(200)
        self.send_header("Set-Cookie", "demo_session=" + token + "; HttpOnly; SameSite=Strict; Path=/; Max-Age=3600")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding"):
            raise ServiceError("transfer_encoding_not_supported")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ServiceError("invalid_content_length") from exc
        if not 0 < length <= 65536:
            raise ServiceError("invalid_body_size", 413)
        self.connection.settimeout(10)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ServiceError("incomplete_body")
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ServiceError("invalid_json") from exc
        if not isinstance(value, dict):
            raise ServiceError("body_must_be_object")
        return value

    @staticmethod
    def _fields(body: dict[str, Any], fields: set[str]) -> None:
        if set(body) != fields:
            raise ServiceError("unexpected_or_missing_fields")

    def _dispatch(self, post: bool) -> None:
        path = urlparse(self.path).path
        if not post and path == "/healthz":
            self._send(200, {"status": "ok"})
            return
        if not post and path == "/api/ui-config":
            self._send(200, {"demo_accounts": list(self.server.demo_accounts),
                             "model_configured": self.server.service.model is not None})
            return
        if post:
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                raise ServiceError("cross_origin_request_rejected", 403)
        if post and path == "/api/demo-login":
            self._demo_login(self._body())
            return
        if not post and path == "/":
            raw = Path(__file__).with_name("index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(raw)
            return
        customer = self._customer()
        body = self._body() if post else {}
        parts = path.strip("/").split("/")
        service = self.server.service
        if not post and path == "/api/orders":
            result = service.orders(customer)
        elif post and path == "/api/conversations":
            self._fields(body, set())
            result = service.create_conversation(customer)
        elif len(parts) == 3 and parts[:2] == ["api", "conversations"] and not post:
            result = service.history(customer, parts[2])
        elif len(parts) == 4 and parts[:2] == ["api", "conversations"] and post:
            if parts[3] == "messages":
                self._fields(body, {"request_id", "text"})
                result = service.message(customer, parts[2], body["request_id"], body["text"])
            elif parts[3] == "preview-cancel":
                self._fields(body, {"order_id"})
                result = service.preview(customer, parts[2], body["order_id"])
            else:
                raise ServiceError("route_not_found", 404)
        elif len(parts) == 3 and parts[:2] == ["api", "operations"] and not post:
            result = service.get_operation(customer, parts[2])
        elif len(parts) == 4 and parts[:2] == ["api", "operations"] and parts[3] == "confirm" and post:
            self._fields(body, set())
            result = service.confirm(customer, parts[2])
        else:
            raise ServiceError("route_not_found", 404)
        self._send(200, result)

    def _handle(self, post: bool) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        label = "/".join(parts[:2]) if parts[0] in {"api", "healthz", ""} else "unknown_route"
        with self.server.service.telemetry.span("http", ("POST " if post else "GET ") + label):
            self._handle_inner(post)

    def _handle_inner(self, post: bool) -> None:
        if not self.server.slots.acquire(blocking=False):
            self._send(429, {"error": "server_busy"})
            return
        try:
            self._dispatch(post)
        except ServiceError as exc:
            self._send(exc.status, {"error": exc.code})
        except Exception:
            self._send(502, {"error": "service_or_backend_unavailable"})
        finally:
            self.server.slots.release()

    def do_GET(self) -> None:
        self._handle(False)

    def do_POST(self) -> None:
        self._handle(True)

    def log_message(self, format: str, *args: Any) -> None:
        # Do not log tokens, prompts, or order payloads.
        pass
