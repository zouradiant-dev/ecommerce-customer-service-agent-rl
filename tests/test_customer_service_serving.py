from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecommerce_customer_service.serving.api import OnlineServer, load_auth
from ecommerce_customer_service.serving.backend import BackendRejected, HttpBusinessBackend, MockBusinessBackend
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.runtime import exclusive_runtime
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.storage import ServiceError, Store, encode


class SequenceModel:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def complete(self, messages, tools):
        self.calls.append((messages[:], tools))
        return self.responses.pop(0)


def call(name, **arguments):
    return {"tool_calls": [{"id": "provider_call", "type": "function", "function": {
        "name": name, "arguments": encode(arguments)}}]}


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "sessions.sqlite")
        self.backend = MockBusinessBackend(self.root / "business.sqlite")
        self.service = CustomerService(self.store, self.backend)
        self.conversation = self.service.create_conversation("customer_alice")["conversation_id"]

    def tearDown(self):
        self.temp.cleanup()

    def preview(self, order="order_A"):
        return self.service.preview("customer_alice", self.conversation, order)

    def error(self, code, fn):
        with self.assertRaises(ServiceError) as context:
            fn()
        self.assertEqual(code, context.exception.code)

    def test_owner_bound_conversations_and_orders(self):
        self.error("conversation_not_found", lambda: self.service.history("customer_bob", self.conversation))
        self.error("conversation_not_found", lambda: self.service.preview("customer_bob", self.conversation, "order_D"))
        self.error("order_not_found", lambda: self.preview("order_D"))
        self.assertEqual(3, len(self.service.orders("customer_alice")["orders"]))

    def test_confirmation_executes_once_and_refund_is_not_completed(self):
        operation = self.preview()
        self.assertEqual("paid", self.backend.get_order("customer_alice", "order_A")["order"]["status"])
        first = self.service.confirm("customer_alice", operation["id"])
        second = self.service.confirm("customer_alice", operation["id"])
        self.assertEqual(first, second)
        self.assertEqual("succeeded", first["status"])
        self.assertEqual("pending", first["receipt"]["refund_status"])
        self.assertIn("尚未确认到账", self.service.operation_message(first))
        with self.backend.transaction() as db:
            self.assertEqual(1, db.execute("SELECT COUNT(*) FROM refunds").fetchone()[0])
            self.assertEqual(1, db.execute("SELECT COUNT(*) FROM cancellations").fetchone()[0])

    def test_other_customer_cannot_confirm_or_read_operation(self):
        operation = self.preview()
        self.error("operation_not_found", lambda: self.service.confirm("customer_bob", operation["id"]))
        self.error("operation_not_found", lambda: self.service.get_operation("customer_bob", operation["id"]))

    def test_changed_target_invalidates_old_confirmation(self):
        old = self.preview()
        new = self.preview("order_B")
        self.error("confirmation_invalidated", lambda: self.service.confirm("customer_alice", old["id"]))
        self.service.confirm("customer_alice", new["id"])
        self.assertEqual("paid", self.backend.get_order("customer_alice", "order_A")["order"]["status"])

    def test_user_message_invalidates_old_preview_and_text_confirmation_does_not_write(self):
        operation = self.preview()
        result = self.service.message("customer_alice", self.conversation, "request_1", "确认取消")
        self.assertEqual("manual", result["mode"])
        self.error("confirmation_invalidated", lambda: self.service.confirm("customer_alice", operation["id"]))
        self.assertEqual("paid", self.backend.get_order("customer_alice", "order_A")["order"]["status"])

    def test_confirmation_expiry(self):
        operation = self.preview()
        with self.store.transaction() as db:
            db.execute("UPDATE operations SET expires_at=0 WHERE id=?", (operation["id"],))
        self.error("confirmation_expired", lambda: self.service.confirm("customer_alice", operation["id"]))
        self.assertEqual("expired", self.service.get_operation("customer_alice", operation["id"])["status"])

    def test_order_changes_between_preview_and_execute(self):
        operation = self.preview()
        with self.backend.transaction() as db:
            db.execute("UPDATE orders SET status='shipped',version=version+1 WHERE id='order_A'")
        result = self.service.confirm("customer_alice", operation["id"])
        self.assertEqual("failed", result["status"])
        self.assertEqual("order_changed_reconfirm_required", result["error"])
        with self.backend.transaction() as db:
            self.assertEqual(0, db.execute("SELECT COUNT(*) FROM refunds").fetchone()[0])

    def test_backend_idempotency_rejects_parameter_reuse(self):
        receipt = self.backend.cancel("customer_alice", "order_A", "stable_operation", 1)
        self.assertEqual(receipt, self.backend.cancel("customer_alice", "order_A", "stable_operation", 1))
        with self.assertRaises(BackendRejected):
            self.backend.cancel("customer_alice", "order_B", "stable_operation", 1)

    def test_parallel_confirmation_is_one_backend_write(self):
        operation = self.preview()
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: self.service.confirm("customer_alice", operation["id"]), range(6)))
        result = self.service.get_operation("customer_alice", operation["id"])
        self.assertEqual("succeeded", result["status"])
        with self.backend.transaction() as db:
            self.assertEqual(1, db.execute("SELECT COUNT(*) FROM cancellations").fetchone()[0])

    def test_message_replay_returns_cached_result_and_rejects_different_text(self):
        first = self.service.message("customer_alice", self.conversation, "request_1", "查询订单")
        self.assertEqual(first, self.service.message("customer_alice", self.conversation, "request_1", "查询订单"))
        self.error("request_id_parameter_mismatch", lambda: self.service.message(
            "customer_alice", self.conversation, "request_1", "换一笔订单"))
        self.assertEqual(2, len(self.service.history("customer_alice", self.conversation)["messages"]))

    def test_restart_preserves_history_preview_and_credentials(self):
        self.service.message("customer_alice", self.conversation, "request_1", "查询订单")
        operation = self.preview()
        fresh = CustomerService(Store(self.root / "sessions.sqlite"), MockBusinessBackend(self.root / "business.sqlite"))
        self.assertEqual(3, len(fresh.history("customer_alice", self.conversation)["messages"]))
        self.assertEqual("succeeded", fresh.confirm("customer_alice", operation["id"])["status"])
        auth_path = self.root / "demo-auth.json"
        auth = load_auth(auth_path, create_demo=True)
        self.assertEqual(auth, load_auth(auth_path, create_demo=True))

    def test_lost_response_after_commit_is_reconciled_without_resubmit(self):
        operation = self.preview()
        original = self.backend.cancel
        calls = []
        def lose(*args):
            calls.append(args)
            original(*args)
            raise TimeoutError("lost_ack")
        with patch.object(self.backend, "cancel", side_effect=lose):
            result = self.service.confirm("customer_alice", operation["id"])
            self.assertEqual("succeeded", result["status"])
            self.service.confirm("customer_alice", operation["id"])
        self.assertEqual(1, len(calls))

    def test_unknown_result_blocks_new_operation_across_conversations(self):
        operation = self.preview()
        with patch.object(self.backend, "cancel", side_effect=TimeoutError), patch.object(self.backend, "lookup", return_value=None):
            self.assertEqual("unknown", self.service.confirm("customer_alice", operation["id"])["status"])
            self.assertEqual("unknown", self.service.confirm("customer_alice", operation["id"])["status"])
            second = self.service.create_conversation("customer_alice")["conversation_id"]
            self.error("order_has_unresolved_operation", lambda: self.service.preview("customer_alice", second, "order_A"))

    def test_restart_recovers_committed_write_with_missing_local_receipt(self):
        operation = self.preview()
        with self.store.transaction() as db:
            db.execute("UPDATE operations SET status='executing' WHERE id=?", (operation["id"],))
        self.backend.cancel("customer_alice", "order_A", operation["id"], operation["order_version"])
        fresh = CustomerService(Store(self.root / "sessions.sqlite"), self.backend)
        self.assertEqual(1, fresh.recover()["uncertain_operations"])
        self.assertEqual("succeeded", fresh.get_operation("customer_alice", operation["id"])["status"])

    def test_recovery_drops_partial_model_protocol_but_preserves_audit_history(self):
        with self.store.transaction() as db:
            db.execute("INSERT INTO turns VALUES (?,?,'hello','running',NULL,0)", (self.conversation, "crashed"))
        self.store.append(self.conversation, "crashed", call("list_orders"))
        self.assertEqual(1, self.service.recover()["interrupted_turns"])
        self.assertEqual([], self.store.model_history("customer_alice", self.conversation))
        self.assertEqual(1, len(self.store.history("customer_alice", self.conversation)))

    def test_model_queries_and_previews_without_executing(self):
        model = SequenceModel([call("list_orders"), call("preview_cancel", order_id="order_A"),
                               {"content": "订单已取消，退款到账"}])
        self.service.model = model
        result = self.service.message("customer_alice", self.conversation, "request_1", "取消耳机")
        self.assertIn("待确认", result["message"])
        self.assertNotIn("退款到账", result["message"])
        self.assertEqual("paid", self.backend.get_order("customer_alice", "order_A")["order"]["status"])
        self.assertEqual(3, len(model.calls))
        self.assertEqual("list_orders", model.calls[1][0][-1]["name"])

    def test_model_cannot_execute_cancel_or_impersonate_customer(self):
        model = SequenceModel([call("cancel_order", order_id="order_A", confirmed=True),
                               call("get_order", order_id="order_D", customer_id="customer_bob"), {"content": "需要用户确认"}])
        self.service.model = model
        self.service.message("customer_alice", self.conversation, "request_1", "操作")
        self.assertEqual("paid", self.backend.get_order("customer_alice", "order_A")["order"]["status"])
        self.assertIn("tool_not_allowed", model.calls[1][0][-1]["content"])
        self.assertIn("tool_not_allowed", model.calls[2][0][-1]["content"])

    def test_handoff_stops_remaining_tools_and_future_writes(self):
        first = call("escalate_to_human", reason="请求人工")
        first["tool_calls"] += call("preview_cancel", order_id="order_A")["tool_calls"]
        self.service.model = SequenceModel([first])
        result = self.service.message("customer_alice", self.conversation, "request_1", "人工")
        self.assertTrue(result["handed_off"])
        self.error("conversation_handed_off", lambda: self.preview())
        self.error("conversation_handed_off", lambda: self.service.message("customer_alice", self.conversation, "next", "取消"))
        self.assertEqual([], self.service.history("customer_alice", self.conversation)["operations"])

    def test_overlapping_turns_are_rejected(self):
        entered, release = threading.Event(), threading.Event()
        class BlockingModel:
            def complete(self, messages, tools):
                entered.set()
                release.wait(5)
                return {"content": "请提供订单"}
        self.service.model = BlockingModel()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.service.message, "customer_alice", self.conversation, "first", "查询")
            self.assertTrue(entered.wait(3))
            try:
                self.error("request_in_progress", lambda: self.service.message("customer_alice", self.conversation, "first", "查询"))
                self.error("conversation_busy", lambda: self.service.message("customer_alice", self.conversation, "second", "取消"))
            finally:
                release.set()
            pending.result(5)

    def test_no_model_fallback_on_endpoint_failure(self):
        class BrokenModel:
            def complete(self, messages, tools):
                raise ConnectionError("unavailable")
        self.service.model = BrokenModel()
        result = self.service.message("customer_alice", self.conversation, "request_1", "取消")
        self.assertEqual("model_or_turn_failed", result["error"])
        self.assertEqual([], self.store.model_history("customer_alice", self.conversation))

    def test_malformed_business_receipt_is_unknown_not_success(self):
        operation = self.preview()
        bad = {"operation_id": operation["id"], "order_id": "wrong_order", "status": "succeeded"}
        with patch.object(self.backend, "cancel", return_value=bad):
            self.assertEqual("unknown", self.service.confirm("customer_alice", operation["id"])["status"])

    def test_malformed_business_preview_is_rejected_before_confirmation(self):
        with patch.object(self.backend, "preview_cancel", return_value={
            "order_id": "order_A", "order_version": 1, "refund_cents": "19900"}):
            self.error("invalid_business_preview", lambda: self.preview())
        self.assertEqual([], self.service.history("customer_alice", self.conversation)["operations"])

    def test_exclusive_runtime_lock(self):
        with exclusive_runtime(self.root):
            with self.assertRaises(RuntimeError):
                with exclusive_runtime(self.root):
                    pass


class HttpTests(unittest.TestCase):
    def setUp(self):
        ServingTests.setUp(self)
        self.token_a, self.token_b = "a" * 32, "b" * 32
        auth = {hashlib.sha256(self.token_a.encode()).hexdigest(): "customer_alice",
                hashlib.sha256(self.token_b.encode()).hexdigest(): "customer_bob"}
        self.server = OnlineServer(("127.0.0.1", 0), self.service, auth)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        ServingTests.tearDown(self)

    def request(self, path, body=None, token=None):
        request = Request(self.base + path, data=encode(body).encode() if body is not None else None,
                          headers={"Authorization": "Bearer " + (token or self.token_a), "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_http_auth_ownership_and_body_validation(self):
        self.assertEqual(401, self.request("/api/orders", token="bad")[0])
        self.assertEqual(404, self.request("/api/conversations/" + self.conversation, token=self.token_b)[0])
        self.assertEqual(400, self.request("/api/conversations", {"customer_id": "customer_bob"})[0])
        self.assertEqual(400, self.request("/api/conversations", [])[0])

    def test_http_create_chat_preview_confirm_restore(self):
        status, result = self.request("/api/conversations", {})
        self.assertEqual(200, status)
        conversation = result["conversation_id"]
        self.assertEqual(200, self.request("/api/conversations/" + conversation + "/messages",
                                          {"request_id": "http_request", "text": "查询订单"})[0])
        status, operation = self.request("/api/conversations/" + conversation + "/preview-cancel", {"order_id": "order_A"})
        self.assertEqual(200, status)
        status, receipt = self.request("/api/operations/" + operation["id"] + "/confirm", {})
        self.assertEqual("succeeded", receipt["status"])
        self.assertEqual(200, self.request("/api/conversations/" + conversation)[0])
        self.assertEqual(200, self.request("/api/operations/" + operation["id"])[0])

    def test_demo_cookie_login_without_exposing_bearer_token(self):
        self.server.demo_accounts = {"alice": "customer_alice", "bob": "customer_bob"}
        request = Request(self.base + "/api/demo-login", data=encode({"account": "alice"}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            cookie = response.headers["Set-Cookie"]
            body = json.loads(response.read())
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertNotIn("token", body)
        self.assertEqual(self.conversation, body["conversation_id"])
        with urlopen(Request(self.base + "/api/orders", headers={"Cookie": cookie.split(";")[0]}), timeout=5) as response:
            self.assertEqual(3, len(json.loads(response.read())["orders"]))

    def test_demo_login_disabled_outside_explicit_demo_mode(self):
        self.assertEqual(404, self.request("/api/demo-login", {"account": "alice"})[0])
        with self.assertRaises(ValueError):
            OnlineServer(("0.0.0.0", 0), self.service, {}, demo_accounts={"alice": "customer_alice"})

    def test_cookie_session_expiry_requires_new_login(self):
        self.server.demo_accounts = {"alice": "customer_alice"}
        request = Request(self.base + "/api/demo-login", data=encode({"account": "alice"}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            cookie = response.headers["Set-Cookie"].split(";")[0]
        with self.server.session_lock:
            self.server.demo_sessions = {key: (customer, 0) for key, (customer, expiry) in self.server.demo_sessions.items()}
        with self.assertRaises(HTTPError) as context:
            urlopen(Request(self.base + "/api/orders", headers={"Cookie": cookie}), timeout=5)
        self.assertEqual(401, context.exception.code)

    def test_cross_origin_posts_are_rejected(self):
        request = Request(self.base + "/api/conversations", data=b"{}", headers={
            "Authorization": "Bearer " + self.token_a, "Origin": "https://other.invalid"})
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=5)
        self.assertEqual(403, context.exception.code)

    def test_actual_model_http_transport_with_compatible_endpoint(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        received = []
        class Provider(BaseHTTPRequestHandler):
            def do_POST(handler):
                payload = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
                received.append(payload)
                raw = encode({"choices": [{"message": {"content": "请告诉我订单号"}}]}).encode()
                handler.send_response(200)
                handler.send_header("Content-Length", str(len(raw)))
                handler.end_headers()
                handler.wfile.write(raw)
            def log_message(self, *args): pass
        provider = HTTPServer(("127.0.0.1", 0), Provider)
        thread = threading.Thread(target=provider.serve_forever, daemon=True)
        thread.start()
        try:
            self.service.model = OpenAIChatModel("http://127.0.0.1:" + str(provider.server_port) + "/v1", "served-adapter")
            _, response = self.request("/api/conversations/" + self.conversation + "/messages",
                                       {"request_id": "model_request", "text": "查询订单"})
            self.assertEqual("model", response["mode"])
            self.assertEqual("served-adapter", received[0]["model"])
            self.assertFalse(received[0]["stream"])
            self.assertEqual(4, len(received[0]["tools"]))
        finally:
            provider.shutdown()
            provider.server_close()
            thread.join(3)


class AdapterTests(unittest.TestCase):
    def test_http_backend_contract_and_ownership(self):
        adapter = HttpBusinessBackend("https://example.invalid", "not-a-real-key")
        order = {"id": "order_A", "customer_id": "customer_alice"}
        with patch.object(adapter, "_request", side_effect=[{"orders": [order]}, {"order": order},
                          {"order": order}, {"order_id": "order_A", "order_version": 1, "refund_cents": 19900},
                          {"status": "pending"}, None]) as request:
            adapter.list_orders("customer_alice")
            adapter.get_order("customer_alice", "order_A")
            adapter.preview_cancel("customer_alice", "order_A")
            adapter.cancel("customer_alice", "order_A", "op_1", 1)
            adapter.lookup("customer_alice", "op_1")
        self.assertEqual(6, request.call_count)
        self.assertEqual("op_1", request.call_args_list[4].args[1]["operation_id"])
        with patch.object(adapter, "_request", return_value={"orders": [{"customer_id": "customer_bob"}]}):
            with self.assertRaises(ValueError):
                adapter.list_orders("customer_alice")

    def test_http_rejection_and_ambiguous_failure_are_distinct_and_not_retried(self):
        adapter = HttpBusinessBackend("https://example.invalid", "not-a-real-key")
        def error(code, body):
            return HTTPError("https://example.invalid/cancellations", code, "error", {}, BytesIO(encode(body).encode()))
        with patch("ecommerce_customer_service.serving.backend.urlopen", side_effect=error(409, {"status": "rejected", "code": "changed"})) as request:
            with self.assertRaises(BackendRejected):
                adapter.cancel("customer_alice", "order_A", "op_1", 1)
            self.assertEqual(1, request.call_count)
        with patch("ecommerce_customer_service.serving.backend.urlopen", side_effect=error(503, {})) as request:
            with self.assertRaises(HTTPError):
                adapter.cancel("customer_alice", "order_A", "op_1", 1)
            self.assertEqual(1, request.call_count)
        with patch("ecommerce_customer_service.serving.backend.urlopen", side_effect=error(404, {})):
            self.assertIsNone(adapter.lookup("customer_alice", "op_1"))


if __name__ == "__main__":
    unittest.main()
