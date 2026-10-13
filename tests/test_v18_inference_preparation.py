from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import tarfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ecommerce_customer_service.serving.artifacts import BASE_REVISION, extract_verified_archive, file_sha256, verify_adapter, verify_local_base
from ecommerce_customer_service.serving.connection_check import check_endpoint
from ecommerce_customer_service.serving.context_builder import HFTokenCounter
from ecommerce_customer_service.serving.inference import BoundedGenerator, InferenceServer
from ecommerce_customer_service.serving.model import ModelProtocolError, terminal_content
from ecommerce_customer_service.serving.backend import MockBusinessBackend
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.storage import Store
from types import SimpleNamespace


class InferenceTransportTests(unittest.TestCase):
    """Transport fixtures exercise HTTP contracts, never claim actual SFT quality."""

    def setUp(self):
        self.calls = []
        self.result = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "fixture", "type": "function", "function": {"name": "list_orders", "arguments": "{}"}}],
            "local_input_tokens": 18, "local_generated_tokens": 9, "local_finish_reason": "stop"}

        def generate(messages, tools, maximum):
            self.calls.append((deepcopy(messages), deepcopy(tools), maximum))
            return deepcopy(self.result)

        self.server = InferenceServer(("127.0.0.1", 0), generate, api_key="test-key")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, payload=None, key="test-key"):
        request = Request(self.base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    def payload(self, **changes):
        return {"model": "ecommerce-v18-sft", "messages": [{"role": "user", "content": "查询订单"}], **changes}

    def test_authentication_checked_before_generation(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/v1/chat/completions", self.payload(), key="wrong")
        self.assertEqual(401, error.exception.code)
        self.assertFalse(self.calls)

    def test_model_identity_and_real_probe_contract(self):
        report = check_endpoint(self.base + "/v1", "test-key", generate=True)
        self.assertTrue(report["generation_checked"])
        self.assertFalse(report["online_quality_verified"])
        self.assertEqual(1, len(self.calls))

    def test_metadata_check_does_not_generate(self):
        report = check_endpoint(self.base + "/v1", "test-key")
        self.assertFalse(report["generation_checked"])
        self.assertFalse(self.calls)

    def test_wrong_protocol_hash_rejected(self):
        metadata = self.server.metadata()
        metadata["online_tools_sha256"] = "wrong"
        with patch.object(self.server, "metadata", return_value=metadata):
            with self.assertRaisesRegex(ValueError, "protocol_mismatch"):
                check_endpoint(self.base + "/v1", "test-key")

    def test_unsupported_request_does_not_generate(self):
        for changes in ({"model": "other"}, {"stream": True}, {"temperature": 0.5},
                        {"max_tokens": True}, {"max_tokens": 513}, {"tools": ["bad"]},
                        {"messages": [{"role": "developer", "content": "x"}]}, {"seed": 1},
                        {"messages": [{"role": "user", "content": ["not text"]}]},
                        {"messages": [{"role": "assistant", "tool_calls": ["bad"]}]}):
            with self.subTest(changes=changes), self.assertRaises(HTTPError) as error:
                self.request("/v1/chat/completions", self.payload(**changes))
            self.assertEqual(400, error.exception.code)
        self.assertFalse(self.calls)

    def test_generation_usage_and_limit(self):
        result = self.request("/v1/chat/completions", self.payload(max_tokens=123))
        self.assertEqual(27, result["usage"]["total_tokens"])
        self.assertEqual(123, self.calls[0][2])
        self.assertEqual("tool_calls", result["choices"][0]["finish_reason"])

    def test_truncation_not_returned_as_valid_action(self):
        self.result["local_finish_reason"] = "length"
        with self.assertRaises(HTTPError) as error:
            self.request("/v1/chat/completions", self.payload())
        self.assertEqual(422, error.exception.code)

    def test_busy_generation_not_queued_or_retried(self):
        self.server.inference_lock.acquire()
        try:
            with self.assertRaises(HTTPError) as error:
                self.request("/v1/chat/completions", self.payload())
            self.assertEqual(429, error.exception.code)
            self.assertFalse(self.calls)
        finally:
            self.server.inference_lock.release()

    def test_probe_rejects_old_cancel_and_customer_argument(self):
        for name, arguments in (("cancel_order", '{"order_id":"order_A","confirmed":true}'),
                                ("list_orders", '{"customer_id":"customer_alice"}')):
            self.result["tool_calls"][0]["function"] = {"name": name, "arguments": arguments}
            with self.assertRaisesRegex(ValueError, "probe_tool_protocol_mismatch"):
                check_endpoint(self.base + "/v1", "test-key", generate=True)


class ArtifactAndProtocolTests(unittest.TestCase):
    def test_archive_paths_and_links_rejected_without_partial_destination(self):
        for name, link in (("../outside", False), ("v18-sft-full-newgpu-r1/adapter_model.safetensors", True)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "malicious.tar.gz"
                with tarfile.open(archive, "w:gz") as bundle:
                    item = tarfile.TarInfo(name)
                    if link:
                        item.type = tarfile.SYMTYPE
                        item.linkname = "../outside"
                        bundle.addfile(item)
                    else:
                        item.size = 1
                        bundle.addfile(item, BytesIO(b"x"))
                with patch("ecommerce_customer_service.serving.artifacts.ARCHIVE_SHA256", file_sha256(archive)):
                    with self.assertRaises(ValueError):
                        extract_verified_archive(archive, root / "adapter")
                self.assertFalse((root / "adapter").exists())
                self.assertFalse((root / "outside").exists())

    def test_existing_destination_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            archive.write_bytes(b"fixture")
            destination = root / "adapter"
            destination.mkdir()
            marker = destination / "keep"
            marker.write_text("original", encoding="utf-8")
            with patch("ecommerce_customer_service.serving.artifacts.ARCHIVE_SHA256", file_sha256(archive)):
                with self.assertRaises(FileExistsError):
                    extract_verified_archive(archive, destination)
            self.assertEqual("original", marker.read_text(encoding="utf-8"))

    def test_base_revision_required_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen2", "vocab_size": 152064}))
            with self.assertRaisesRegex(ValueError, "revision_unverified"):
                verify_local_base(root)

    def test_pinned_base_still_requires_matching_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / BASE_REVISION
            root.mkdir()
            (root / "config.json").write_text(json.dumps({"model_type": "qwen2", "vocab_size": 152064}))
            (root / "tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "base_tokenizer_mismatch"):
                verify_local_base(root)

    def test_generation_limit_restored_after_failure(self):
        class Completer:
            generation_config = SimpleNamespace(max_new_tokens=512)
            def __call__(self, messages, tools):
                self.observed_limit = self.generation_config.max_new_tokens
                raise RuntimeError("fixture failure")
        completer = Completer()
        previous = completer.generation_config
        with self.assertRaises(RuntimeError):
            BoundedGenerator(completer)([], [], 17)
        self.assertEqual(17, completer.observed_limit)
        self.assertIs(previous, completer.generation_config)

    def test_bad_archive_not_extracted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fake.tar.gz"
            archive.write_bytes(b"wrong artifact")
            with self.assertRaisesRegex(ValueError, "archive_hash_mismatch"):
                extract_verified_archive(archive, root / "adapter")
            self.assertFalse((root / "adapter").exists())

    def test_incomplete_adapter_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "missing_or_linked_artifact"):
                verify_adapter(Path(temporary))

    def test_counter_normalizes_tool_arguments_like_inference(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
                return [1, 2, 3]
        tokenizer = Tokenizer()
        messages = [{"role": "user", "content": "查询"}, {"role": "assistant", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "get_order", "arguments": '{"order_id":"order_A"}'}}]}]
        self.assertEqual(3, HFTokenCounter(tokenizer).count(messages, []))
        self.assertEqual({"order_id": "order_A"}, tokenizer.arguments)
        self.assertIsInstance(messages[1]["tool_calls"][0]["function"]["arguments"], str)

    def test_terminal_json_unwraps_only_nonwrite_response(self):
        terminal = {"decision": "ASK_CLARIFICATION", "action_taken": False, "action_id": None, "message": "您指哪一单？"}
        self.assertEqual("您指哪一单？", terminal_content(json.dumps(terminal)))
        terminal.update(action_taken=True, action_id="fake_id", message="已取消")
        with self.assertRaises(ModelProtocolError):
            terminal_content(json.dumps(terminal))

    def test_invalid_terminal_schema_fails_closed(self):
        for raw in ('{"message":"已取消"}', '{bad', '{"decision":"x","action_taken":0,"action_id":null,"message":"x"}',
                    '{"decision":"x","action_taken":true,"action_taken":false,"action_id":null,"message":"x"}'):
            with self.subTest(raw=raw), self.assertRaises(ModelProtocolError):
                terminal_content(raw)

    def test_model_terminal_claim_never_changes_business_state(self):
        class ClaimModel:
            def complete(self, messages, tools):
                return {"content": json.dumps({"decision": "CANCELLED", "action_taken": True,
                                                "action_id": "fake", "message": "退款已到账"})}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = MockBusinessBackend(root / "business.sqlite")
            service = CustomerService(Store(root / "sessions.sqlite"), backend, ClaimModel())
            conversation = service.create_conversation("customer_alice")["conversation_id"]
            result = service.message("customer_alice", conversation, "request", "取消耳机")
            self.assertEqual("model_protocol_mismatch", result["error"])
            self.assertEqual("paid", backend.get_order("customer_alice", "order_A")["order"]["status"])
            self.assertNotIn("到账", result["message"])


if __name__ == "__main__":
    unittest.main()
