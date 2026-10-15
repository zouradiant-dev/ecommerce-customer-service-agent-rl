from __future__ import annotations

import json
import re
from hashlib import sha256
import time
from typing import Any
from uuid import uuid4

from .backend import BackendRejected, BusinessBackend
from .context_builder import ContextBuilder
from .model import ChatModel, ModelOutputError, ModelProtocolError, ModelServiceError, terminal_content
from .policy import get_policy
from .storage import ServiceError, Store, encode
from .telemetry import ObservedBackend, Telemetry, traced
from .target_binding import TARGET_MESSAGES, preview_claim, require_target
from .knowledge import KnowledgeIndex, augment_policy


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ServiceError("invalid_identifier")
    return value


class CustomerService:
    def __init__(self, store: Store, backend: BusinessBackend, model: ChatModel | None = None,
                 *, confirmation_ttl: float = 300, max_steps: int = 8,
                 context_builder: ContextBuilder | None = None, policy_profile: str = "baseline_v1",
                 knowledge: KnowledgeIndex | None = None) -> None:
        self.store, self.model = store, model
        self.telemetry = Telemetry(store)
        self.backend = ObservedBackend(backend, self.telemetry)
        self.context_builder = context_builder or ContextBuilder()
        self.confirmation_ttl, self.max_steps = confirmation_ttl, max_steps
        self.policy = get_policy(policy_profile)
        self.knowledge = knowledge
        if knowledge:
            self.policy = augment_policy(self.policy)

    def recover(self) -> dict[str, int]:
        spans = self.telemetry.recover()
        turns = self.store.recover_interrupted_turns()
        with self.store.transaction() as db:
            count = db.execute("UPDATE operations SET status='unknown',error='service_interrupted',updated_at=? "
                               "WHERE status='executing'", (time.time(),)).rowcount
        return {"interrupted_turns": turns, "uncertain_operations": count, "interrupted_spans": spans}

    def create_conversation(self, customer: str) -> dict[str, Any]:
        conversation = "conv_" + uuid4().hex
        with self.store.transaction() as db:
            db.execute("INSERT INTO conversations(id,customer_id,created_at) VALUES (?,?,?)",
                       (conversation, customer, time.time()))
        return {"conversation_id": conversation}

    @staticmethod
    def _active(db: Any, conversation: str) -> None:
        if db.execute("SELECT status FROM conversations WHERE id=?", (conversation,)).fetchone()[0] != "active":
            raise ServiceError("conversation_handed_off", 409)

    @staticmethod
    def _idle(db: Any, conversation: str) -> None:
        running = db.execute("SELECT 1 FROM turns WHERE conversation_id=? AND status='running'",
                             (conversation,)).fetchone()
        writing = db.execute("SELECT 1 FROM operations WHERE conversation_id=? AND status='executing'",
                             (conversation,)).fetchone()
        if running or writing:
            raise ServiceError("conversation_busy", 409)

    def history(self, customer: str, conversation: str) -> dict[str, Any]:
        messages = self.store.history(customer, conversation)
        with self.store.transaction() as db:
            db.execute("UPDATE operations SET status='expired',updated_at=? WHERE conversation_id=? "
                       "AND status='pending_confirmation' AND expires_at<=?", (time.time(), conversation, time.time()))
            operations = [self._public(row) for row in db.execute(
                "SELECT * FROM operations WHERE conversation_id=? ORDER BY created_at", (conversation,))]
            status = db.execute("SELECT status FROM conversations WHERE id=?", (conversation,)).fetchone()[0]
        return {"conversation_id": conversation, "status": status, "messages": messages,
                "operations": operations, "mode": "model" if self.model else "manual"}

    @traced("request")
    def orders(self, customer: str) -> dict[str, Any]:
        return self.backend.list_orders(customer)

    @traced("request")
    def order(self, customer: str, order: str) -> dict[str, Any]:
        return self.backend.get_order(customer, identifier(order))

    def search_policies(self, query: str) -> dict[str, Any]:
        if self.knowledge is None:
            raise ServiceError("knowledge_not_configured",404)
        with self.telemetry.span("retrieval","search_policies") as span:
            result = self.knowledge.search(query)
            span.set(index_sha256=result['index_sha256'],hit_count=len(result['hits']))
            return result

    def _preview(self, customer: str, conversation: str, order: str,
                 *, request_id: str | None = None, matched_version: int | None = None) -> dict[str, Any]:
        order = identifier(order)
        preview = self.backend.preview_cancel(customer, order)
        if (not isinstance(preview, dict) or preview.get("order_id") != order or type(preview.get("order_version")) is not int
                or type(preview.get("refund_cents")) is not int or preview["refund_cents"] < 0):
            raise ServiceError("invalid_business_preview", 502)
        if matched_version is not None and preview["order_version"] != matched_version:
            raise ServiceError("target_order_changed", 409)
        now, operation = time.time(), "op_" + uuid4().hex
        with self.store.transaction() as db:
            self.store.require_owner(db, customer, conversation)
            self._active(db, conversation)
            if request_id is None:
                self._idle(db, conversation)
            else:
                running = db.execute("SELECT status FROM turns WHERE conversation_id=? AND request_id=?",
                                     (conversation, request_id)).fetchone()
                if not running or running[0] != "running":
                    raise ServiceError("turn_not_active", 409)
            db.execute("UPDATE operations SET status='superseded',updated_at=? "
                       "WHERE conversation_id=? AND status='pending_confirmation'", (now, conversation))
            live = db.execute("SELECT 1 FROM operations WHERE customer_id=? AND order_id=? "
                              "AND status IN ('pending_confirmation','executing','unknown','submitted')",
                              (customer, order)).fetchone()
            if live:
                raise ServiceError("order_has_unresolved_operation", 409)
            db.execute("INSERT INTO operations VALUES (?,?,?,?,?,?,?, ?,NULL,NULL,?,?)",
                       (operation, conversation, customer, order, preview["refund_cents"],
                        preview["order_version"], "pending_confirmation", now + self.confirmation_ttl, now, now))
        return self.get_operation(customer, operation)

    @traced("request")
    def preview(self, customer: str, conversation: str, order: str) -> dict[str, Any]:
        # Authorize before even querying an external order service.
        with self.store.transaction() as db:
            self.store.require_owner(db, customer, conversation)
            self._active(db, conversation)
            self._idle(db, conversation)
        result = self._preview(customer, conversation, order)
        self.store.record(conversation, "manual_preview", "preview_cancel", {"order_id": order}, result)
        self.store.append(conversation, "manual_preview", {"role": "assistant", "content": self._preview_message(result)}, turn_message=False)
        return result

    @staticmethod
    def _preview_message(operation: dict[str, Any]) -> str:
        return (f"待确认：取消订单 {operation['order_id']}，预计退款 {operation['amount_cents']/100:.2f} 元。"
                "请核对下方申请，并点击“确认取消”。")

    @staticmethod
    def _public(row: Any) -> dict[str, Any]:
        return {key: (json.loads(row[key]) if key == "receipt" and row[key] else row[key]) for key in
                ("id", "conversation_id", "order_id", "amount_cents", "order_version", "status",
                 "expires_at", "receipt", "error")}

    @staticmethod
    def _validate_receipt(row: Any, receipt: Any) -> None:
        if (not isinstance(receipt, dict) or receipt.get("operation_id") != row["id"]
                or receipt.get("order_id") != row["order_id"]
                or receipt.get("status") not in {"pending", "succeeded", "failed"}):
            raise ValueError("invalid_business_receipt")
        if receipt["status"] == "succeeded":
            if (receipt.get("order_status") != "cancelled" or type(receipt.get("refund_cents")) is not int
                    or receipt.get("refund_cents") != row["amount_cents"]):
                raise ValueError("business_receipt_mismatch")
            if row["amount_cents"] and (not isinstance(receipt.get("refund_id"), str)
                    or not receipt["refund_id"] or receipt.get("refund_status") not in {"pending", "succeeded", "failed"}):
                raise ValueError("invalid_refund_receipt")

    def _save_receipt(self, operation: str, receipt: dict[str, Any]) -> None:
        status = "submitted" if receipt["status"] == "pending" else receipt["status"]
        with self.store.transaction() as db:
            db.execute("UPDATE operations SET status=?,receipt=?,error=NULL,updated_at=? WHERE id=? "
                       "AND status IN ('executing','unknown','submitted')",
                       (status, encode(receipt), time.time(), operation))

    @traced("request")
    def get_operation(self, customer: str, operation: str) -> dict[str, Any]:
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM operations WHERE id=? AND customer_id=?", (operation, customer)).fetchone()
            if not row:
                raise ServiceError("operation_not_found", 404)
            if row["status"] == "pending_confirmation" and row["expires_at"] <= time.time():
                db.execute("UPDATE operations SET status='expired',updated_at=? WHERE id=?", (time.time(), operation))
                row = db.execute("SELECT * FROM operations WHERE id=?", (operation,)).fetchone()
            result = self._public(row)
        if row["status"] in {"unknown", "submitted"}:
            try:
                receipt = self.backend.lookup(customer, operation)
                if receipt is not None:
                    self._validate_receipt(row, receipt)
                    self._save_receipt(operation, receipt)
                    with self.store.transaction() as db:
                        result = self._public(db.execute("SELECT * FROM operations WHERE id=?", (operation,)).fetchone())
            except Exception:
                # A failed/missing lookup never justifies resubmission.
                pass
        return result

    @traced("operation")
    def confirm(self, customer: str, operation: str) -> dict[str, Any]:
        now = time.time()
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM operations WHERE id=? AND customer_id=?", (operation, customer)).fetchone()
            if not row:
                raise ServiceError("operation_not_found", 404)
            if row["status"] in {"executing", "unknown", "submitted", "succeeded", "failed"}:
                return self._public(row)
            self._active(db, row["conversation_id"])
            self._idle(db, row["conversation_id"])
            if row["status"] != "pending_confirmation":
                raise ServiceError("confirmation_invalidated", 409)
            if row["expires_at"] <= now:
                raise ServiceError("confirmation_expired", 409)
            db.execute("UPDATE operations SET status='executing',updated_at=? WHERE id=?", (now, operation))
        self.store.record(row["conversation_id"], operation, "user_confirm", {"operation_id": operation}, {"authorized": True})
        try:
            receipt = self.backend.cancel(customer, row["order_id"], operation, row["order_version"])
            self._validate_receipt(row, receipt)
            self._save_receipt(operation, receipt)
        except BackendRejected as exc:
            with self.store.transaction() as db:
                db.execute("UPDATE operations SET status='failed',error=?,updated_at=? WHERE id=?",
                           (str(exc), time.time(), operation))
        except Exception:
            with self.store.transaction() as db:
                db.execute("UPDATE operations SET status='unknown',error='business_result_unknown',updated_at=? WHERE id=?",
                           (time.time(), operation))
        result = self.get_operation(customer, operation)
        self.store.record(row["conversation_id"], operation, "cancel_order", {"order_id": row["order_id"]}, result)
        self.store.append(row["conversation_id"], operation, {"role": "assistant", "content": self.operation_message(result)}, turn_message=False)
        return result

    @staticmethod
    def operation_message(operation: dict[str, Any]) -> str:
        status, order = operation["status"], operation["order_id"]
        if status == "succeeded":
            receipt = operation["receipt"]
            refund = receipt.get("refund_status")
            ending = {"pending": "退款处理中，尚未确认到账。", "succeeded": "后台报告退款成功。",
                      "failed": "退款失败，请联系人工处理。", None: "无需退款。"}[refund]
            return f"订单 {order} 已取消。{ending}"
        if status in {"unknown", "executing", "submitted"}:
            return f"订单 {order} 的操作结果尚未确定，请查询原操作状态，不要重复发起。"
        if status == "failed":
            return f"订单 {order} 本次取消未成功，原因：{operation.get('error') or '后台拒绝'}。"
        return f"订单 {order} 操作状态：{status}。"

    def _tool(self, customer: str, conversation: str, request: str, name: str, arguments: Any) -> Any:
        expected = {"list_orders": set(), "get_order": {"order_id"}, "preview_cancel": {"order_id"},
                    "escalate_to_human": {"reason"}}
        if self.knowledge is not None:
            expected['search_policies']={'query'}
        if name not in expected or not isinstance(arguments, dict) or set(arguments) != expected[name]:
            raise ServiceError("tool_not_allowed_or_invalid_arguments")
        if self.policy.name == 'binding_state_v8' and name == 'escalate_to_human':
            from .request_workflow import current
            from .policy_evidence import handoff_requested
            _,text=current(self.store,customer,conversation,request)
            if not handoff_requested(text):raise ServiceError('handoff_not_requested',409)
        if self.policy.name == 'binding_state_v8' and self.knowledge is not None and name in {'preview_cancel','escalate_to_human'}:
            from .request_workflow import current
            from .policy_evidence import route
            payload,text=current(self.store,customer,conversation,request)
            if route(text,payload['binding']) is not None:
                raise ServiceError('target_needs_clarification' if name=='preview_cancel' else 'handoff_not_requested',409)
        if self.policy.name == 'binding_scoped_v7' and self.knowledge is not None and name in {'preview_cancel','escalate_to_human'}:
            from .policy_answer import route
            with self.store.transaction() as db:
                turn = db.execute('SELECT text FROM turns WHERE conversation_id=? AND request_id=?',(conversation,request)).fetchone()
            if turn is not None and route(turn['text']) is not None:
                raise ServiceError('target_needs_clarification' if name=='preview_cancel' else 'handoff_not_requested',409)
        if name == "list_orders":
            listing = self.backend.list_orders(customer)
            if self.policy.name in {'binding_catalog_v6', 'binding_scoped_v7'}:
                from .target_binding import ALIASES
                listing = {**listing, 'orders': [{**row, 'aliases': list(ALIASES.get(row.get('product'), ()))}
                                                 for row in listing['orders']]}
            if self.policy.name == 'binding_scoped_v7':
                from .intent_target import guidance
                users = [m['content'] for m in self.store.model_history(customer, conversation) if m['role']=='user']
                listing = {**listing, 'request_binding': guidance(users[-1], users[:-1], listing['orders'])}
            if self.policy.name == 'binding_state_v8':
                from .request_workflow import current
                from .order_facts import enrich
                payload,_=current(self.store,customer,conversation,request)
                listing={**listing,'orders':[enrich(r) for r in listing['orders']], 'request_binding':payload['binding']}
            return listing
        if name == "search_policies":
            return self.search_policies(arguments['query'])
        if name == "get_order":
            result=self.backend.get_order(customer, identifier(arguments["order_id"]))
            if self.policy.name == 'binding_state_v8' and isinstance(result.get('order'),dict):
                from .order_facts import enrich
                result={**result,'order':enrich(result['order'])}
            return result
        if name == "preview_cancel":
            evidence = None
            if self.policy.name == 'binding_state_v8':
                from .request_workflow import require_target as require_workflow_target
                evidence=require_workflow_target(self.store,self.backend,customer,conversation,request,arguments['order_id'])
            if self.policy.name in {"binding_guard_v4", "binding_grounded_v5", "binding_catalog_v6", "binding_scoped_v7"}:
                evidence = require_target(self.store, self.backend, customer, conversation, request, arguments["order_id"],
                                          grounded=self.policy.name in {"binding_grounded_v5", "binding_catalog_v6"},
                                          catalog_aliases=self.policy.name == 'binding_catalog_v6',
                                          scoped=self.policy.name == 'binding_scoped_v7')
            return self._preview(customer, conversation, arguments["order_id"], request_id=request,
                                 matched_version=evidence["order_version"] if evidence else None)
        reason = arguments["reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise ServiceError("invalid_handoff_reason")
        with self.store.transaction() as db:
            db.execute("UPDATE conversations SET status='handed_off' WHERE id=?", (conversation,))
            db.execute("UPDATE operations SET status='superseded',updated_at=? WHERE conversation_id=? "
                       "AND status='pending_confirmation'", (time.time(), conversation))
        return {"handed_off": True, "reason": reason, "note": "已进入本地待人工队列，尚未连接企业工单系统。"}

    @traced("turn")
    def message(self, customer: str, conversation: str, request: str, text: str) -> dict[str, Any]:
        request = identifier(request)
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ServiceError("invalid_message")
        with self.store.transaction() as db:
            self.store.require_owner(db, customer, conversation)
            existing = db.execute("SELECT * FROM turns WHERE conversation_id=? AND request_id=?",
                                  (conversation, request)).fetchone()
            if existing:
                if existing["text"] != text:
                    raise ServiceError("request_id_parameter_mismatch", 409)
                if existing["status"] == "running":
                    raise ServiceError("request_in_progress", 409)
                return json.loads(existing["response"])
            self._active(db, conversation)
            self._idle(db, conversation)
            db.execute("UPDATE operations SET status='superseded',updated_at=? "
                       "WHERE conversation_id=? AND status='pending_confirmation'", (time.time(), conversation))
            db.execute("INSERT INTO turns VALUES (?,?,?,'running',NULL,?)", (conversation, request, text, time.time()))
            db.execute("INSERT INTO messages(conversation_id,request_id,payload,created_at,turn_message) VALUES (?,?,?,?,1)",
                       (conversation, request, encode({"role": "user", "content": text}), time.time()))
        try:
            result = self._run_turn(customer, conversation, request)
            status = "completed"
        except Exception as exc:
            result = ({"error": "model_output_invalid", "message": "这次回复没有完成操作，请重新咨询或通过订单卡片操作。"}
                      if isinstance(exc, ModelOutputError) else
                      {"error": "model_protocol_mismatch", "message": "模型回复与当前操作流程不兼容，本轮未完成。请通过订单卡片操作。"}
                      if isinstance(exc, ModelProtocolError) else
                      {"error": "model_or_turn_failed", "message": "本轮未完成，请稍后重新发送，或通过订单卡片操作。"})
            if isinstance(exc, ModelServiceError):
                self.store.record(conversation, request, "model_failure", {}, {"code": exc.code, **exc.metadata})
            status = "failed"
            with self.store.transaction() as db:
                db.execute("UPDATE operations SET status='superseded',updated_at=? WHERE conversation_id=? "
                           "AND status='pending_confirmation'", (time.time(), conversation))
            self.store.append(conversation, request, {"role": "assistant", "content": result["message"]})
        with self.store.transaction() as db:
            db.execute("UPDATE turns SET status=?,response=? WHERE conversation_id=? AND request_id=?",
                       (status, encode(result), conversation, request))
        return result

    def _rebuild_preview_after_model_omission(self, customer, conversation, request, binding):
        """Only renew an earlier preview after a fresh, explicit user continuation.

        Never fill in an initial cancel intent, arbitrary target, or business write.
        Both reads/preview are visible as controller actions, not LLM tool calls.
        """
        if not binding.get('unique_target') or not binding.get('inherited_target'):
            return None
        target = binding['candidate_order_ids'][0]
        with self.store.transaction() as db:
            previous = db.execute("SELECT 1 FROM operations WHERE conversation_id=? AND order_id=? "
                                  "AND status IN ('superseded','expired') LIMIT 1", (conversation,target)).fetchone()
        if not previous:
            return None
        operation = None
        for name in ('get_order','preview_cancel'):
            call_id = 'controller_' + uuid4().hex
            arguments = {'order_id':target}
            call = {'role':'assistant','content':'','tool_calls':[{'id':call_id,'type':'function',
                'function':{'name':name,'arguments':encode(arguments)}}]}
            self.store.append(conversation,request,call)
            try:
                with self.telemetry.span('controller','preview_rebuild_'+name,authorization_for_write=False):
                    output = self._tool(customer,conversation,request,name,arguments)
            except Exception as error:
                code = error.code if isinstance(error,ServiceError) else 'controller_read_failed'
                output = {'error':code}
            self.store.record(conversation,request,name,arguments,{**output,'initiated_by':'controller_preview_rebuild'})
            self.store.append(conversation,request,{'role':'tool','name':name,'tool_call_id':call_id,'content':encode(output)})
            if output.get('error'):
                return None
            if name == 'get_order':
                order = output.get('order') or {}
                if order.get('id') != target or order.get('status') not in {'paid','pending'}:
                    return None
            else:
                operation = output
        return operation

    def _policy_evidence_response(self,conversation,request,evidence):
        from .policy_answer import extractive_answer
        content=extractive_answer(evidence)
        with self.telemetry.span('controller','policy_tool_loop_cutoff',authorization_for_write=False,
                                 hit_count=len(evidence['hits'])):
            self.store.record(conversation,request,'policy_loop_cutoff',{},
                {'max_model_tool_rounds':1,'evidence_source':'original_user_query','no_business_write':True})
        self.store.append(conversation,request,{'role':'assistant','content':content})
        return {'message':content,'mode':'model','operation':None,'answer_source':'controller_evidence',
            'sources':[{'citation':h['citation'],'source':h['source'],'version':h['version'],'demo_only':h['demo_only']}
                       for h in evidence['hits']]}

    def _run_turn(self, customer: str, conversation: str, request: str) -> dict[str, Any]:
        if self.model is None:
            content = "智能客服暂未连接。您可以先查看订单，并通过订单卡片申请取消。"
            self.store.append(conversation, request, {"role": "assistant", "content": content})
            return {"message": content, "mode": "manual"}
        messages = self.store.model_history(customer, conversation)
        schemas, system = self.policy.tools, self.policy.prompt
        target_error = None
        knowledge_result = None
        request_binding = None
        preview_replan_used = False
        protocol_replan_used = False
        target_is_previewable = False
        policy_topics = None
        policy_answer_evidence = None
        if self.policy.name in {'binding_scoped_v7','binding_state_v8'}:
            if self.policy.name == 'binding_state_v8':
                from .request_workflow import prepare
                from .policy_evidence import route,select as scoped_result
            else:
                from .intent_target import guidance
                from .policy_answer import route, scoped_result
            users = [m['content'] for m in messages if m['role']=='user']
            with self.telemetry.span('controller', 'read_preview_intent', authorization_for_write=False):
                listing = self.backend.list_orders(customer)
                orders = listing.get('orders') if isinstance(listing, dict) else None
                if not isinstance(orders, list) or any(row.get('customer_id') != customer for row in orders):
                    raise ServiceError('invalid_owned_order_evidence', 502)
                if self.policy.name == 'binding_state_v8':
                    from .request_workflow import validated_orders
                    orders=validated_orders(listing,customer)
                request_binding = (prepare(self.store,customer,conversation,request,users[-1],orders)
                                   if self.policy.name == 'binding_state_v8' else guidance(users[-1], users[:-1], orders))
                target_is_previewable = request_binding['unique_target'] and any(
                    row['id'] in request_binding['candidate_order_ids'] and row['status'] in {'pending','paid'} for row in orders)
                self.store.record(conversation, request, 'controller_catalog', {}, listing)
            policy_topics = ((route(users[-1],request_binding) if self.policy.name == 'binding_state_v8' else route(users[-1]))
                             if self.knowledge is not None else None)
            if policy_topics is not None:
                # An explicit read-only controller action, recorded as such. It is
                # not an LLM-selected tool call or a generated supervision target.
                call_id = 'controller_' + uuid4().hex
                query = {'query': users[-1]}
                lookup_call = {'role':'assistant', 'content':'', 'tool_calls':[{'id':call_id, 'type':'function',
                    'function':{'name':'search_policies', 'arguments':encode(query)}}]}
                with self.telemetry.span('controller', 'required_policy_lookup', authorization_for_write=False) as evidence_span:
                    knowledge_result = scoped_result(self.search_policies(users[-1]), policy_topics)
                    if self.policy.name == 'binding_state_v8':
                        policy_answer_evidence=knowledge_result
                        diagnostic=knowledge_result['evidence_diagnostics']
                        evidence_span.set(topic_count=len(diagnostic['topic_ids']),raw_hit_count=len(diagnostic['raw_citations']),
                            selected_hit_count=len(diagnostic['selected_citations']),discarded_hit_count=len(diagnostic['discarded_citations']),
                            no_evidence_reason=diagnostic['no_evidence_reason'])
                self.store.record(conversation, request, 'search_policies', query,
                                  {**knowledge_result, 'initiated_by':'controller'})
                lookup_result = {'role':'tool', 'name':'search_policies', 'tool_call_id':call_id,
                                 'content':encode(knowledge_result)}
                for row in (lookup_call, lookup_result):
                    self.store.append(conversation, request, row); messages.append(row)
        for step in range(self.max_steps):
            with self.telemetry.span("context", "build_context") as span:
                state = self.store.context_state(customer, conversation)
                if request_binding is not None: state['request_binding'] = request_binding
                built = self.context_builder.build(system, messages, schemas,
                    state)
                span.set(**built.metadata)
            with self.telemetry.span("model", "complete", attempt=step + 1,
                    model_id=getattr(self.model, "model", type(self.model).__name__),
                    prompt_sha256=sha256(system.encode()).hexdigest(),
                    tools_sha256=sha256(encode(schemas).encode()).hexdigest()) as span:
                assistant = self.model.complete(built.messages, schemas)
                metadata = assistant.get("_provider_metadata") or {}
                if isinstance(metadata, dict):
                    span.set(**{key: metadata[key] for key in
                        ("response_model", "finish_reason", "input_tokens", "output_tokens") if key in metadata})
            calls = assistant.get("tool_calls") or []
            if not isinstance(calls, list) or len(calls) > 8:
                raise ValueError("invalid_tool_calls")
            message = {"role": "assistant", "content": str(assistant.get("content") or "")}
            if calls:
                clean_calls = []
                for call in calls:
                    function = call.get("function") or {}
                    clean_calls.append({"id": "call_" + uuid4().hex, "type": "function", "function": {
                        "name": str(function.get("name") or ""),
                        "arguments": function.get("arguments", "{}") if isinstance(function.get("arguments", "{}"), str)
                        else encode(function["arguments"])}})
                message["tool_calls"] = clean_calls
                self.store.append(conversation, request, message)
                messages.append(message)
                handed_off = False
                for call in clean_calls:
                    name = call["function"]["name"]
                    arguments: Any = call["function"]["arguments"]
                    known = name in {schema['function']['name'] for schema in schemas}
                    with self.telemetry.span("tool", name if known else "__invalid_tool__", tool_call_id=call["id"]) as span:
                        try:
                            arguments = json.loads(arguments)
                            if handed_off:
                                raise ServiceError("conversation_handed_off", 409)
                            output = self._tool(customer, conversation, request, name, arguments)
                            if name == 'search_policies':
                                if policy_topics is not None and self.policy.name in {'binding_scoped_v7','binding_state_v8'}:
                                    if self.policy.name == 'binding_state_v8':
                                        from .policy_evidence import select as scoped_result
                                    else:
                                        from .policy_answer import scoped_result
                                    output = scoped_result(output, policy_topics)
                                knowledge_result = output
                            handed_off = name == "escalate_to_human"
                            if isinstance(output, dict) and output.get("id"):
                                span.set(operation_id=output["id"])
                        except Exception as exc:
                            code = exc.code if isinstance(exc, ServiceError) else "tool_failed"
                            span.fail(code)
                            if code in TARGET_MESSAGES:
                                target_error = code
                            span.set(exception_type=type(exc).__name__)
                            output = {"error": code}
                    self.store.record(conversation, request, name, arguments, output)
                    tool = {"role": "tool", "tool_call_id": call["id"], "name": name, "content": encode(output)}
                    messages.append(tool)
                    self.store.append(conversation, request, tool)
                if handed_off:
                    content = "已进入本地待人工队列，自动操作已停止；尚未接入企业工单系统。"
                    self.store.append(conversation, request, {"role": "assistant", "content": content})
                    return {"message": content, "handed_off": True}
                if self.policy.name == 'binding_state_v8' and policy_topics is not None and policy_answer_evidence is not None:
                    # The original user lookup is authoritative. Model-generated
                    # searches cannot overwrite it or repeat indefinitely.
                    return self._policy_evidence_response(conversation,request,policy_answer_evidence)
                continue
            with self.store.transaction() as db:
                db.execute("UPDATE operations SET status='expired',updated_at=? WHERE conversation_id=? "
                           "AND status='pending_confirmation' AND expires_at<=?", (time.time(), conversation, time.time()))
                pending = db.execute("SELECT * FROM operations WHERE conversation_id=? "
                                     "AND status='pending_confirmation' ORDER BY created_at DESC LIMIT 1", (conversation,)).fetchone()
            # Any write request is shown as a server-generated preview, not model-authorized success.
            if (self.policy.name in {'binding_scoped_v7','binding_state_v8'} and not pending and target_is_previewable
                    and policy_topics is None and not preview_replan_used and step + 1 < self.max_steps):
                # One bounded planning correction, not an automatic preview/write.
                # The model must still call get_order/preview_cancel; that tool
                # rechecks live target evidence and the button remains mandatory.
                with self.telemetry.span('controller','missing_preview_replan',authorization_for_write=False):
                    self.store.record(conversation,request,'controller_replan',request_binding,
                                      {'reason':'unique_cancel_intent_without_current_preview','attempt':1})
                preview_replan_used = True
                system = self.policy.prompt + '\n本轮规划检查：用户已明确要求继续取消，当前没有有效预览。不要再次询问聊天确认，也不要只说旧卡失效。现在按下列只读意图提示，重新 get_order 查询该唯一目标，若仍可取消则调用 preview_cancel；不能执行取消。\n' + encode(request_binding)
                continue
            if (self.policy.name in {'binding_scoped_v7','binding_state_v8'} and not pending and target_is_previewable
                    and preview_replan_used and policy_topics is None):
                rebuilt = self._rebuild_preview_after_model_omission(customer,conversation,request,request_binding)
                if rebuilt:
                    content = self._preview_message(rebuilt)
                    self.store.append(conversation,request,{'role':'assistant','content':content})
                    return {'message':content,'mode':'model','operation':rebuilt,'controller_preview_rebuild':True}
            if pending:
                operation = self._public(pending)
                message["content"] = self._preview_message(operation)
            else:
                try:
                    message["content"] = terminal_content(message["content"])
                    facts=None
                    if self.policy.name == 'binding_state_v8':
                        from .order_facts import reply
                        facts=reply(users[-1],request_binding,orders)
                    if self.policy.name in {'binding_scoped_v7','binding_state_v8'} and policy_topics is not None and knowledge_result is not None:
                        from .policy_answer import extractive_answer
                        message['content'] = extractive_answer(knowledge_result)
                    elif facts is not None:
                        with self.telemetry.span('controller','render_order_facts',authorization_for_write=False):
                            message['content']=facts
                    elif target_error:
                        message["content"] = TARGET_MESSAGES[target_error]
                    elif preview_claim(message["content"]):
                        message["content"] = "当前没有有效的待确认取消申请，请重新指定商品或订单，或通过订单卡片申请取消。"
                    elif knowledge_result is not None:
                        hits=knowledge_result['hits']
                        if not hits:
                            message['content']='没有检索到有效说明，暂时无法给出有依据的政策回答，请联系人工核实。'
                        elif (not any('['+hit['citation']+']' in message['content'] for hit in hits)
                              or not set(re.findall(r'\[([a-z0-9_-]+@[a-z0-9_.-]+)\]',message['content']))
                                 <= {hit['citation'] for hit in hits}):
                            label='（仅供演示）' if any(hit['demo_only'] for hit in hits) else ''
                            message['content']='检索到以下说明'+label+'：\n'+'\n'.join(
                                '['+hit['citation']+'] '+hit['text'] for hit in hits)
                        elif any(hit['demo_only'] for hit in hits) and '演示' not in message['content']:
                            message['content']='以下为演示说明，不能当作实际商家政策：\n'+message['content']
                except ModelProtocolError:
                    self.store.record(conversation, request, "terminal_validation", {},
                                      {"error": "model_protocol_mismatch",'rejected_output_sha256':sha256(message['content'].encode()).hexdigest()})
                    if self.policy.name == 'binding_state_v8' and not protocol_replan_used and step+1<self.max_steps:
                        with self.telemetry.span('controller','terminal_protocol_replan',authorization_for_write=False):
                            protocol_replan_used=True
                        # Reject the JSON as an action. It is never converted into
                        # a tool call or a write; ask the model once to reformat.
                        messages.append(message)
                        system=self.policy.prompt+'\n上一条纯文本JSON未通过终态协议校验，未作为工具调用执行。若要调用工具，必须使用tools定义的tool_calls；若只是解释或澄清，请直接用普通中文文本，不输出工具JSON或操作成功字段。最多纠正这一次。'
                        continue
                    raise
            self.store.append(conversation, request, message)
            return {"message": message["content"], "mode": "model", "operation": self._public(pending) if pending else None,
                    **({'sources':[{'citation':hit['citation'],'source':hit['source'],'version':hit['version'],
                                    'demo_only':hit['demo_only']} for hit in knowledge_result['hits']]}
                       if knowledge_result is not None else {})}
        raise ServiceError("max_model_steps", 502)
