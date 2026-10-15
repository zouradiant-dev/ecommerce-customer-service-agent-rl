import json
import unittest

from ecommerce_customer_service.serving.request_workflow import analyze,catalog_mentions,validated_orders
from ecommerce_customer_service.serving.order_facts import yuan
from ecommerce_customer_service.serving.policy_evidence import route,select,handoff_requested
from ecommerce_customer_service.serving.storage import ServiceError
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.context_builder import ContextBuilder
from tests import test_scoped_workflow as fixtures
Sequence,preview=fixtures.Sequence,fixtures.preview


ROWS=[dict(id='order_X',product='硅胶揉面垫',customer_id='customer_alice',version=1,paid_cents=51302,status='paid'),
      dict(id='order_Y',product='擀面杖支架',customer_id='customer_alice',version=1,paid_cents=509,status='paid')]


class FrameTests(unittest.TestCase):
    def test_clause_roles_resolve_abbreviation_without_first_row_preference(self):
        for rows in (ROWS,list(reversed(ROWS))):
            result=analyze('揉面垫保留，取消擀面杖支架。',None,rows)
            self.assertEqual(['order_Y'],result['candidate_order_ids'])
            self.assertEqual(['order_X'],result['retained_order_ids'])
            self.assertEqual(['keep','cancel'],[c['role'] for c in result['clauses']])

    def test_alias_boundaries_and_collisions_fail_closed(self):
        rows=[dict(ROWS[0],product='无线耳机'),dict(ROWS[1],product='有线耳机')]
        self.assertEqual({'order_Y'},{m['order_id'] for m in catalog_mentions('取消有线耳机。',rows)})
        rows=[dict(ROWS[0],product='红色防水背包'),dict(ROWS[1],product='蓝色防水背包')]
        self.assertFalse(analyze('防水背包取消。',None,rows)['unique_target'])
        self.assertEqual([],list(catalog_mentions('取消无线耳机收纳盒。',[dict(ROWS[0],product='无线耳机')])))

    def test_physical_close_is_not_purchase_cancellation(self):
        self.assertFalse(analyze('把硅胶揉面垫关闭。',None,ROWS)['unique_target'])
        self.assertTrue(analyze('硅胶揉面垫买错了，这单关了吧。',None,ROWS)['unique_target'])

    def test_request_state_is_distinct_from_confirmation_card(self):
        frame=analyze('硅胶揉面垫请退订。',None,ROWS)
        next_frame=analyze('我确定退订该购买，但还没有按页面按钮。',frame,ROWS)
        self.assertTrue(next_frame['inherited_target']);self.assertTrue(next_frame['unique_target'])
        next_frame=analyze('刚才那笔我现在确定取消，请给我可点击的新确认项。',frame,ROWS)
        self.assertEqual(['order_X'],next_frame['candidate_order_ids'])

    def test_unknown_new_product_or_withdrawal_cannot_borrow_old_request(self):
        frame=analyze('硅胶揉面垫请取消。',None,ROWS)
        self.assertFalse(analyze('改成取消光照度仪这笔。',frame,ROWS)['unique_target'])
        self.assertFalse(analyze('硅胶揉面垫我需要，取消别的未匹配商品。',None,ROWS)['unique_target'])
        self.assertFalse(analyze('硅胶揉面垫买了，取消不存在的商品那笔。',None,ROWS)['unique_target'])
        for reset in ('先留着硅胶揉面垫。','只查硅胶揉面垫的价格。','取消硅胶揉面垫是同事的要求，我没有授权。'):
            state=analyze(reset,frame,ROWS)
            self.assertFalse(analyze('确认取消。',state,ROWS)['unique_target'])

    def test_same_name_amount_selection_and_other_retained(self):
        rows=[ROWS[0],dict(ROWS[1],product=ROWS[0]['product'])]
        state=analyze('硅胶揉面垫买了两笔，我只要取消一笔，先让我看清。',None,rows)
        self.assertEqual('awaiting_choice',state['dialogue_state'])
        choice=analyze('请办理金额513.02元那一笔的取消，另一张留下。',state,rows)
        self.assertEqual(['order_X'],choice['candidate_order_ids'])

    def test_negative_scope_and_hypothetical_never_grant_preview(self):
        for text in ('请勿取消硅胶揉面垫。','我反对取消硅胶揉面垫，保持原样。',
                     '我没叫你取消硅胶揉面垫，请保留这件商品。','我从来没让你退订硅胶揉面垫，这笔购买继续。',
                     '硅胶揉面垫的取消申请别建，我只是问订单现在的情况。','查询硅胶揉面垫有没有被取消，不要替我取消。',
                     '我打消取消硅胶揉面垫的念头，这单保留。','假如我取消硅胶揉面垫，会怎样？'):
            self.assertFalse(analyze(text,None,ROWS)['unique_target'],text)

    def test_integer_money_rendering(self):
        for value,expected in ((0,'0.00'),(1,'0.01'),(509,'5.09'),(51302,'513.02')):
            self.assertEqual(expected,yuan(value))
        with self.assertRaises(ValueError):yuan(1.5)

    def test_rejects_malformed_or_duplicate_live_evidence(self):
        for rows in ([ROWS[0],ROWS[0]],[dict(ROWS[0],version=True)],
                     [dict(ROWS[0],customer_id='other')],[dict(ROWS[0],paid_cents='51302')]):
            with self.assertRaisesRegex(ServiceError,'invalid_owned_order_evidence'):
                validated_orders({'orders':rows},'customer_alice')


class PersistedServiceTests(unittest.TestCase):
    setUp=fixtures.ServiceTests.setUp
    tearDown=fixtures.ServiceTests.tearDown

    def service(self,model):
        service=CustomerService(self.store,self.backend,model,policy_profile='binding_state_v8',knowledge=self.index,
                                context_builder=ContextBuilder(max_input_tokens=32768))
        return service,service.create_conversation('customer_alice')['conversation_id']

    def test_restart_continuation_creates_new_card_and_stale_button_fails(self):
        service,conv=self.service(Sequence(preview('order_A'),{'content':'请确认'}))
        old=service.message('customer_alice',conv,'a','耳机订单退订。')['operation']
        service=CustomerService(self.store,self.backend,Sequence(preview('order_A'),{'content':'文字不是写入授权'}),
            policy_profile='binding_state_v8',knowledge=self.index,context_builder=ContextBuilder(max_input_tokens=32768))
        service.recover()
        result=service.message('customer_alice',conv,'b','我确定要退订该购买，但还没有按页面按钮。')
        self.assertNotEqual(old['id'],result['operation']['id'])
        with self.assertRaisesRegex(ServiceError,'confirmation_invalidated'):service.confirm('customer_alice',old['id'])
        self.assertEqual('paid',self.backend.get_order('customer_alice','order_A')['order']['status'])

    def test_server_formats_ambiguous_order_facts_even_if_model_miscalculates(self):
        with self.backend.transaction() as db:
            db.execute("UPDATE orders SET product='硅胶揉面垫',paid_cents=51302 WHERE id='order_A'")
            db.execute("UPDATE orders SET product='硅胶揉面垫',paid_cents=509 WHERE id='order_B'")
        service,conv=self.service(Sequence({'content':'金额分别为51.30和0.50元。'}))
        result=service.message('customer_alice',conv,'a','硅胶揉面垫要取消一笔，请列出两笔让我挑选。')
        self.assertIn('513.02 元',result['message']);self.assertIn('5.09 元',result['message'])
        self.assertNotIn('51.30',result['message']);self.assertIsNone(result['operation'])
        service.model=Sequence(preview('order_B'),{'content':'已办妥'})
        result=service.message('customer_alice',conv,'b','选择5.09元那张办理取消，另一张保留。')
        self.assertEqual('order_B',result['operation']['order_id'])

    def test_live_catalog_change_and_other_owner_cannot_override_binding(self):
        class ChangeDuringGeneration:
            model='adversarial_fixture'
            def complete(inner,messages,tools):
                with self.backend.transaction() as db:db.execute("UPDATE orders SET version=version+1 WHERE id='order_A'")
                return preview('order_A')
        service,conv=self.service(ChangeDuringGeneration())
        service.max_steps=1
        result=service.message('customer_alice',conv,'a','耳机请取消。')
        self.assertIsNone(result.get('operation'))
        with self.store.transaction() as db:self.assertEqual(0,db.execute('SELECT COUNT(*) FROM operations').fetchone()[0])
        service.model=Sequence(preview('order_B'),{'content':'我已选择另一件'})
        service.max_steps=3
        result=service.message('customer_alice',conv,'b','耳机保留，取消order_D。')
        self.assertIsNone(result.get('operation'))

    def test_policy_word_order_has_evidence_and_retrieval_diagnostics(self):
        service,conv=self.service(Sequence(preview('order_A'),{'content':'我替您取消了。'}))
        result=service.message('customer_alice',conv,'a','我只想了解演示规则：发货后还能取消吗？先不处理我的订单。')
        self.assertEqual(['cancel-preview@1'],[s['citation'] for s in result['sources']])
        self.assertIsNone(result['operation']);self.assertIn('已发货订单不能直接取消',result['message'])
        lookup=next(s for s in service.telemetry.rows() if s['name']=='required_policy_lookup')
        self.assertGreaterEqual(lookup['attributes']['raw_hit_count'],1)
        self.assertEqual(1,lookup['attributes']['selected_hit_count'])

    def test_unknown_policy_never_uses_loosely_related_cancellation_doc(self):
        service,conv=self.service(Sequence({'content':'保修三年。'}))
        result=service.message('customer_alice',conv,'a','取消之后延保的收费政策是什么？')
        self.assertEqual([],result['sources']);self.assertIn('没有检索到有效说明',result['message'])

    def test_handoff_needs_explicit_user_request_not_model_preference(self):
        handoff={'tool_calls':[{'function':{'name':'escalate_to_human','arguments':'{"reason":"我想转接"}'}}]}
        service,conv=self.service(Sequence(handoff,{'content':'您没有要求取消，本轮没有操作。'}))
        result=service.message('customer_alice',conv,'a','别人说过取消耳机，这不是我的要求。')
        self.assertEqual('active',service.history('customer_alice',conv)['status'])
        self.assertIsNone(result['operation'])
        self.assertFalse(handoff_requested('不需要转人工，请解释接管队列。'))
        self.assertTrue(handoff_requested('请转人工处理。'))

    def test_wrong_model_search_does_not_replace_original_policy_evidence(self):
        search={'tool_calls':[{'function':{'name':'search_policies','arguments':'{"query":"订单退款规则"}'}}]}
        model=Sequence(search)
        service,conv=self.service(model)
        result=service.message('customer_alice',conv,'a','人工接管队列的演示说明是什么？请解释政策，不要转接。')
        self.assertEqual(1,len(model.requests))
        self.assertEqual(['handoff@1'],[s['citation'] for s in result['sources']])
        self.assertEqual('controller_evidence',result['answer_source'])

    def test_invalid_terminal_tool_json_is_rejected_before_one_bounded_replan(self):
        bad={'content':'{"name":"preview_cancel","arguments":{"order_id":"order_A"}}'}
        service,conv=self.service(Sequence(bad,{'content':'您还没有决定，本轮没有取消。'}))
        result=service.message('customer_alice',conv,'a','耳机先不要动，我还没有决定。')
        self.assertIsNone(result['operation'])
        self.assertEqual(1,sum(s['name']=='terminal_protocol_replan' for s in service.telemetry.rows()))
        self.assertEqual([],service.history('customer_alice',conv)['operations'])
        service.model=Sequence(bad,bad)
        result=service.message('customer_alice',conv,'b','耳机先留着，我还没决定。')
        self.assertEqual('model_protocol_mismatch',result['error'])

    def test_request_frame_never_leaks_between_conversations(self):
        service,first=self.service(Sequence(preview('order_A'),{'content':'请核对确认卡。'}))
        service.message('customer_alice',first,'a','耳机请申请取消。')
        second=service.create_conversation('customer_alice')['conversation_id']
        service.model=Sequence(preview('order_A'),{'content':'需要明确是哪笔订单。'})
        result=service.message('customer_alice',second,'b','确认取消。')
        self.assertIsNone(result['operation'])
        self.assertEqual([],service.history('customer_alice',second)['operations'])

    def test_failed_turn_cannot_supply_next_confirmation_target(self):
        bad={'content':'{"name":"preview_cancel","arguments":{"order_id":"order_A"}}'}
        service,conv=self.service(Sequence(bad,bad,bad))
        failed=service.message('customer_alice',conv,'a','耳机这单请取消。')
        self.assertEqual('model_protocol_mismatch',failed['error'])
        service.model=Sequence(preview('order_A'),{'content':'请重新明确目标。'})
        result=service.message('customer_alice',conv,'b','确认取消。')
        self.assertIsNone(result['operation'])


if __name__=='__main__':unittest.main()
