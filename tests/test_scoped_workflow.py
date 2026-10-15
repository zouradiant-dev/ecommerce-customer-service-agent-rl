from pathlib import Path
import json
import tempfile
import unittest

from ecommerce_customer_service.serving.intent_target import resolve, guidance
from ecommerce_customer_service.serving.policy_answer import route
from ecommerce_customer_service.serving.storage import Store, ServiceError, encode
from ecommerce_customer_service.serving.backend import MockBusinessBackend
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.context_builder import ContextBuilder

ORDERS = [dict(id='A', product='雕刻机', version=1, paid_cents=17990),
          dict(id='B', product='雕刻刀套装', version=2, paid_cents=5690)]


class IntentTests(unittest.TestCase):
    def test_indirect_cancel_requires_purchasing_context_for_close_verb(self):
        for text in ('雕刻机买重复了，这笔订单给我关掉。', '雕刻机不需要了。', '雕刻机这项购买终止交易。'):
            with self.subTest(text=text): self.assertEqual('A', resolve(text, [], ORDERS)['order_id'])
        with self.assertRaises(ServiceError): resolve('把雕刻机关掉。', [], ORDERS)

    def test_denial_and_attribution_cannot_create_preview(self):
        for text in ('说取消雕刻机的是别人，我并没有提出要求。', '我没有授权取消雕刻机。',
                     '不是我想取消雕刻机。', '同事要求取消雕刻机，我还没决定。',
                     '我不是真的想取消雕刻机。', '雕刻机不能取消，不准操作。'):
            with self.subTest(text=text), self.assertRaises(ServiceError): resolve(text, [], ORDERS)

    def test_keep_and_cancel_are_opposite_roles(self):
        text = '雕刻刀套装留下，雕刻机不买了。'
        self.assertEqual('A', resolve(text, [], ORDERS)['order_id'])
        binding = guidance(text, [], ORDERS)
        self.assertEqual(['B'], binding['retained_order_ids'])
        self.assertFalse(binding['authorization_for_write'])

    def test_confirmation_does_not_resurrect_withdrawn_or_queried_intent(self):
        for latest in ('我确定要取消这笔购买。', '我在聊天里同意取消，请先给我看确认页面。', '现在这单不需要了。'):
            with self.subTest(latest=latest): self.assertEqual('A', resolve(latest, ['雕刻机这笔撤单。'], ORDERS)['order_id'])
            for intervening in ('先保留订单。', '只查询雕刻机，不要操作。', '聊点别的，天气怎样？'):
                with self.subTest(latest=latest, intervening=intervening), self.assertRaises(ServiceError):
                    resolve(latest, ['雕刻机这笔撤单。', intervening], ORDERS)

    def test_new_unknown_target_never_inherits_old_id(self):
        with self.assertRaises(ServiceError): resolve('取消榨油机这笔。', ['雕刻机这笔撤单。'], ORDERS)
        with self.assertRaises(ServiceError): resolve('取消雕刻机和雕刻刀套装。', [], ORDERS)

    def test_duplicate_names_require_real_user_disambiguation(self):
        orders = [ORDERS[0], dict(ORDERS[0], id='C', paid_cents=27990)]
        with self.assertRaisesRegex(ServiceError, 'target_ambiguous'): resolve('雕刻机的订单关掉。', [], orders)
        self.assertEqual('C', resolve('取消金额279.90元的那张订单。', ['雕刻机的订单关掉。'], orders)['order_id'])
        self.assertEqual('C', resolve('取消金额 279.90 元的那一笔。', ['雕刻机的订单关掉。'], orders)['order_id'])


class Sequence:
    model = 'explicit_fixture_not_llm'
    def __init__(self, *responses): self.responses=list(responses); self.requests=[]
    def complete(self, messages, tools):
        self.requests.append(messages)
        return self.responses.pop(0)


def preview(order):
    return {'tool_calls': [{'function': {'name': 'preview_cancel', 'arguments': encode({'order_id': order})}}]}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.root=Path(self.temporary.name)
        self.backend=MockBusinessBackend(self.root/'b.sqlite'); self.store=Store(self.root/'s.sqlite')
        self.index=KnowledgeIndex(Path(__file__).resolve().parents[1]/'data/demo-policies.jsonl')

    def tearDown(self): self.temporary.cleanup()

    def service(self, model):
        # Fixtures use the conservative UTF-8 byte estimator, not a tokenizer.
        # Real-model evaluation separately enforces the actual 8192-token budget.
        service=CustomerService(self.store,self.backend,model,policy_profile='binding_scoped_v7',knowledge=self.index,
                                context_builder=ContextBuilder(max_input_tokens=32768))
        return service,service.create_conversation('customer_alice')['conversation_id']

    def test_adversarial_model_cannot_preview_denied_or_retained_order(self):
        for text, order in (('说取消耳机的是别人，我并未提出要求。','order_A'),
                            ('耳机留下，取消键盘。','order_A')):
            model=Sequence(preview(order),{'content':'已生成预览。'},{'content':'仍无法处理。'})
            service, conv=self.service(model)
            result=service.message('customer_alice',conv,'r',text)
            self.assertIsNone(result.get('operation'))
            self.assertEqual([],service.history('customer_alice',conv)['operations'])
        with self.backend.transaction() as db: self.assertEqual(0, db.execute('SELECT COUNT(*) FROM cancellations').fetchone()[0])

    def test_model_context_has_target_roles_and_button_remains_required(self):
        model=Sequence(preview('order_B'),{'content':'已经取消。'})
        service,conv=self.service(model)
        result=service.message('customer_alice',conv,'r','耳机保留，取消键盘。')
        state=json.loads(model.requests[0][1]['content'].split('\n',1)[1])['business_state']['request_binding']
        self.assertEqual(['order_B'],state['candidate_order_ids']); self.assertEqual(['order_A'],state['retained_order_ids'])
        self.assertEqual('pending_confirmation',result['operation']['status'])
        self.assertEqual('paid',self.backend.get_order('customer_alice','order_B')['order']['status'])

    def test_missing_preview_has_one_planning_correction_and_still_no_write(self):
        model=Sequence({'content':'确认继续取消？'},preview('order_A'),{'content':'已取消。'})
        service,conv=self.service(model)
        result=service.message('customer_alice',conv,'r','耳机这一单请取消。')
        self.assertEqual('pending_confirmation',result['operation']['status'])
        self.assertEqual('paid',self.backend.get_order('customer_alice','order_A')['order']['status'])
        self.assertEqual(1,sum(s['name']=='missing_preview_replan' for s in service.telemetry.rows()))
        model=Sequence({'content':'确认吗？'},{'content':'请再确认。'})
        service,conv=self.service(model)
        result=service.message('customer_alice',conv,'r','耳机请取消。')
        self.assertIsNone(result['operation']);self.assertEqual(2,len(model.requests))

    def test_controller_only_rebuilds_prior_preview_after_explicit_continuation(self):
        service,conv=self.service(Sequence(preview('order_A'),{'content':'请确认'}))
        first=service.message('customer_alice',conv,'initial','耳机这笔请取消。')['operation']
        service.model=Sequence({'content':'确认继续吗？'},{'content':'请重新申请。'})
        result=service.message('customer_alice',conv,'followup','我确定要取消这笔订单。')
        self.assertTrue(result['controller_preview_rebuild'])
        self.assertNotEqual(first['id'],result['operation']['id'])
        self.assertEqual('paid',self.backend.get_order('customer_alice','order_A')['order']['status'])
        with self.assertRaisesRegex(ServiceError,'confirmation_invalidated'):service.confirm('customer_alice',first['id'])
        service.model=Sequence({'content':'已停止'})
        service.message('customer_alice',conv,'withdraw','先保留订单，不取消了。')
        service.model=Sequence({'content':'请核对目标'})
        blocked=service.message('customer_alice',conv,'stale_followup','确认取消。')
        self.assertIsNone(blocked['operation'])

    def test_required_retrieval_and_source_extract_override_wrong_model_answer(self):
        model=Sequence({'content':'点击确认取消只会预览，不会取消。[cancel-preview@1]'})
        service,conv=self.service(model)
        result=service.message('customer_alice',conv,'r','演示预览按钮和确认按钮有什么区别？')
        self.assertIn('用户须在界面核对订单和金额后点击确认',result['message'])
        self.assertNotIn('点击确认取消只会预览',result['message']); self.assertTrue(result['sources'])
        history=service.history('customer_alice',conv)['messages']
        self.assertTrue(any(m['role']=='tool' and m.get('name')=='search_policies' for m in history))
        self.assertTrue(any(span['kind']=='controller' for span in service.telemetry.rows()))

    def test_policy_route_cannot_be_overridden_by_model_preview_or_handoff(self):
        for call in (preview('order_A'),{'tool_calls':[{'function':{'name':'escalate_to_human','arguments':encode({'reason':'误判'})}}]}):
            service,conv=self.service(Sequence(call,{'content':'政策说明。'}))
            result=service.message('customer_alice',conv,'faq','无线耳机的取消政策是什么？')
            self.assertIsNone(result['operation']);self.assertEqual('active',service.history('customer_alice',conv)['status'])
            self.assertEqual([],service.history('customer_alice',conv)['operations'])
            self.assertTrue(result['sources'])

    def test_unknown_policy_abstains_and_handoff_request_is_not_faq(self):
        self.assertIsNone(route('可以由人工继续处理吗？'))
        self.assertEqual(['handoff'],route('我只想了解演示人工接管队列，不需要转接。'))
        self.assertEqual(['cancel-preview'],route('已发货订单的取消政策是什么？'))
        model=Sequence({'content':'安装费200元。'})
        service,conv=self.service(model)
        result=service.message('customer_alice',conv,'r','安装收费有没有明确说明？')
        self.assertEqual([],result['sources']); self.assertIn('没有检索到有效说明',result['message'])


if __name__=='__main__': unittest.main()
