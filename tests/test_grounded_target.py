import tempfile
from pathlib import Path
import unittest

from ecommerce_customer_service.serving.grounded_target import resolve
from ecommerce_customer_service.serving.backend import MockBusinessBackend
from ecommerce_customer_service.serving.storage import Store, ServiceError, encode
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.policy import get_policy


ORDERS = [dict(id='A', product='无线耳机', version=1, paid_cents=19900),
          dict(id='B', product='键盘', version=2, paid_cents=29900)]


class ResolverTests(unittest.TestCase):
    def test_paraphrases_are_grounded_without_full_sentence_templates(self):
        for text in ('想撤销无线耳机这笔订单，请帮忙处理。', '计划取消无线耳机，麻烦查查这笔。',
                     '请把无线耳机那张订单撤掉，谢谢', '无线耳机不用了', '我想退订无线耳机订单',
                     '麻烦作废键盘的订单'):
            self.assertEqual('B' if '键盘' in text else 'A', resolve(text, [], ORDERS)['order_id'])

    def test_alias_does_not_match_conflicting_product_prefix(self):
        with self.assertRaises(ServiceError): resolve('取消有线耳机订单', [], ORDERS)
        self.assertEqual('A', resolve('帮我取消耳机订单', [], ORDERS)['order_id'])

    def test_same_name_orders_never_select_by_price_without_user_evidence(self):
        orders = ORDERS + [dict(ORDERS[0], id='C', paid_cents=9900)]
        with self.assertRaisesRegex(ServiceError, 'target_ambiguous'):
            resolve('想撤销无线耳机这笔订单，请帮忙处理。', [], orders)
        self.assertEqual('C', resolve('就99元的那一笔。', ['想撤销无线耳机这笔订单，请帮忙处理。'], orders)['order_id'])
        self.assertEqual('A', resolve('取消金额199.00元的那一笔', ['无线耳机不用了'], orders)['order_id'])

    def test_negation_quote_query_hypothesis_and_contradiction_fail_closed(self):
        for text in ('别取消耳机', '只是查询耳机订单，不要操作', '照着例子说：取消耳机',
                     '如果取消无线耳机会怎样', '怎么取消无线耳机', '无线耳机能取消吗',
                     '我没有要求取消耳机', '我不是真的想取消无线耳机', "'取消无线耳机'", '取消无线耳机，耳机保留', '取消耳机，等等不要取消',
                     '有人说取消键盘', '查询无线耳机取消进度', '取消耳机和键盘'):
            with self.subTest(text=text), self.assertRaises(ServiceError): resolve(text, [], ORDERS)

    def test_switch_excludes_retained_order_and_withdrawal_clears_context(self):
        self.assertEqual('B', resolve('别取消耳机，改成取消键盘', ['取消耳机'], ORDERS)['order_id'])
        self.assertEqual('B', resolve('改为取消键盘，耳机保留', ['取消耳机'], ORDERS)['order_id'])
        for history in (['取消耳机', '先保留订单，不取消了'], ['取消耳机', '先查询键盘订单']):
            with self.assertRaises(ServiceError): resolve('确认取消', history, ORDERS)

    def test_confirmation_is_only_preview_intent_and_prompt_control_is_unchanged(self):
        result = resolve('确认取消。', ['无线耳机不用了'], ORDERS)
        self.assertEqual('cancel_preview_only', result['intent'])
        self.assertEqual(get_policy('binding_guard_v4').prompt, get_policy('binding_grounded_v5').prompt)
        self.assertEqual(get_policy('binding_guard_v4').tools, get_policy('binding_grounded_v5').tools)

    def test_catalog_function_words_preserve_similar_product_exclusion(self):
        orders=ORDERS+[dict(id='C',product='双肩背包',version=1,paid_cents=8900)]
        self.assertEqual('C',resolve('麻烦给背包办理退订。',[],orders,catalog_aliases=True)['order_id'])
        self.assertEqual('A',resolve('这笔耳机的购买请撤单。',[],orders,catalog_aliases=True)['order_id'])
        for text in ('请给有线耳机办理退订','请给头戴耳机办理退订','不要给耳机办理退订','如果给耳机办理退订'):
            with self.subTest(text=text), self.assertRaises(ServiceError):resolve(text,[],orders,catalog_aliases=True)


class Sequence:
    def __init__(self, order): self.responses = [dict(tool_calls=[dict(function=dict(name='preview_cancel', arguments=encode(dict(order_id=order))))]),dict(content='请确认')]
    def complete(self, messages, tools): return self.responses.pop(0)


class ServiceTests(unittest.TestCase):
    def test_catalog_payload_exposes_only_owned_controlled_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'s.sqlite');backend=MockBusinessBackend(Path(temp)/'b.sqlite')
            service=CustomerService(store,backend,policy_profile='binding_catalog_v6')
            conv=service.create_conversation('customer_alice')['conversation_id']
            listing=service._tool('customer_alice',conv,'read','list_orders',{})
            self.assertEqual({'order_A','order_B','order_C'},{row['id'] for row in listing['orders']})
            self.assertEqual(['耳机'],next(row['aliases'] for row in listing['orders'] if row['id']=='order_A'))
            service.model=Sequence('order_A')
            result=service.message('customer_alice',conv,'preview','麻烦给耳机办理退订。')
            self.assertEqual('pending_confirmation',result['operation']['status'])
            self.assertEqual('paid',backend.get_order('customer_alice','order_A')['order']['status'])

    def test_restart_amount_clarification_confirmation_and_stale_card(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'sessions.sqlite'); backend=MockBusinessBackend(Path(temp)/'business.sqlite')
            with backend.transaction() as db: db.execute("UPDATE orders SET product='无线耳机' WHERE id='order_B'")
            service=CustomerService(store,backend,Sequence('order_A'),policy_profile='binding_grounded_v5')
            conv=service.create_conversation('customer_alice')['conversation_id']
            first=service.message('customer_alice',conv,'one','想撤销无线耳机这笔订单，请帮忙处理。')
            self.assertIsNone(first['operation'])
            service=CustomerService(Store(store.path),backend,Sequence('order_B'),policy_profile='binding_grounded_v5');service.recover()
            result=service.message('customer_alice',conv,'two','就299元的那一笔。')
            operation=result['operation']; self.assertEqual('order_B',operation['order_id'])
            self.assertEqual('paid',backend.get_order('customer_alice','order_B')['order']['status'])
            service.model=Sequence('order_A')
            refused=service.message('customer_alice',conv,'three','改为取消订单order_A')
            self.assertEqual('order_A',refused['operation']['order_id'])
            with self.assertRaisesRegex(ServiceError,'confirmation_invalidated'):service.confirm('customer_alice',operation['id'])
            service.confirm('customer_alice',refused['operation']['id'])
            self.assertEqual('cancelled',backend.get_order('customer_alice','order_A')['order']['status'])
            self.assertEqual('paid',backend.get_order('customer_alice','order_B')['order']['status'])

    def test_model_order_id_cannot_replace_user_or_cross_owner_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'s.sqlite'); backend=MockBusinessBackend(Path(temp)/'b.sqlite')
            service=CustomerService(store,backend,Sequence('order_D'),policy_profile='binding_grounded_v5')
            conv=service.create_conversation('customer_alice')['conversation_id']
            result=service.message('customer_alice',conv,'r','计划取消无线耳机，请帮忙处理。')
            self.assertIsNone(result['operation'])
            with backend.transaction() as db: self.assertEqual(0,db.execute('SELECT COUNT(*) FROM cancellations').fetchone()[0])


if __name__=='__main__': unittest.main()
