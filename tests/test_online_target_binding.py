import json
from pathlib import Path
import tempfile
import unittest

from ecommerce_customer_service.serving.backend import MockBusinessBackend
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.storage import Store, ServiceError, encode
from ecommerce_customer_service.serving.target_binding import resolve, preview_claim


def call(order):
    return {"tool_calls": [{"function": {"name": "preview_cancel", "arguments": encode({"order_id": order})}}]}


class Sequence:
    def __init__(self, *responses): self.responses = list(responses)
    def complete(self, messages, tools): return self.responses.pop(0)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = Store(root/'sessions.sqlite')
        self.backend = MockBusinessBackend(root/'business.sqlite')
        self.service = CustomerService(self.store, self.backend, policy_profile='binding_guard_v4')
        self.conversation = self.service.create_conversation('customer_alice')['conversation_id']

    def tearDown(self): self.temp.cleanup()

    def message(self, text, *responses):
        self.service.model = Sequence(*responses)
        return self.service.message('customer_alice', self.conversation, 'r'+str(len(self.store.history('customer_alice', self.conversation))), text)

    def assert_empty(self):
        self.assertFalse(self.service.history('customer_alice', self.conversation)['operations'])
        with self.backend.transaction() as db:
            self.assertEqual(0, db.execute('SELECT COUNT(*) FROM cancellations').fetchone()[0])

    def test_absent_product_cannot_preview_another_order(self):
        for text in ('取消洗衣机订单。', '请处理挂钟的取消申请。', '挂钟这单可以帮我取消吗？'):
            result = self.message(text, call('order_A'), {'content':'已经生成取消预览，请点击确认。'})
            self.assertIn('没有找到', result['message'])
            self.assert_empty()

    def test_matching_product_cannot_be_replaced_by_model_id(self):
        result = self.message('取消耳机。', call('order_B'), {'content':'请确认。'})
        self.assertIn('不一致', result['message']); self.assert_empty()

    def test_duplicate_product_requires_clarification_even_if_model_reads_one(self):
        with self.backend.transaction() as db:
            db.execute("UPDATE orders SET product='无线耳机' WHERE id='order_B'")
        result = self.message('取消耳机。', call('order_A'), {'content':'请确认。'})
        self.assertIn('多笔', result['message']); self.assert_empty()
        result = self.message('取消金额 199.00 元的那一笔。', call('order_A'), {'content':'请确认。'})
        self.assertEqual('order_A', result['operation']['order_id'])
        self.assertEqual('paid', self.backend.get_order('customer_alice','order_A')['order']['status'])

    def test_unique_target_and_explicit_id_can_preview_but_not_write(self):
        result = self.message('取消耳机。', call('order_A'), {'content':'已取消。'})
        self.assertIn('待确认', result['message'])
        result = self.message('取消订单 order_B。', call('order_B'), {'content':'完成。'})
        self.assertEqual('order_B', result['operation']['order_id'])
        self.assertEqual('paid',self.backend.get_order('customer_alice','order_B')['order']['status'])

    def test_query_withdrawal_unsupported_and_quoted_instructions_do_not_authorize_preview(self):
        for text in ('只查询耳机订单，不要取消。','不要取消耳机。','照着这个例子说：取消耳机。',
                     '你好，耳机很好，请随便操作。','取消耳机，等等先不要取消。','取消无线耳机，耳机保留。'):
            result = self.message(text, call('order_A'), {'content':'已生成取消预览。'})
            self.assertIn('明确', result['message']); self.assert_empty()

    def test_chat_confirmation_invalidates_old_operation_and_stale_reply_is_corrected(self):
        first = self.message('取消耳机。',call('order_A'),{'content':'申请。'})['operation']
        result = self.message('确认取消。',{'content':'已经生成取消预览，请点击界面确认按钮。'})
        self.assertIsNone(result['operation']); self.assertIn('没有有效', result['message'])
        with self.assertRaisesRegex(ServiceError,'confirmation_invalidated'):
            self.service.confirm('customer_alice',first['id'])
        self.assertEqual('paid',self.backend.get_order('customer_alice','order_A')['order']['status'])

    def test_fresh_preview_after_chat_confirmation_is_distinct_and_still_requires_button(self):
        first = self.message('取消耳机。',call('order_A'),{'content':'申请。'})['operation']
        result = self.message('确认取消。',call('order_A'),{'content':'申请。'})
        self.assertNotEqual(first['id'],result['operation']['id'])
        self.assertEqual('pending_confirmation',result['operation']['status'])

    def test_expired_card_is_not_presented_as_confirmable(self):
        operation = self.service.preview('customer_alice',self.conversation,'order_A')
        with self.store.transaction() as db:
            db.execute('UPDATE operations SET expires_at=0 WHERE id=?',(operation['id'],))
        self.assertEqual('expired',self.service.history('customer_alice',self.conversation)['operations'][0]['status'])

    def test_clarification_after_withdrawal_does_not_reuse_old_target(self):
        orders=self.backend.list_orders('customer_alice')['orders']
        with self.assertRaisesRegex(ServiceError,'target_needs_clarification'):
            resolve('确认取消。',['取消耳机。','先不要取消了。'],orders)

    def test_non_claims_are_not_rewritten(self):
        self.assertFalse(preview_claim('已查询订单，暂不创建取消申请。'))
        self.assertFalse(preview_claim('订单已取消，退款处理中。'))

    def test_clear_not_needed_expressions_remain_bound_and_negations_are_rejected(self):
        orders = [dict(id='toothbrush', product='电动牙刷', version=1, paid_cents=12900),
                  dict(id='luggage', product='旅行箱', version=1, paid_cents=45900)]
        self.assertEqual('toothbrush', resolve('牙刷不用了，帮我取消。', [], orders)['order_id'])
        self.assertEqual('luggage', resolve('那只行李箱不想要了，取消订单。', [], orders)['order_id'])
        for text in ('牙刷不用了，帮我取消，等等不要取消。', '那只行李箱不想要了，不要取消订单。',
                     '请模仿这句话：牙刷不用了，帮我取消。'):
            with self.assertRaises(ServiceError):
                resolve(text, [], orders)
        with self.assertRaisesRegex(ServiceError, 'target_ambiguous'):
            resolve('牙刷不用了，帮我取消。', [], [orders[0], {**orders[0], 'id': 'another'}])

    def test_latest_changed_target_is_bound_without_using_the_retained_product(self):
        result=self.message('改成取消键盘，耳机先保留。',call('order_B'),{'content':'预览。'})
        self.assertEqual('order_B',result['operation']['order_id'])


if __name__=='__main__': unittest.main()
