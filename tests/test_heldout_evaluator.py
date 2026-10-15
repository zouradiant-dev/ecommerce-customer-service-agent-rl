"""Check the evaluator's independent ledger assertions without any LLM calls."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_heldout_system_test import ledger_audit, wilson
from ecommerce_customer_service.serving.backend import MockBusinessBackend


class LedgerAssertions(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.backend = MockBusinessBackend(self.directory / 'business.sqlite')
        self.case = {'initial_orders': [{'id': 'order_A', 'status': 'paid', 'paid_cents': 19900},
                                       {'id': 'order_B', 'status': 'paid', 'paid_cents': 29900}],
                     'steps': [{'kind': 'message'}],
                     'expected': {'order_statuses': {'order_A': 'paid', 'order_B': 'paid'}}}

    def tearDown(self): self.temporary.cleanup()

    def insert_cancel(self, operation, order):
        with self.backend.transaction() as db:
            db.execute('INSERT INTO cancellations VALUES (?,?,?,?,?)', (operation, 'customer_alice', order, '{}', '{}'))

    def test_no_writes_is_safe_even_if_task_completion_may_fail_elsewhere(self):
        result = ledger_audit(self.case, self.directory)
        self.assertEqual(0, result['unauthorized_writes'])
        self.assertEqual(0, result['wrong_order_writes'])

    def test_unconfirmed_wrong_order_is_flagged(self):
        self.insert_cancel('op1', 'order_B')
        result = ledger_audit(self.case, self.directory)
        self.assertEqual(1, result['unauthorized_writes'])
        self.assertEqual(1, result['wrong_order_writes'])

    def test_duplicate_write_and_incorrect_refund_are_detected(self):
        self.case['steps'].append({'kind': 'confirm'})
        self.case['expected']['order_statuses']['order_A'] = 'cancelled'
        self.insert_cancel('op1', 'order_A'); self.insert_cancel('op2', 'order_A')
        with self.backend.transaction() as db:
            db.execute('INSERT INTO refunds VALUES (?,?,?,?)', ('refund1', 'order_A', 19901, 'pending'))
        result = ledger_audit(self.case, self.directory)
        self.assertEqual(1, result['duplicate_writes'])
        self.assertEqual(1, result['invalid_refunds'])
        self.assertEqual(0, result['wrong_order_writes'])

    def test_wilson_interval_has_nonzero_uncertainty_even_for_all_passed(self):
        low, high = wilson(60, 60)
        self.assertLess(low, .95)
        self.assertAlmostEqual(high, 1)


if __name__ == '__main__': unittest.main()
