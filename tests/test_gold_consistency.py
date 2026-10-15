import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_scoped_development import validate_gold


class GoldTests(unittest.TestCase):
    def test_contradictory_terminal_checkpoint_is_rejected_before_model_calls(self):
        with self.assertRaisesRegex(ValueError,'contradicts'):
            validate_gold([dict(case_id='invalid',steps=[{'kind':'checkpoint','assertions':{'pending_count':1}}],expected={'pending_count':0})])

    def test_consistent_terminal_checkpoint_is_allowed(self):
        validate_gold([dict(case_id='valid',steps=[{'kind':'checkpoint','assertions':{'pending_count':1}}],expected={'pending_count':1})])


if __name__=='__main__':unittest.main()
