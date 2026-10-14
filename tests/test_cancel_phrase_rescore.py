import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ecommerce_customer_service.serving.online_evaluation import unsupported_cancel_success

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rescore', ROOT / 'scripts/rescore_online_cancel_phrases.py')
rescore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescore)


class ClaimTests(unittest.TestCase):
    def test_negated_success_is_not_a_success_assertion(self):
        for text in ('预览不是取消成功。', '预览不等于已经取消。', '预览不意味着已取消。',
                     '当前无法取消。', '订单没有取消成功。'):
            self.assertFalse(unsupported_cancel_success(text), text)

    def test_negated_preview_cannot_mask_a_separate_positive_claim(self):
        for text in ('订单已取消。', '预览不是取消成功，但该订单已经取消。',
                     '预览不等于已经取消；我已经取消订单。'):
            self.assertTrue(unsupported_cancel_success(text), text)

    def test_posthoc_correction_keeps_other_failures_and_original_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            summary = {'complete': True, 'real_model_execution': True, 'independent_test_claim': False,
                       'case_ids': ['quote', 'lookup', 'positive']}
            (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            for name, reply, other in [('quote', '预览不是取消成功。', []),
                                       ('lookup', '预览不是取消成功。', ['knowledge_lookup_missing']),
                                       ('positive', '订单已取消。', [])]:
                folder = run / 'cases' / name
                folder.mkdir(parents=True)
                result = {'passed': False,
                          'failures': ['unsupported_cancel_success_phrase', *other]}
                (folder / 'result.json').write_text(json.dumps(result), encoding='utf-8')
                (folder / 'history.json').write_text(json.dumps({'messages': [{'role': 'assistant', 'content': reply}]}), encoding='utf-8')
            before = {path: path.read_bytes() for path in run.rglob('*.json')}
            report = rescore.rescore(run)
            self.assertEqual(['quote'], report['changed_case_ids'])
            self.assertEqual(1, report['rescored_passed'])
            self.assertFalse(report['new_model_execution'])
            self.assertEqual(before, {path: path.read_bytes() for path in run.rglob('*.json')})


if __name__ == '__main__':
    unittest.main()
