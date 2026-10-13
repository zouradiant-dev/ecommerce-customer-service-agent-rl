"""File-format fixtures only; no claims of served-model execution in these tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('policy_comparison', ROOT / 'scripts' / 'compare_online_policy_runs.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def run_fixture(self, folder, *, complete=True, budget=8192, smoke=False):
        folder.mkdir()
        case_ids = ['first', 'second']
        summary = {'mode': 'smoke' if smoke else 'model', 'real_model_execution': not smoke,
                   'evaluation_kind': 'served_model_on_simulated_business', 'model': 'fixture-format-only',
                   'complete': complete, 'case_sha256': 'same', 'case_ids': case_ids,
                   'context_counter': 'counter', 'context_budget': budget, 'context_policy': 'structured', 'recent_turns': 6,
                   'passed_cases': 1, 'evaluated_cases': 2 if complete else 1, 'expected_cases': 2,
                   'case_pass_rate': .5 if complete else 1.0, 'model_calls': 2, 'tool_calls': 1,
                   'system_prompt_sha256': 'prompt', 'tools_sha256': 'tools', 'source_sha256': {}, 'failures': {}}
        (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        for name in case_ids if complete else case_ids[:1]:
            case = folder / 'cases' / name
            case.mkdir(parents=True)
            (case / 'result.json').write_text(json.dumps({'passed': name == 'first'}), encoding='utf-8')

    def test_incomplete_run_is_not_ranked_even_if_input_file_claims_a_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_fixture(root / 'base')
            self.run_fixture(root / 'partial', complete=False)
            report = comparison.compare(root / 'base', {'candidate': root / 'partial'})
            row = report['rows'][0]
            self.assertIsNone(row['pass_rate'])
            self.assertIsNone(row['improved_cases'])
            self.assertIsNone(row['regressed_cases'])

    def test_changed_context_budget_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_fixture(root / 'base')
            self.run_fixture(root / 'candidate', budget=4096)
            with self.assertRaisesRegex(ValueError, 'nonpaired_run'):
                comparison.compare(root / 'base', {'candidate': root / 'candidate'})

    def test_smoke_cannot_be_presented_as_real_model_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_fixture(root / 'base', smoke=True)
            with self.assertRaisesRegex(ValueError, 'requires_real_model_mode'):
                comparison.compare(root / 'base', {'base': root / 'base'})


if __name__ == '__main__':
    unittest.main()
