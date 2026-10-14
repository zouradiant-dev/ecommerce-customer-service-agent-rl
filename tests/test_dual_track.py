"""Meaningful CPU guards: reward diagnosis, frozen inputs, no optimizer and knowledge wiring."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ecommerce_customer_service.dual_track import commands, runtime_hashes, validate, write_json
from ecommerce_customer_service.reward_signal import describe
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.knowledge_cases import cases
from ecommerce_customer_service.serving.online_evaluation import run_suite

ROOT = Path(__file__).resolve().parents[1]


class RewardSignalTests(unittest.TestCase):
    def row(self, trial, strict=1, shaped=1):
        return {'task_id': 'v18_train_fixture_0000', 'trial': trial, 'split': 'train', 'stage': 'sft',
                'learned_model_result': True, 'verification': {'hard_safety_failures': []},
                'reward_components': {'strict_binary_v18': strict, 'train_shaped_v18': shaped}}

    def test_flat_groups_do_not_authorize_training(self):
        result = describe([self.row(i) for i in range(4)])
        self.assertEqual(0, result['rewards']['train_shaped_v18']['mixed_group_rate'])
        self.assertFalse(result['formal_rl_gate_passed'])
        self.assertFalse(result['human_review_complete'])

    def test_real_variation_and_unsafe_positive_are_reported_separately(self):
        bad = self.row(0, 0, .2)
        bad['verification']['hard_safety_failures'] = ['unsafe']
        result = describe([bad, self.row(1)])
        self.assertEqual(1, result['rewards']['train_shaped_v18']['mixed_groups'])
        self.assertEqual(['v18_train_fixture_0000'], result['unsafe_positive_ids'])
        self.assertFalse(result['reward_calibrated'])

    def test_test_rows_fixtures_duplicates_and_nonfinite_rewards_are_rejected(self):
        for field, value in [('split', 'dev'), ('stage', 'grpo'), ('learned_model_result', False)]:
            bad = self.row(0); bad[field] = value
            with self.assertRaises(ValueError):
                describe([bad])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            describe([self.row(0), self.row(0)])
        with self.assertRaisesRegex(ValueError, 'invalid_reward'):
            describe([self.row(0, shaped=float('nan'))])


class DualTrackTests(unittest.TestCase):
    def test_commands_keep_protocols_and_model_identities_separate(self):
        root = Path('ROOT')
        jobs = commands(root, Path('plan'), Path('out'), python='python', manifest=Path('tasks'),
                        original=Path('ORIGINAL'), candidate=Path('CANDIDATE'), base=Path('BASE'))
        self.assertEqual(4, len(jobs['agent']))
        self.assertEqual(1, len(jobs['rl']))
        for job in jobs['agent']:
            self.assertIn('--policy-profile', job)
            self.assertEqual('binding_guard_v4', job[job.index('--policy-profile') + 1])
        self.assertNotIn('--knowledge-file', jobs['agent'][0])
        self.assertIn('--knowledge-file', jobs['agent'][2])
        self.assertEqual('ORIGINAL', jobs['rl'][0][jobs['rl'][0].index('--adapter') + 1])
        self.assertNotIn('CANDIDATE', jobs['rl'][0])
        self.assertTrue(all('train_grpo' not in token and 'train_sft' not in token for group in jobs.values()
                            for job in group for token in job))

    def test_source_payload_and_artifact_drift_block_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'src').mkdir(); (root / 'scripts').mkdir(); (root / 'configs').mkdir()
            (root / 'src/runtime.py').write_text('version=1', encoding='utf-8')
            (root / 'configs/v18-reward.json').write_text('{}', encoding='utf-8')
            payload = root / 'payload.json'; payload.write_text('{}', encoding='utf-8')
            from ecommerce_customer_service.local_hf import sha256_file
            plan = {'schema_version': 'dual_track_dev_plan_v1', 'training_authorized': False,
                    'source_sha256': runtime_hashes(root), 'payload_sha256': {'payload.json': sha256_file(payload)},
                    'artifacts': {'original': {'a': 'fixed'}, 'candidate': {'a': 'fixed'}}}
            from ecommerce_customer_service.dual_track import BASE_REVISION, SCENARIOS, SELECTION_SEED
            plan.update(base_revision=BASE_REVISION, agent={'policy_profile': 'binding_guard_v4'},
                        rl={'selection_seed': SELECTION_SEED, 'scenarios': sorted(SCENARIOS)})
            write_json(root / 'plan.json', plan)
            options = dict(manifest=root / 'unused', original=root, candidate=root, track='agent')
            with patch('ecommerce_customer_service.dual_track.adapter_files', return_value={'a': 'fixed'}):
                validate(root, root, **options)
                (root / 'src/runtime.py').write_text('version=2', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'runtime_changed'):
                    validate(root, root, **options)
                (root / 'src/runtime.py').write_text('version=1', encoding='utf-8')
                payload.write_text('{"changed":true}', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'payload_changed'):
                    validate(root, root, **options)
                payload.write_text('{}', encoding='utf-8')
            with patch('ecommerce_customer_service.dual_track.adapter_files', return_value={'a': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'adapter_changed'):
                    validate(root, root, **options)

    def test_knowledge_suite_checks_lookup_abstention_restart_and_no_writes(self):
        index = KnowledgeIndex(ROOT / 'data/demo-policies.jsonl')
        with tempfile.TemporaryDirectory() as temporary:
            result = run_suite(cases(), Path(temporary) / 'good', mode='smoke', policy_profile='binding_guard_v4', knowledge=index)
            self.assertEqual(6, result['passed_cases'])
            self.assertFalse(result['real_model_execution'])
            bad = deepcopy(cases()[0]); bad['smoke_responses'] = [{'content': '退款立即到账。'}]
            result = run_suite([bad], Path(temporary) / 'bad', mode='smoke', policy_profile='binding_guard_v4', knowledge=index)
            self.assertEqual(0, result['passed_cases'])
            self.assertIn('knowledge_lookup_missing', result['failures'])


if __name__ == '__main__':
    unittest.main()
