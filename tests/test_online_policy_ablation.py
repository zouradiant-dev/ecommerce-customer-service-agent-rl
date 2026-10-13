from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from ecommerce_customer_service.serving.context_builder import ContextBuilder
from ecommerce_customer_service.serving.model import SYSTEM_PROMPT, online_protocol_metadata, tool_schemas
from ecommerce_customer_service.serving.online_cases import defaults
from ecommerce_customer_service.serving.online_evaluation import run_suite
from ecommerce_customer_service.serving.policy import FACTORIAL_PROFILES, PROFILES, get_policy
from ecommerce_customer_service.serving.binding_dev_cases import cases as binding_cases
from ecommerce_customer_service.serving.online_cases import prepare


class PolicyAblationTests(unittest.TestCase):
    def test_baseline_is_byte_identical_to_first_real_protocol(self):
        policy = get_policy()
        self.assertEqual(SYSTEM_PROMPT, policy.prompt)
        self.assertEqual(tool_schemas(), policy.tools)
        metadata = policy.metadata()
        self.assertEqual(online_protocol_metadata(), {key: metadata[key] for key in online_protocol_metadata()})

    def test_ablations_change_only_the_named_factor(self):
        baseline, prompt, tools, combined = [get_policy(profile) for profile in FACTORIAL_PROFILES]
        self.assertNotEqual(baseline.prompt, prompt.prompt)
        self.assertEqual(baseline.tools, prompt.tools)
        self.assertEqual(baseline.prompt, tools.prompt)
        self.assertNotEqual(baseline.tools, tools.tools)
        self.assertEqual(prompt.prompt, combined.prompt)
        self.assertEqual(tools.tools, combined.tools)
        for policy in (prompt, tools, combined):
            for before, after in zip(baseline.tools, policy.tools):
                self.assertEqual(before['function']['name'], after['function']['name'])
                self.assertEqual(before['function']['parameters'], after['function']['parameters'])

    def test_each_call_gets_its_own_schema_objects(self):
        candidate = get_policy('binding_tools_v2')
        candidate.tools[0]['function']['parameters']['properties']['customer_id'] = {'type': 'string'}
        self.assertNotIn('customer_id', get_policy('binding_tools_v2').tools[0]['function']['parameters']['properties'])

    def test_unknown_profile_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            with self.assertRaisesRegex(ValueError, 'unknown_policy_profile'):
                run_suite(defaults(), output, mode='smoke', policy_profile='unknown')
            self.assertFalse(output.exists())

    def test_profiles_preserve_engineering_behavior_and_record_actual_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for profile in PROFILES:
                with self.subTest(profile=profile):
                    output = Path(temporary) / profile
                    result = run_suite(defaults(), output, mode='smoke', policy_profile=profile,
                                       context=ContextBuilder(max_input_tokens=32768))
                    self.assertEqual(15, result['passed_cases'])
                    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
                    metadata = get_policy(profile).metadata()
                    self.assertEqual(profile, protocol['policy_profile'])
                    self.assertEqual(metadata['online_prompt_sha256'], protocol['system_prompt_sha256'])
                    self.assertEqual(metadata['online_tools_sha256'], protocol['tools_sha256'])
                    self.assertFalse(protocol['real_model_execution'])
                    self.assertFalse(protocol['independent_test_claim'])

    def test_extended_guards_are_dev_and_use_new_orders(self):
        cases = binding_cases()
        self.assertEqual(12, len(cases))
        self.assertTrue(all(case['split'] == 'dev' for case in cases))
        self.assertTrue(all('order_A' not in case['expected']['order_statuses'] for case in cases))
        with tempfile.TemporaryDirectory() as temporary:
            result = run_suite(cases, Path(temporary) / 'run', mode='smoke',
                policy_profile='binding_combined_v2', context=ContextBuilder(max_input_tokens=32768))
            self.assertEqual(12, result['passed_cases'])

    def test_first_turn_ambiguity_is_checked_even_when_final_order_is_correct(self):
        case = deepcopy(next(case for case in binding_cases() if case['case_id'] == 'same_product_clarification'))
        # Incorrectly guessing before clarification can otherwise pass final state checks.
        case['smoke_responses'] = prepare('R-731') + prepare('R-731', listing=False)
        with tempfile.TemporaryDirectory() as temporary:
            result = run_suite([case], Path(temporary) / 'run', mode='smoke',
                context=ContextBuilder(max_input_tokens=32768))
            self.assertEqual(0, result['passed_cases'])
            self.assertIn('checkpoint', result['failures'])

    def test_duplicate_initial_order_rejected(self):
        case = binding_cases()[0]
        case['initial_orders'].append(deepcopy(case['initial_orders'][0]))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'duplicate_initial_order'):
                run_suite([case], Path(temporary) / 'run', mode='smoke')


if __name__ == '__main__':
    unittest.main()
