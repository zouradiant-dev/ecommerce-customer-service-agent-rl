"""Post-hoc development re-scoring of recorded replies; preserve all original evidence."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.serving.online_evaluation import CANCEL_REPLY_CHECK_VERSION, unsupported_cancel_success


def rescore(run):
    summary_raw = (run / 'summary.json').read_bytes()
    summary = json.loads(summary_raw)
    if summary.get('complete') is not True or summary.get('real_model_execution') is not True:
        raise ValueError('requires_complete_real_model_development_run')
    if summary.get('independent_test_claim') is not False:
        raise ValueError('posthoc_rescore_requires_development_not_formal_test')
    cases = []
    for case_id in summary['case_ids']:
        if Path(case_id).name != case_id or case_id in {'.', '..'}:
            raise ValueError('invalid_case_id')
        folder = run / 'cases' / case_id
        raw = (folder / 'result.json').read_bytes()
        result = json.loads(raw)
        history_raw = (folder / 'history.json').read_bytes()
        history = json.loads(history_raw)
        failures = list(result['failures'])
        # Re-check every existing phrase failure on the exact recorded content.
        # All other state, citation, checkpoint and lookup checks remain frozen.
        if 'unsupported_cancel_success_phrase' in failures:
            failures = [item for item in failures if item != 'unsupported_cancel_success_phrase']
            failures += ['unsupported_cancel_success_phrase' for message in history['messages']
                         if message.get('role') == 'assistant' and unsupported_cancel_success(str(message.get('content', '')))]
        cases.append({'case_id': case_id, 'original_passed': result['passed'], 'rescored_passed': not failures,
                      'original_failures': result['failures'], 'rescored_failures': failures,
                      'original_result_sha256': sha256(raw).hexdigest(),
                      'original_history_sha256': sha256(history_raw).hexdigest()})
    return {'schema_version': 'recorded_reply_development_rescore_v1',
            'original_summary_sha256': sha256(summary_raw).hexdigest(),
            'original_check_version': summary.get('cancel_reply_check_version', 'legacy_v1'),
            'new_check_version': CANCEL_REPLY_CHECK_VERSION,
            'check_source_sha256': sha256((ROOT / 'src/ecommerce_customer_service/serving/online_evaluation.py').read_bytes()).hexdigest(),
            'cases': cases, 'original_passed': sum(case['original_passed'] for case in cases),
            'rescored_passed': sum(case['rescored_passed'] for case in cases),
            'changed_case_ids': [case['case_id'] for case in cases if case['original_passed'] != case['rescored_passed']],
            'new_model_execution': False, 'weights_updated': False, 'original_results_modified': False,
            'independent_test_claim': False, 'automatic_promotion': False,
            'limitations': ['posthoc_development_measurement_correction', 'limited_phrase_check_not_semantic_entailment']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = rescore(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('original_passed', 'rescored_passed', 'changed_case_ids')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
