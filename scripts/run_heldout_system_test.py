"""One-shot evaluation of a frozen synthetic holdout; never tune from test here."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.serving.context_builder import ContextBuilder, load_token_counter
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.online_evaluation import initial_orders, run_case
from ecommerce_customer_service.serving.backend import MockBusinessBackend


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def wilson(passed, total):
    if not total: return None
    z = 1.959963984540054; proportion = passed / total
    denominator = 1 + z*z/total
    middle = (proportion + z*z/(2*total)) / denominator
    width = z*math.sqrt(proportion*(1-proportion)/total + z*z/(4*total*total))/denominator
    return [max(0, middle-width), min(1, middle+width)]


class RecordingModel:
    """Only the service's generation input reaches the remote model, never gold."""
    def __init__(self, target, stream):
        self.target, self.stream, self.model, self.case_id = target, stream, target.model, None

    def complete(self, messages, tools):
        payload = {'messages': messages, 'tools': tools}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        record = dict(case_id=self.case_id, input_sha256=hashlib.sha256(encoded.encode()).hexdigest(), **payload)
        started = time.perf_counter()
        try:
            response = self.target.complete(messages, tools)
            record['response'] = response
            return response
        except Exception as error:
            record['exception_type'] = type(error).__name__
            raise
        finally:
            record['elapsed_seconds'] = time.perf_counter() - started
            self.stream.write(json.dumps(record, ensure_ascii=False) + '\n'); self.stream.flush()


def ledger_audit(case, directory):
    db = sqlite3.connect((directory / 'business.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    cancelled = [dict(r) for r in db.execute('SELECT * FROM cancellations')]
    refunds = [dict(r) for r in db.execute('SELECT * FROM refunds')]; db.close()
    allowed = {order for order, status in case['expected']['order_statuses'].items()
               if status == 'cancelled' and next(r['status'] for r in case['initial_orders'] if r['id'] == order) != 'cancelled'}
    counts = Counter(r['order_id'] for r in cancelled)
    paid = {r['id']: r['paid_cents'] for r in case['initial_orders']}
    return dict(unauthorized_writes=len(cancelled) if not any(s['kind'] == 'confirm' for s in case['steps']) else 0,
                wrong_order_writes=sum(r['order_id'] not in allowed for r in cancelled),
                duplicate_writes=sum(max(0, count-1) for count in counts.values()),
                invalid_refunds=sum(r['order_id'] not in allowed or r['amount_cents'] != paid.get(r['order_id']) for r in refunds),
                cancellations=cancelled, refunds=refunds)


def check_runtime(protocol, api_base):
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT / name) != digest: raise ValueError('frozen_runtime_changed:' + name)
    with urlopen(api_base.removesuffix('/v1') + '/healthz', timeout=10) as stream: health = json.load(stream)
    if not health['ready']: raise ValueError('model_not_ready')
    for key in ('model_id', 'adapter_sha256', 'base_revision', 'precision', 'tokenizer_sha256'):
        if health[key] != protocol['health'][key]: raise ValueError('frozen_model_changed:' + key)
    return health


def preflight(cases):
    if len(cases) != 60 or len({c['case_id'] for c in cases}) != 60: raise ValueError('expected_60_unique_cases')
    # Structural validation only; neither model nor business resolver is called.
    with tempfile.TemporaryDirectory(prefix='heldout-preflight-') as temporary:
        for case in cases:
            if case['split'] != 'test' or 'smoke_responses' in case: raise ValueError('invalid_holdout')
            initial_orders(case, MockBusinessBackend(Path(temporary) / (case['case_id'] + '.sqlite')))
            ids = {r['id'] for r in case['initial_orders']}
            if set(case['expected']['order_statuses']) != ids: raise ValueError('all_orders_must_be_asserted')
            for step in case['steps']:
                if step['kind'] not in {'message', 'checkpoint', 'confirm', 'event', 'query'}: raise ValueError('unknown_step')
                if step['kind'] == 'message' and not step['text'].strip(): raise ValueError('empty_user_input')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api-base', default='http://127.0.0.1:8647/v1')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume-after-ledger-path-error', action='store_true',
                        help='Recover the first completed raw case without issuing its model calls again.')
    args = parser.parse_args()
    manifest = json.loads((args.cases_dir / 'manifest.json').read_text(encoding='utf-8'))
    for name, key in [('cases.jsonl', 'cases_sha256'), ('system-freeze.json', 'protocol_sha256'), ('overlap-audit.json', 'overlap_audit_sha256')]:
        if sha(args.cases_dir / name) != manifest[key]: raise ValueError('frozen_test_changed:' + name)
    if not manifest['gate_passed']: raise ValueError('overlap_gate_failed')
    protocol = json.loads((args.cases_dir / 'system-freeze.json').read_text(encoding='utf-8'))
    cases = [json.loads(line) for line in (args.cases_dir / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    preflight(cases)
    check_runtime(protocol, args.api_base)
    receipt = json.loads((ROOT / 'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    context = ContextBuilder(counter=load_token_counter(receipt['selected_adapter_dir']), max_input_tokens=8192, recent_turns=6)
    knowledge = KnowledgeIndex(Path(receipt['knowledge_file']))
    if knowledge.sha256 != protocol['knowledge_sha256']: raise ValueError('frozen_knowledge_changed')
    if args.check_only:
        print('HOLDOUT CPU/HEALTH PREFLIGHT PASSED; no test model request was made.'); return 0
    results = []
    if args.resume_after_ledger_path_error:
        claim = json.loads((args.cases_dir / 'execution-claim.json').read_text(encoding='utf-8'))
        if claim['output'] != str(args.output.resolve()) or claim['manifest_sha256'] != sha(args.cases_dir / 'manifest.json'):
            raise ValueError('original_execution_claim_mismatch')
        directories = list((args.output / 'cases').iterdir())
        first = args.output / 'cases' / cases[0]['case_id']
        if directories != [first] or not (first / 'result.json').exists() or (first / 'ledger-audit.json').exists():
            raise ValueError('only_first_case_postprocessing_error_may_be_resumed')
        requests = [json.loads(line) for line in (args.output / 'model-requests.jsonl').read_text(encoding='utf-8').splitlines()]
        if not requests or any(r['case_id'] != cases[0]['case_id'] or 'exception_type' in r for r in requests):
            raise ValueError('first_case_model_calls_not_fully_completed')
        row = json.loads((first / 'result.json').read_text(encoding='utf-8'))
        audit = ledger_audit(cases[0], first)
        with (args.output / 'postprocessing-recovery.json').open('x', encoding='utf-8') as receipt_stream:
            json.dump({'reason': 'relative_path_as_uri_in_ledger_postprocessing', 'model_calls_replayed': 0,
                       'retained_case': cases[0]['case_id'], 'raw_result_sha256': sha(first / 'result.json'),
                       'request_log_before_resume_sha256': sha(args.output / 'model-requests.jsonl'),
                       'first_case_already_completed_model_calls': len(requests),
                       'runner_after_fix_sha256': sha(Path(__file__)),
                       'frozen_business_runtime_unchanged': True}, receipt_stream, indent=2)
        write(first / 'ledger-audit.json', audit)
        row['ledger_audit'] = {k:v for k,v in audit.items() if k not in {'cancellations', 'refunds'}}
        for name, count in row['ledger_audit'].items():
            if count: row['failures'].append('ledger:' + name)
        row['passed'] = not row['failures']; results.append(row)
        write(first / 'result.json', row)
        print('RECOVERED first raw case; no model call replayed. Continue with case 2.', flush=True)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        # Atomic claim survives failure and refuses rerunning this test elsewhere.
        with (args.cases_dir / 'execution-claim.json').open('x', encoding='utf-8') as stream:
            json.dump({'started_at': datetime.now(timezone.utc).isoformat(), 'output': str(args.output.resolve()),
                       'manifest_sha256': sha(args.cases_dir / 'manifest.json'), 'one_attempt': True}, stream)
        write(args.output / 'protocol.json', protocol)
    with (args.output / 'model-requests.jsonl').open('a' if args.resume_after_ledger_path_error else 'x', encoding='utf-8') as stream:
        model = RecordingModel(OpenAIChatModel(args.api_base, protocol['health']['model_id']), stream)
        for index, case in enumerate(cases):
            if index < len(results) and results[index]['case_id'] == case['case_id']:
                continue
            check_runtime(protocol, args.api_base)
            model.case_id = case['case_id']
            print(json.dumps(dict(event='HELDOUT_CASE_START', index=index+1, total=len(cases), case_id=case['case_id'], category=case['category'])), flush=True)
            directory = args.output / 'cases' / case['case_id']
            row = run_case(case, directory, model, context, policy_profile=protocol['policy_profile'], knowledge=knowledge)
            audit = ledger_audit(case, directory)
            write(directory / 'ledger-audit.json', audit)
            row['ledger_audit'] = {k:v for k,v in audit.items() if k not in {'cancellations', 'refunds'}}
            for name, count in row['ledger_audit'].items():
                if count: row['failures'].append('ledger:' + name)
            row['passed'] = not row['failures']; results.append(row)
            write(directory / 'result.json', row)
            print(json.dumps(dict(event='HELDOUT_CASE_DONE', completed=index+1, total=len(cases),
                                  passed=row['passed'], cumulative_passes=sum(r['passed'] for r in results))), flush=True)
            if row['model_infrastructure_failures']: break
    check_runtime(protocol, args.api_base)
    passed = sum(r['passed'] for r in results)
    complete = len(results) == len(cases)
    ledger = {key: sum(r['ledger_audit'][key] for r in results) for key in ('unauthorized_writes', 'wrong_order_writes', 'duplicate_writes', 'invalid_refunds')}
    infrastructure = sum(r['model_infrastructure_failures'] for r in results)
    summary = dict(evaluation='frozen_synthetic_holdout_once', complete=complete,
        expected_cases=len(cases), evaluated_cases=len(results), passed_cases=passed,
        strict_case_pass_rate=passed/len(cases) if complete else None,
        descriptive_wilson_95=wilson(passed, len(results)),
        confidence_interval_caution='Hand-authored stratified synthetic cases are not IID traffic; this is descriptive only.',
        by_category={category: {'passed': sum(r['passed'] for r in results if r['category']==category),
                                'total': sum(r['category']==category for r in results)} for category in sorted({r['category'] for r in results})},
        ledger_audit=ledger, model_output_failures=sum(r['model_output_failures'] for r in results),
        infrastructure_failures=infrastructure, model_calls=sum(r['model_calls'] for r in results),
        tool_calls=sum(r['tool_calls'] for r in results),
        failures=dict(Counter(f.split(':')[0] for r in results for f in r['failures'])),
        preregistered_criteria_met=complete and passed/len(cases)>=.90 and not any(ledger.values()) and not infrastructure,
        no_training=True, no_test_driven_runtime_changes=True, production_promoted=False,
        labels_in_model_input=False, inspected_test_now_consumed=True,
        evaluator_postprocessing_recovered_without_case_replay=args.resume_after_ledger_path_error,
        limitations=['Synthetic test authored by the same project agent, not independent humans or real traffic.',
                     'No adaptive user simulation; explicit UI events are specified in advance.',
                     'Automatic state checks plus limited phrase/citation checks do not assess every reply semantically.',
                     'No new training and no independent estimate of improvement versus a baseline model.'])
    write(args.output / 'summary.json', summary)
    (args.output / 'bad_cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in results if not r['passed']), encoding='utf-8')
    write(args.output / 'completed.json', {'completed': complete, 'one_attempt': True, 'test_consumed': True,
        'summary_sha256': sha(args.output / 'summary.json'), 'no_automatic_retry_or_training': True})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if complete else 2


if __name__ == '__main__': raise SystemExit(main())
