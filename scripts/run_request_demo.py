"""Start only a completed v8 synthetic acceptance with the same model/code."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

from evaluate_scoped_workflow import ROOT,sha,assert_frozen
from request_workflow_acceptance import health_check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test',type=Path,default=ROOT/'outputs/request-workflow-test-20261015-r1')
    p.add_argument('--data-dir',type=Path,default=ROOT/'outputs/request-workflow-demo-20261015-r1')
    p.add_argument('--port',type=int,default=8766);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    result=json.loads((args.test/'summary.json').read_text(encoding='utf-8'))
    complete=json.loads((args.test/'completed.json').read_text(encoding='utf-8'))
    protocol=json.loads((args.test/'protocol.json').read_text(encoding='utf-8'))
    if not complete['complete'] or complete['summary_sha256']!=sha(args.test/'summary.json') or not result['predefined_criteria_met']:
        raise ValueError('completed_qualified_test_required')
    assert_frozen(protocol['source_sha256']);health=health_check(protocol)
    if args.check_only:
        print('Frozen v8 acceptance and inference identity verified. Mock business only.');return 0
    receipt=json.loads((ROOT/'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    return subprocess.call([sys.executable,str(ROOT/'scripts/run_customer_service.py'),
        '--mode','model','--model',health['model_id'],'--api-base','http://127.0.0.1:8647/v1',
        '--tokenizer-path',receipt['selected_adapter_dir'],'--knowledge-file',receipt['knowledge_file'],
        '--data-dir',str(args.data_dir),'--port',str(args.port),'--policy-profile','binding_state_v8',
        '--context-budget','8192','--recent-turns','6'])


if __name__=='__main__':raise SystemExit(main())
