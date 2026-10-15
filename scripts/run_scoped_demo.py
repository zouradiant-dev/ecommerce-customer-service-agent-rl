"""Start the tested v7 demo with exact frozen code/model and isolated mock data."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from urllib.request import urlopen

from evaluate_scoped_workflow import ROOT,sha,assert_frozen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test',type=Path,default=ROOT/'outputs/scoped-workflow-test-20261015-r1')
    p.add_argument('--data-dir',type=Path,default=ROOT/'outputs/scoped-workflow-demo-20261015-r1')
    p.add_argument('--api-base',default='http://127.0.0.1:8647/v1');p.add_argument('--port',type=int,default=8766)
    p.add_argument('--check-only',action='store_true');args=p.parse_args()
    result=json.loads((args.test/'summary.json').read_text(encoding='utf-8'))
    protocol=json.loads((args.test/'protocol.json').read_text(encoding='utf-8'))
    if not result['predefined_criteria_met']:raise ValueError('test_criteria_not_met_do_not_start_this_demo')
    assert_frozen(protocol['source_sha256'])
    with urlopen(args.api_base.removesuffix('/v1')+'/healthz',timeout=10) as stream:health=json.load(stream)
    if not health['ready'] or any(health[k]!=protocol['health'][k] for k in ('adapter_sha256','base_revision','model_id','precision')):
        raise ValueError('tested_model_changed')
    receipt=json.loads((ROOT/'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    if args.check_only:
        print('V7 frozen code/model checks passed. Mock business only.');return 0
    return subprocess.call([sys.executable,str(ROOT/'scripts/run_customer_service.py'),
        '--mode','model','--model',health['model_id'],'--api-base',args.api_base,
        '--tokenizer-path',receipt['selected_adapter_dir'],'--knowledge-file',receipt['knowledge_file'],
        '--data-dir',str(args.data_dir),'--port',str(args.port),'--policy-profile','binding_scoped_v7',
        '--context-budget','8192','--recent-turns','6'])


if __name__=='__main__':raise SystemExit(main())
