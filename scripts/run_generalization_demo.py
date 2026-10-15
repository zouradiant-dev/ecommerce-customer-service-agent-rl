"""Open the evaluated controller with its exact model; isolated mock business only."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir',type=Path,default=ROOT/'outputs/catalog-generalization-real-20261015-r2')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'outputs/catalog-generalization-demo-20261015-r1')
    parser.add_argument('--api-base',default='http://127.0.0.1:8647/v1')
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    result=json.loads((args.evaluation_dir/'completed.json').read_text(encoding='utf-8'))
    protocol=json.loads((args.evaluation_dir/'protocol.json').read_text(encoding='utf-8'))
    if not result['gate_passed'] or not result['acceptance_performed'] or not result['acceptance_result']['complete']:
        raise ValueError('completed_dev_and_acceptance_required_for_this_demo')
    for name,sha in protocol['frozen_source'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('evaluated_source_changed:'+name)
    with urlopen(args.api_base.removesuffix('/v1')+'/healthz',timeout=10) as response:health=json.load(response)
    if not health['ready'] or any(health[key]!=protocol['health'][key] for key in ('model_id','adapter_sha256','base_revision','precision')):
        raise ValueError('evaluated_inference_identity_changed')
    profile=result.get('profile','binding_grounded_v5')
    print('Evaluated controller: '+profile+'. Same frozen adapter. Mock business only.',flush=True)
    if args.check_only:return 0
    receipt=json.loads((ROOT/'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    return subprocess.call([sys.executable,str(ROOT/'scripts/run_customer_service.py'),
        '--mode','model','--model',health['model_id'],'--api-base',args.api_base,
        '--tokenizer-path',receipt['selected_adapter_dir'],'--knowledge-file',receipt['knowledge_file'],
        '--data-dir',str(args.data_dir),'--port',str(args.port),'--policy-profile',profile,
        '--context-budget','8192','--recent-turns','6'])


if __name__=='__main__':raise SystemExit(main())
