"""Exclusive one-shot synthetic acceptance after a hash-bound dev selection."""
from pathlib import Path
import json
from evaluate_closeout_adapter import parser, evaluate
from ecommerce_customer_service.serving.closeout_experiment import CONFIG, claim_test, verify_online_adapter, choose_adapter
from ecommerce_customer_service.serving.artifacts import file_sha256


def main():
    p=parser()
    p.description=__doc__
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    args=p.parse_args()
    selection=json.loads(args.selection.read_text(encoding='utf-8'))
    if selection.get('protocol_config')!=CONFIG: raise ValueError('selected_protocol_changed')
    comparisons={}
    for suite,digest in selection['paired_comparisons_sha256'].items():
        path=args.selection.parent/(suite+'-comparison.json')
        if suite not in {'business','knowledge'} or file_sha256(path)!=digest:
            raise ValueError('dev_selection_evidence_changed')
        comparisons[suite]=json.loads(path.read_text(encoding='utf-8'))
    if choose_adapter(comparisons)['selected']!=selection['selected']:
        raise ValueError('selection_does_not_follow_frozen_dev_rule')
    identity=verify_online_adapter(args.adapter_dir)
    if identity['adapter_sha256']!=selection['selected_adapter_sha256']:
        raise ValueError('selected_adapter_changed_before_test')
    if file_sha256(args.knowledge_file)!=selection['knowledge_sha256']:
        raise ValueError('selected_knowledge_changed_before_test')
    claim_test(selection,args.cases,args.ledger)
    return evaluate(args,sealed=True)


if __name__=='__main__': raise SystemExit(main())
