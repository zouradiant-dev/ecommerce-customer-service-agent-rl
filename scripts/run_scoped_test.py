"""Paired, once-only synthetic test after v7 has passed development selection."""
from pathlib import Path
import argparse
import json
import sys
from urllib.request import urlopen

from evaluate_scoped_workflow import ROOT,sha,write,read_rows,assert_frozen,run_stage
from run_heldout_system_test import wilson
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.context_builder import ContextBuilder,load_token_counter
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.model import OpenAIChatModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases-dir',type=Path,required=True);p.add_argument('--development',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--api-base',default='http://127.0.0.1:8647/v1');args=p.parse_args()
    protocol=json.loads((args.cases_dir/'protocol.json').read_text(encoding='utf-8'))
    selection_path=args.development/protocol.get('development_selection_name','selection.json')
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    manifest=json.loads((args.cases_dir/'manifest.json').read_text(encoding='utf-8'))
    if not selection['gate_passed']:raise ValueError('development_gate_not_passed')
    if sha(selection_path)!=protocol['development_selection_sha256']:raise ValueError('selected_development_changed')
    for name,digest in manifest['sha256'].items():
        if sha(args.cases_dir/name)!=digest:raise ValueError('frozen_test_changed:'+name)
    assert_frozen(protocol['source_sha256'])
    cases=read_rows(args.cases_dir/'cases.jsonl')
    if len(cases)!=60 or any(c['split']!='test' or 'smoke_responses' in c for c in cases):raise ValueError('expected_60_unexecuted_test_cases')
    with urlopen(args.api_base.removesuffix('/v1')+'/healthz',timeout=10) as stream:health=json.load(stream)
    if not health['ready'] or any(health[k]!=protocol['health'][k] for k in ('adapter_sha256','base_revision','model_id','precision','tokenizer_sha256')):
        raise ValueError('frozen_inference_changed')
    receipt=json.loads((ROOT/'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    context=ContextBuilder(counter=load_token_counter(receipt['selected_adapter_dir']),max_input_tokens=8192,recent_turns=6)
    knowledge=KnowledgeIndex(Path(receipt['knowledge_file']))
    if knowledge.sha256!=protocol['knowledge_sha256']:raise ValueError('frozen_knowledge_changed')
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.cases_dir/'execution-claim.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(one_execution_per_profile=True,profiles=['binding_catalog_v6','binding_scoped_v7'],
                       output=str(args.output.resolve()),manifest_sha256=sha(args.cases_dir/'manifest.json')),stream)
    write(args.output/'protocol.json',protocol)
    baseline,old=run_stage(cases,args.output/'test-v6','binding_catalog_v6',OpenAIChatModel(args.api_base,health['model_id']),context,knowledge,protocol['source_sha256'])
    if not baseline['complete'] or baseline['infrastructure_failures']:raise ValueError('incomplete_baseline_stop_no_retry')
    candidate,new=run_stage(cases,args.output/'test-v7','binding_scoped_v7',OpenAIChatModel(args.api_base,health['model_id']),context,knowledge,protocol['source_sha256'])
    if not candidate['complete'] or candidate['infrastructure_failures']:raise ValueError('incomplete_candidate_stop_no_retry')
    old={r['case_id']:r for r in old};new={r['case_id']:r for r in new}
    wins=[name for name in old if not old[name]['passed'] and new[name]['passed']]
    losses=[name for name in old if old[name]['passed'] and not new[name]['passed']]
    negative_failures=[r['case_id'] for r in new.values() if r['category']=='denial_query' and not r['passed']]
    passed=(candidate['case_pass_rate']>=.90 and not negative_failures and not any(candidate['ledger_audit'].values())
            and not candidate['model_output_failures'] and not candidate['infrastructure_failures'])
    summary=dict(evaluation='once_only_paired_heldout_synthetic_test',baseline=baseline,candidate=candidate,
        paired_improvements=wins,paired_regressions=losses,denial_query_failures=negative_failures,
        paired_pass_rate_delta=candidate['case_pass_rate']-baseline['case_pass_rate'],
        descriptive_wilson_95=wilson(candidate['passed_cases'],candidate['evaluated_cases']),
        predefined_criteria_met=passed,no_training=True,weights_unchanged=True,test_used_for_selection=False,
        test_is_now_consumed=True,production_promoted=False,
        limitations=['Same project agent authors synthetic cases; not independent humans or real customer traffic.',
                     'Scenario semantics intentionally overlap; text audit does not prove semantic or pretraining independence.',
                     'v7 adds deterministic intent context, bounded planning support and extractive demo policy answers.',
                     'Any improvement is an end-to-end system gain, not a learned SFT or GRPO gain.',
                     'Descriptive intervals for hand-authored, non-IID cases do not estimate production accuracy.'])
    write(args.output/'summary.json',summary)
    write(args.output/'completed.json',dict(complete=True,once_per_profile=True,summary_sha256=sha(args.output/'summary.json'),
        predefined_criteria_met=passed,production_promoted=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
