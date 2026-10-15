"""Select v8 on contrastive dev and explicitly consumed test regressions."""
from pathlib import Path
import argparse
import json
import shutil
import tarfile
from urllib.request import urlopen

from evaluate_scoped_workflow import ROOT,sha,write,read_rows,source_hashes,run_stage
from ecommerce_customer_service.serving.context_builder import ContextBuilder,load_token_counter
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.knowledge import augment_policy
from ecommerce_customer_service.serving.policy import get_policy
from ecommerce_customer_service.serving.model import OpenAIChatModel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-baseline',type=Path);args=p.parse_args()
    manifest=json.loads((args.cases_dir/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['sha256'].items():
        if sha(args.cases_dir/name)!=digest:raise ValueError('development_cases_changed')
    with urlopen('http://127.0.0.1:8647/healthz',timeout=10) as stream:health=json.load(stream)
    receipt=json.loads((ROOT/'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    if not health['ready'] or health['adapter_sha256']!=receipt['selected_identity']['adapter_sha256'] or health['precision']!='bf16':raise ValueError('model_changed')
    context=ContextBuilder(counter=load_token_counter(receipt['selected_adapter_dir']),max_input_tokens=8192,recent_turns=6)
    knowledge=KnowledgeIndex(Path(receipt['knowledge_file']))
    args.output.mkdir(parents=True,exist_ok=False);hashes=source_hashes()
    for name in ('evaluate_request_workflow.py','build_request_development.py'):
        hashes['scripts/'+name]=sha(ROOT/'scripts'/name)
    criteria=dict(new_development_minimum=.95,no_paired_dev_regressions=True,denial_query_failures=0,
        old_development_passed=40,consumed_first_passed=60,consumed_second_minimum=.95,
        no_previous_second_test_passed_regressions=True,prior_business_passed=57,prior_knowledge_passed=6,
        ledger_errors=0,output_errors=0,infrastructure_errors=0)
    write(args.output/'protocol.json',dict(source_sha256=hashes,health=health,
        cases_manifest_sha256=sha(args.cases_dir/'manifest.json'),knowledge_sha256=knowledge.sha256,
        criteria=criteria,baseline='binding_scoped_v7',candidate='binding_state_v8',
        no_training=True,no_new_test_authored=True,persistent_request_frames=True))
    with tarfile.open(args.output/'frozen-source.tar.gz','w:gz') as archive:
        for name in hashes:archive.add(ROOT/name,arcname=name)
    stages={};details={}
    if args.reuse_baseline:
        old=json.loads((args.reuse_baseline/'protocol.json').read_text(encoding='utf-8'))
        baseline=args.reuse_baseline/'new-dev-v7'
        summary=json.loads((baseline/'summary.json').read_text(encoding='utf-8'))
        if old['cases_manifest_sha256']!=sha(args.cases_dir/'manifest.json') or not summary['complete'] or summary['infrastructure_failures']:
            raise ValueError('baseline_cases_or_completion_changed')
        if any(old['health'][k]!=health[k] for k in ('adapter_sha256','base_revision','model_id','precision','tokenizer_sha256')):
            raise ValueError('baseline_model_changed')
        with (baseline/'model-requests.jsonl').open(encoding='utf-8') as stream:first=json.loads(next(stream))
        policy=augment_policy(get_policy('binding_scoped_v7'))
        if first['messages'][0]['content']!=policy.prompt or first['tools']!=policy.tools:raise ValueError('baseline_prompt_tools_changed')
        shutil.copytree(baseline,args.output/'new-dev-v7')
        stages['new-dev-v7']=summary
        details['new-dev-v7']=[json.loads(p.read_text(encoding='utf-8')) for p in (baseline/'cases').glob('*/result.json')]
        write(args.output/'baseline-reuse.json',dict(source=str(args.reuse_baseline),
            source_protocol_sha256=sha(args.reuse_baseline/'protocol.json'),requests_sha256=sha(baseline/'model-requests.jsonl'),
            identical_case_manifest_model_prompt_tools=True,v7_branch_unchanged_by_v8_only_fix=True,baseline_rerun=False))
    for name,casefile,profile in [('new-dev-v7','new-development','binding_scoped_v7'),
        ('prior-knowledge-v8','prior-knowledge','binding_state_v8'),('old-dev-v8','old-development','binding_state_v8'),
        ('new-dev-v8','new-development','binding_state_v8'),('consumed-second-v8','consumed-second','binding_state_v8'),
        ('consumed-first-v8','consumed-first','binding_state_v8'),('prior-business-v8','prior-business','binding_state_v8')]:
        if name in stages:continue
        stages[name],details[name]=run_stage(read_rows(args.cases_dir/(casefile+'.jsonl')),args.output/name,profile,
            OpenAIChatModel('http://127.0.0.1:8647/v1',health['model_id']),context,knowledge,hashes)
        if stages[name]['infrastructure_failures']:raise ValueError('inference_failed_stop')
        if profile=='binding_state_v8':
            minimum={'prior-knowledge-v8':6,'old-dev-v8':40,'new-dev-v8':38,'consumed-second-v8':57,
                     'consumed-first-v8':60,'prior-business-v8':57}[name]
            if stages[name]['passed_cases']<minimum or any(stages[name]['ledger_audit'].values()):
                write(args.output/'selection.json',dict(gate_passed=False,profile=profile,complete=False,
                    stopped_after_stage=name,stages=stages,no_training=True,new_test_performed=False,
                    reason='development_stage_below_predeclared_gate',production_promoted=False))
                return 2
    baseline={r['case_id']:r for r in details['new-dev-v7']}
    regressions=[r['case_id'] for r in details['new-dev-v8'] if baseline[r['case_id']]['passed'] and not r['passed']]
    previous={r['case_id']:r['candidate']['passed'] for r in read_rows(ROOT/'data/consumed-scoped-heldout-20261015-r1/paired-results.jsonl')}
    old_regressions=[r['case_id'] for r in details['consumed-second-v8'] if previous[r['case_id']] and not r['passed']]
    negatives=[r['case_id'] for name in ('new-dev-v8','consumed-second-v8') for r in details[name]
               if r['category']=='denial_query' and not r['passed']]
    gate=(not regressions and not old_regressions and not negatives and
        all(s['complete'] and not s['model_output_failures'] and not s['infrastructure_failures'] and not any(s['ledger_audit'].values()) for s in stages.values()) and
        stages['new-dev-v8']['case_pass_rate']>=.95 and stages['old-dev-v8']['passed_cases']==40 and
        stages['consumed-first-v8']['passed_cases']==60 and stages['consumed-second-v8']['case_pass_rate']>=.95 and
        stages['prior-business-v8']['passed_cases']==57 and stages['prior-knowledge-v8']['passed_cases']==6)
    selection=dict(gate_passed=gate,profile='binding_state_v8',regressions=regressions,
        consumed_second_regressions=old_regressions,denial_query_failures=negatives,stages=stages,
        no_training=True,new_test_performed=False,production_promoted=False)
    write(args.output/'selection.json',selection);print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)
    return 0 if gate else 2


if __name__=='__main__':raise SystemExit(main())
