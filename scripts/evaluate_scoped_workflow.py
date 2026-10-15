"""Evaluate opt-in v7 on new development and explicitly consumed regressions."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.context_builder import ContextBuilder,load_token_counter
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.knowledge import augment_policy
from ecommerce_customer_service.serving.policy import get_policy
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.online_evaluation import run_case
from run_heldout_system_test import RecordingModel,ledger_audit


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read_rows(p): return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]


def source_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in [*(ROOT/'src/ecommerce_customer_service/serving').glob('*.py'),
        ROOT/'src/ecommerce_customer_service/training.py',ROOT/'scripts/evaluate_scoped_workflow.py',ROOT/'scripts/run_heldout_system_test.py']}


def assert_frozen(hashes):
    for name,digest in hashes.items():
        if sha(ROOT/name)!=digest: raise ValueError('frozen_source_changed:'+name)


def run_stage(rows,output,profile,model,context,knowledge,hashes):
    output.mkdir(parents=True,exist_ok=False)
    results=[]
    with (output/'model-requests.jsonl').open('x',encoding='utf-8') as stream:
        recorded=RecordingModel(model,stream)
        for index,case in enumerate(rows):
            assert_frozen(hashes);recorded.case_id=case['case_id']
            print(json.dumps(dict(event='SCOPED_CASE_START',stage=output.name,index=index+1,total=len(rows),case_id=case['case_id'])),flush=True)
            directory=(output/'cases'/case['case_id']).resolve()
            result=run_case(case,directory,recorded,context,policy_profile=profile,knowledge=knowledge)
            spans=json.loads((directory/'spans.json').read_text(encoding='utf-8'))
            result['controller_actions']=sum(s['kind']=='controller' for s in spans)
            result['controller_preview_rebuilds']=sum(s['kind']=='controller' and s['name']=='preview_rebuild_preview_cancel' for s in spans)
            result['controller_protocol_replans']=sum(s['kind']=='controller' and s['name']=='terminal_protocol_replan' for s in spans)
            result['controller_policy_loop_cutoffs']=sum(s['kind']=='controller' and s['name']=='policy_tool_loop_cutoff' for s in spans)
            ledger=ledger_audit(case,directory);write(directory/'ledger-audit.json',ledger)
            result['ledger_audit']={k:v for k,v in ledger.items() if k not in {'cancellations','refunds'}}
            for name,value in result['ledger_audit'].items():
                if value: result['failures'].append('ledger:'+name)
            result['passed']=not result['failures'];write(directory/'result.json',result);results.append(result)
            print(json.dumps(dict(event='SCOPED_CASE_DONE',stage=output.name,completed=index+1,total=len(rows),passed=result['passed'],passes=sum(r['passed'] for r in results))),flush=True)
            if result['model_infrastructure_failures']:break
    assert_frozen(hashes)
    summary=dict(profile=profile,expected_cases=len(rows),evaluated_cases=len(results),passed_cases=sum(r['passed'] for r in results),
        complete=len(rows)==len(results),case_pass_rate=sum(r['passed'] for r in results)/len(rows),
        model_output_failures=sum(r['model_output_failures'] for r in results),
        infrastructure_failures=sum(r['model_infrastructure_failures'] for r in results),
        model_calls=sum(r['model_calls'] for r in results),tool_calls=sum(r['tool_calls'] for r in results),
        controller_actions=sum(r['controller_actions'] for r in results),
        controller_preview_rebuilds=sum(r['controller_preview_rebuilds'] for r in results),
        controller_protocol_replans=sum(r['controller_protocol_replans'] for r in results),
        controller_policy_loop_cutoffs=sum(r['controller_policy_loop_cutoffs'] for r in results),
        ledger_audit={name:sum(r['ledger_audit'][name] for r in results) for name in ('unauthorized_writes','wrong_order_writes','duplicate_writes','invalid_refunds')},
        by_category={category:dict(passed=sum(r['passed'] for r in results if r['category']==category),total=sum(r['category']==category for r in results)) for category in sorted({r['category'] for r in results})},
        failures=dict(Counter(f.split(':')[0] for r in results for f in r['failures'])),
        no_training=True,synthetic_business=True,production_promoted=False)
    write(output/'summary.json',summary)
    (output/'bad_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results if not r['passed']),encoding='utf-8')
    return summary,results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--api-base',default='http://127.0.0.1:8647/v1')
    p.add_argument('--reuse-baseline',type=Path,help='Reuse complete v6 trajectories only with identical case manifest, model, prompt and tools.')
    p.add_argument('--reuse-baseline-cases-dir',type=Path)
    args=p.parse_args()
    manifest=json.loads((args.cases_dir/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['sha256'].items():
        if sha(args.cases_dir/name)!=digest:raise ValueError('case_manifest_changed')
    with urlopen(args.api_base.removesuffix('/v1')+'/healthz',timeout=10) as stream:health=json.load(stream)
    receipt=json.loads((ROOT/'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    if not health['ready'] or health['adapter_sha256']!=receipt['selected_identity']['adapter_sha256'] or health['precision']!='bf16':raise ValueError('changed_model')
    args.output.mkdir(parents=True,exist_ok=False);source=source_hashes()
    write(args.output/'protocol.json',dict(source_sha256=source,health=health,cases_manifest_sha256=sha(args.cases_dir/'manifest.json'),
        no_training=True,no_new_test_authored=True,old_test_is_consumed_regression=True,
        selected_by='development_and_consumed_regression_only',controller_lookups_and_extractive_policy_answers=True))
    with tarfile.open(args.output/'frozen-source.tar.gz','w:gz') as archive:
        for name in source:archive.add(ROOT/name,arcname=name)
    model=OpenAIChatModel(args.api_base,health['model_id'])
    context=ContextBuilder(counter=load_token_counter(receipt['selected_adapter_dir']),max_input_tokens=8192,recent_turns=6)
    knowledge=KnowledgeIndex(Path(receipt['knowledge_file']))
    suites={};details={}
    if args.reuse_baseline:
        original=json.loads((args.reuse_baseline/'protocol.json').read_text(encoding='utf-8'))
        baseline_cases=args.reuse_baseline_cases_dir or args.cases_dir
        if original['cases_manifest_sha256']!=sha(baseline_cases/'manifest.json'):raise ValueError('baseline_original_case_manifest_changed')
        if sha(baseline_cases/'new-development.jsonl')!=sha(args.cases_dir/'new-development.jsonl'):raise ValueError('baseline_evaluated_cases_changed')
        if any(original['health'][k]!=health[k] for k in ('adapter_sha256','base_revision','model_id','precision')):raise ValueError('baseline_model_changed')
        baseline_dir=args.reuse_baseline/'new-dev-v6'
        baseline=json.loads((baseline_dir/'summary.json').read_text(encoding='utf-8'))
        if not baseline['complete'] or baseline['infrastructure_failures']:raise ValueError('incomplete_baseline_cannot_be_reused')
        with (baseline_dir/'model-requests.jsonl').open(encoding='utf-8') as stream:first=json.loads(next(stream))
        expected=augment_policy(get_policy('binding_catalog_v6'))
        if first['messages'][0]['content']!=expected.prompt or first['tools']!=expected.tools:raise ValueError('baseline_prompt_or_tools_changed')
        shutil.copytree(baseline_dir,args.output/'new-dev-v6')
        suites['new-dev-v6']=baseline
        details['new-dev-v6']=[json.loads(path.read_text(encoding='utf-8')) for path in (baseline_dir/'cases').glob('*/result.json')]
        write(args.output/'baseline-reuse.json',dict(source=str(args.reuse_baseline),original_protocol_sha256=sha(args.reuse_baseline/'protocol.json'),
            model_case_manifest_prompt_tools_identical=True,baseline_rerun=False,source_controller_v6_branch_unchanged=True))
    for name,casefile,profile in [('new-dev-v6','new-development','binding_catalog_v6'),('new-dev-v7','new-development','binding_scoped_v7'),
        ('consumed-regression-v7','consumed-regression','binding_scoped_v7'),('prior-business-v7','prior-business','binding_scoped_v7'),
        ('prior-knowledge-v7','prior-knowledge','binding_scoped_v7')]:
        if name=='new-dev-v6' and args.reuse_baseline:continue
        suites[name],details[name]=run_stage(read_rows(args.cases_dir/(casefile+'.jsonl')),args.output/name,profile,model,context,knowledge,source)
        if suites[name]['infrastructure_failures']:raise ValueError('inference_infrastructure_failure_stop')
    old={r['case_id']:r for r in details['new-dev-v6']};new={r['case_id']:r for r in details['new-dev-v7']}
    regressions=[name for name in old if old[name]['passed'] and not new[name]['passed']]
    old_consumed={r['case_id']:r for r in read_rows(ROOT/'data/consumed-heldout-system-20261015-r1/bad_cases.jsonl')}
    consumed_regressions=[r['case_id'] for r in details['consumed-regression-v7'] if not r['passed'] and r['case_id'] not in old_consumed]
    safety_failures=[r['case_id'] for r in details['new-dev-v7'] if r['category']=='denial_query_contrast' and not r['passed']]
    gate=(not regressions and not consumed_regressions and not safety_failures
          and all(s['complete'] and not s['infrastructure_failures'] and not any(s['ledger_audit'].values()) for s in suites.values())
          and suites['new-dev-v7']['case_pass_rate']>=.90 and suites['consumed-regression-v7']['passed_cases']==60
          and suites['prior-business-v7']['passed_cases']==57 and suites['prior-knowledge-v7']['passed_cases']>=5)
    selection=dict(gate_passed=gate,profile='binding_scoped_v7',regressions=regressions,consumed_regressions=consumed_regressions,
        denial_query_failures=safety_failures,stages=suites,no_training=True,new_test_performed=False,
        production_promoted=False,scope='development_selection_not_independent_test')
    write(args.output/'selection.json',selection);print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)
    return 0 if gate else 2


if __name__=='__main__':raise SystemExit(main())
