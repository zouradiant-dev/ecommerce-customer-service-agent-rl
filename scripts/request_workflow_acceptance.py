"""Prepare and run a once-only v7/v8 synthetic test after dev selection."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import shutil
import tarfile
from urllib.request import urlopen

from evaluate_scoped_workflow import ROOT,sha,write,read_rows,assert_frozen,run_stage
from run_heldout_system_test import preflight,wilson
from review_scoped_development import validate_gold
from prepare_heldout_system_test import audit
from finalize_heldout_system_test import actual_input_audit
from ecommerce_customer_service.serving.context_builder import ContextBuilder,load_token_counter
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.model import OpenAIChatModel


def health_check(protocol):
    with urlopen('http://127.0.0.1:8647/healthz',timeout=10) as stream:health=json.load(stream)
    if not health['ready'] or any(health[k]!=protocol['health'][k] for k in
        ('model_id','adapter_sha256','base_revision','precision','tokenizer_sha256')):raise ValueError('model_identity_changed')
    return health


def prepare(args):
    selection=json.loads((args.development/'selection.json').read_text(encoding='utf-8'))
    selected=json.loads((args.development/'protocol.json').read_text(encoding='utf-8'))
    if not selection['gate_passed'] or selection['profile']!='binding_state_v8':raise ValueError('selected_v8_development_required')
    assert_frozen(selected['source_sha256']);health=health_check(selected)
    args.cases_dir.mkdir(parents=True,exist_ok=False)
    source=dict(selected['source_sha256'])
    for name in ('request_workflow_acceptance.py','build_request_holdout.py','run_request_demo.py','prepare_heldout_system_test.py',
                 'review_scoped_development.py','finalize_heldout_system_test.py'):
        source['scripts/'+name]=sha(ROOT/'scripts'/name)
    protocol=dict(version='request_workflow_paired_test_v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=source,health=health,knowledge_sha256=selected['knowledge_sha256'],
        development_selection_sha256=sha(args.development/'selection.json'),
        baseline='binding_scoped_v7',candidate='binding_state_v8',temperature=0,max_tokens=512,
        max_input_tokens=8192,recent_turns=6,trials=1,
        criteria=dict(candidate_minimum_pass_rate=.90,denial_query_failures=0,ledger_errors=0,output_errors=0,infrastructure_errors=0),
        no_training=True,same_weights=True,independent_case_author=False,real_customer_data=False,
        test_policy='One run per profile. No runtime changes, replacement cases or retesting failures. This test becomes consumed after inspection.')
    write(args.cases_dir/'protocol.json',protocol)
    rows=read_rows(args.cases_input);preflight(rows);validate_gold(rows)
    extras=[* (ROOT/'outputs').glob('request-development-cases-20261015-r*/*.jsonl'),
            * (ROOT/'outputs').glob('scoped-development-cases-20261015-r*/*.jsonl')]
    overlap=audit(rows,extra_paths=extras);write(args.cases_dir/'overlap-audit.json',overlap)
    if overlap['exact_first_turn_collisions'] or overlap['exact_user_turn_collisions']:raise ValueError('test_text_reused_from_available_history')
    shutil.copy2(args.cases_input,args.cases_dir/'cases.jsonl')
    write(args.cases_dir/'manifest.json',dict(cases=60,frozen_before_model_calls=True,
        author_independent=False,sha256={name:sha(args.cases_dir/name) for name in ('cases.jsonl','protocol.json','overlap-audit.json')}))
    with tarfile.open(args.cases_dir/'frozen-source.tar.gz','w:gz') as archive:
        for name in source:archive.add(ROOT/name,arcname=name)
    print('60 cases frozen; no model request was made.',flush=True)


def run(args):
    protocol=json.loads((args.cases_dir/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((args.cases_dir/'manifest.json').read_text(encoding='utf-8'))
    selection=json.loads((args.development/'selection.json').read_text(encoding='utf-8'))
    if not selection['gate_passed'] or sha(args.development/'selection.json')!=protocol['development_selection_sha256']:
        raise ValueError('development_selection_changed')
    for name,digest in manifest['sha256'].items():
        if sha(args.cases_dir/name)!=digest:raise ValueError('frozen_test_changed')
    assert_frozen(protocol['source_sha256']);health=health_check(protocol)
    rows=read_rows(args.cases_dir/'cases.jsonl');preflight(rows);validate_gold(rows)
    receipt=json.loads((ROOT/'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    context=ContextBuilder(counter=load_token_counter(receipt['selected_adapter_dir']),max_input_tokens=8192,recent_turns=6)
    knowledge=KnowledgeIndex(Path(receipt['knowledge_file']))
    if knowledge.sha256!=protocol['knowledge_sha256']:raise ValueError('knowledge_changed')
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.cases_dir/'execution-claim.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(output=str(args.output.resolve()),manifest_sha256=sha(args.cases_dir/'manifest.json'),one_run_per_profile=True),stream)
    write(args.output/'protocol.json',protocol)
    summaries={};details={}
    for name,profile in [('test-v7',protocol['baseline']),('test-v8',protocol['candidate'])]:
        summaries[name],details[name]=run_stage(rows,args.output/name,profile,
            OpenAIChatModel('http://127.0.0.1:8647/v1',health['model_id']),context,knowledge,protocol['source_sha256'])
        if not summaries[name]['complete'] or summaries[name]['infrastructure_failures']:raise ValueError('inference_failed_no_retry')
    old={r['case_id']:r for r in details['test-v7']};new={r['case_id']:r for r in details['test-v8']}
    candidate=summaries['test-v8']
    improvements=[name for name in old if not old[name]['passed'] and new[name]['passed']]
    regressions=[name for name in old if old[name]['passed'] and not new[name]['passed']]
    negative=[r['case_id'] for r in new.values() if r['category']=='denial_query' and not r['passed']]
    passed=(candidate['case_pass_rate']>=.90 and not negative and not any(candidate['ledger_audit'].values()) and
            not candidate['model_output_failures'] and not candidate['infrastructure_failures'])
    summary=dict(evaluation='third_frozen_synthetic_test_once_per_profile',baseline=summaries['test-v7'],candidate=candidate,
        paired_improvements=improvements,paired_regressions=regressions,denial_query_failures=negative,
        predefined_criteria_met=passed,descriptive_wilson_95=wilson(candidate['passed_cases'],60),
        no_training=True,weights_unchanged=True,test_consumed=True,production_promoted=False,
        limitations=['Same project agent authors synthetic cases; not independent humans or production traffic.',
          'Static user turns and explicit button events do not measure adaptive real users.',
          'One greedy trial; random operation IDs and timestamps can vary later model inputs.',
          'Exact text/input audits do not prove semantic or pretraining independence.',
          'Gains are system engineering, not learned SFT/GRPO gains.'])
    write(args.output/'summary.json',summary)
    write(args.output/'completed.json',dict(complete=True,summary_sha256=sha(args.output/'summary.json'),once_per_profile=True))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def finalize(args):
    complete=json.loads((args.output/'completed.json').read_text(encoding='utf-8'))
    summary=json.loads((args.output/'summary.json').read_text(encoding='utf-8'))
    protocol=json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))
    if not complete['complete'] or complete['summary_sha256']!=sha(args.output/'summary.json'):raise ValueError('completed_test_required')
    assert_frozen(protocol['source_sha256'])
    rows=read_rows(args.cases_dir/'cases.jsonl');results={}
    for name in ('test-v7','test-v8'):
        result=actual_input_audit(args.cases_dir,args.output/name)
        if result['cross_split_exact_canonical_input_collisions']:raise ValueError('actual_input_collision')
        write(args.output/name/'actual-input-audit.json',result)
        results[name]={p.parent.name:json.loads(p.read_text(encoding='utf-8')) for p in (args.output/name/'cases').glob('*/result.json')}
    args.consumed_output.mkdir(parents=True,exist_ok=False)
    for filename in ('cases.jsonl','protocol.json','manifest.json','overlap-audit.json'):
        shutil.copy2(args.cases_dir/filename,args.consumed_output/filename)
    shutil.copy2(args.output/'summary.json',args.consumed_output/'summary.json')
    for name in ('test-v7','test-v8'):
        for filename in ('bad_cases.jsonl','actual-input-audit.json'):
            shutil.copy2(args.output/name/filename,args.consumed_output/(name+'-'+filename))
    paired=[dict(case_id=r['case_id'],category=r['category'],baseline=results['test-v7'][r['case_id']],candidate=results['test-v8'][r['case_id']]) for r in rows]
    (args.consumed_output/'paired-results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in paired),encoding='utf-8')
    report=['# 第三套冻结合成 Test：v7 / v8', '',
        f"同题配对：v7 {summary['baseline']['passed_cases']}/60 → v8 {summary['candidate']['passed_cases']}/60。",
        f"预先条件通过：{summary['predefined_criteria_met']}；改善 {len(summary['paired_improvements'])}，回退 {len(summary['paired_regressions'])}。",'',
        '没有训练。先选开发版本，再另行编写冻结题目，每版各一次。合成题作者了解项目，不能作为真实流量泛化认证。', '',
        '| ID | 切片 | 首轮发言 | v7 | v8 | v8失败 |','|---|---|---|---|---|---|']
    for case,row in zip(rows,paired):
        first=next(s['text'] for s in case['steps'] if s['kind']=='message')
        report.append(f"| {case['case_id']} | {case['category']} | {first.replace('|','/')} | {row['baseline']['passed']} | {row['candidate']['passed']} | {'; '.join(row['candidate']['failures'])} |")
    (args.output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (args.consumed_output/'README.md').write_text('\n'.join(report[:7])+'\n\n这组Test已消费，后续只能用于回归；不得在调参后继续称为独立验收。完整原始请求、业务账本和源码快照存于本机校验归档。\n',encoding='utf-8')
    write(args.consumed_output/'review-receipt.json',dict(source_summary_sha256=sha(args.output/'summary.json'),
        test_consumed=True,model_calls_replayed=0,source_unchanged_since_freeze=True,
        files_sha256={p.name:sha(p) for p in args.consumed_output.iterdir() if p.is_file()}))
    print('Inputs audited and all paired results retained. No inference was replayed.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','finalize'])
    p.add_argument('--development',type=Path,required=True);p.add_argument('--cases-dir',type=Path,required=True)
    p.add_argument('--cases-input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--consumed-output',type=Path)
    args=p.parse_args();globals()[args.mode](args)


if __name__=='__main__':main()
