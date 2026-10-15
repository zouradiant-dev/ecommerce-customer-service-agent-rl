"""Correct contradictory consumed-dev gold using saved trajectories, no inference.

Only final pending-count gold that contradicts the final explicit checkpoint may
change. User inputs, steps, orders, model responses, and raw scores stay intact.
"""
from pathlib import Path
from collections import Counter
import argparse
import json

from evaluate_scoped_workflow import sha,write,read_rows,assert_frozen


def validate_gold(rows):
    for row in rows:
        last=row['steps'][-1]
        if last['kind']=='checkpoint' and 'pending_count' in last['assertions']:
            if row['expected'].get('pending_count')!=last['assertions']['pending_count']:
                raise ValueError('terminal_gold_contradicts_last_checkpoint:'+row['case_id'])


def review_result(case,directory):
    raw=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    history=json.loads((directory/'history.json').read_text(encoding='utf-8'))
    spans=json.loads((directory/'spans.json').read_text(encoding='utf-8'))
    end=max(s['started_at']+s['duration_ms']/1000 for s in spans if s['duration_ms'] is not None)
    pending=sum(op['status']=='pending_confirmation' and op['expires_at']>end for op in history['operations'])
    events=json.loads((directory/'events.json').read_text(encoding='utf-8'))
    if events[-1]['kind']!='checkpoint' or events[-1].get('result',{}).get('pending_count')!=pending:
        raise ValueError('saved_checkpoint_and_final_history_disagree')
    expected=case['expected']['pending_count']
    failures=[f for f in raw['failures'] if f!='pending_operation_count']
    if pending!=expected:failures.append('pending_operation_count')
    return {**raw,'passed':not failures,'failures':failures,'reviewed_terminal_pending_count':pending,
            'raw_result_sha256':sha(directory/'result.json'),'model_inference_replayed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--cases-dir',type=Path,required=True)
    p.add_argument('--corrected-cases-output',type=Path,required=True);args=p.parse_args()
    selection=json.loads((args.development/'selection.json').read_text(encoding='utf-8'))
    protocol=json.loads((args.development/'protocol.json').read_text(encoding='utf-8'))
    assert_frozen(protocol['source_sha256'])
    if any(not s['complete'] for s in selection['stages'].values()):raise ValueError('complete_development_required_for_review')
    args.corrected_cases_output.mkdir(parents=True,exist_ok=False)
    changes=[];corrected=[]
    for path in args.cases_dir.glob('*.jsonl'):
        rows=read_rows(path)
        for row in rows:
            last=row['steps'][-1]
            if last['kind']=='checkpoint' and 'pending_count' in last['assertions'] and row['expected'].get('pending_count')!=last['assertions']['pending_count']:
                if path.name!='consumed-regression.jsonl' or row['split']!='dev':raise ValueError('unexpected_gold_revision_scope')
                changes.append(dict(case_id=row['case_id'],old=row['expected']['pending_count'],new=last['assertions']['pending_count'],
                                    reason='terminal_must_match_final_explicit_checkpoint'))
                row['expected']['pending_count']=last['assertions']['pending_count'];corrected.append(row)
        validate_gold(rows)
        (args.corrected_cases_output/path.name).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    if {r['case_id'] for r in changes}!={'heldout_030','heldout_033'}:raise ValueError('unexpected_gold_changes_stop')
    reviewed=[]
    for case in corrected:
        directory=args.development/'consumed-regression-v7/cases'/case['case_id']
        row=review_result(case,directory);write(directory/'reviewed-result.json',row);reviewed.append(row)
    raw_rows={p.parent.name:json.loads(p.read_text(encoding='utf-8')) for p in (args.development/'consumed-regression-v7/cases').glob('*/result.json')}
    raw_rows.update({row['case_id']:row for row in reviewed})
    summary=dict(selection['stages']['consumed-regression-v7'])
    summary['passed_cases']=sum(r['passed'] for r in raw_rows.values());summary['case_pass_rate']=summary['passed_cases']/60
    summary['by_category']={c:dict(passed=sum(r['passed'] for r in raw_rows.values() if r['category']==c),total=sum(r['category']==c for r in raw_rows.values())) for c in sorted({r['category'] for r in raw_rows.values()})}
    summary['failures']=dict(Counter(f.split(':')[0] for r in raw_rows.values() for f in r['failures']))
    summary['reviewed_gold_only']=True;write(args.development/'consumed-regression-v7/reviewed-summary.json',summary)
    reviewed_selection=json.loads(json.dumps(selection));reviewed_selection['stages']['consumed-regression-v7']=summary
    stages=reviewed_selection['stages']
    gate=(not selection['regressions'] and not selection['consumed_regressions'] and not selection['denial_query_failures']
          and all(s['complete'] and not s['infrastructure_failures'] and not any(s['ledger_audit'].values()) for s in stages.values())
          and stages['new-dev-v7']['case_pass_rate']>=.90 and summary['passed_cases']==60
          and stages['prior-business-v7']['passed_cases']==57 and stages['prior-knowledge-v7']['passed_cases']>=5)
    reviewed_selection.update(gate_passed=gate,raw_selection_sha256=sha(args.development/'selection.json'),
        consumed_gold_revision=changes,no_model_or_runtime_changes_from_review=True,model_calls_replayed=0,
        historical_52_of_60_score_unchanged=True,corrected_cases_directory=str(args.corrected_cases_output))
    write(args.development/'reviewed-selection.json',reviewed_selection)
    write(args.corrected_cases_output/'gold-review-receipt.json',dict(changes=changes,no_new_inference=True,raw_results_overwritten=False,
        original_case_manifest_sha256=sha(args.cases_dir/'manifest.json'),reviewed_selection_sha256=sha(args.development/'reviewed-selection.json')))
    print(json.dumps(dict(gate_passed=gate,reviewed_consumed_passed=summary['passed_cases'],changes=changes,model_calls_replayed=0),indent=2))


if __name__=='__main__':main()
