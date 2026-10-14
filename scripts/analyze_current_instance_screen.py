"""CPU audit of the downloaded dev screen; does not generate or train a model."""
from pathlib import Path
from hashlib import sha256
from dataclasses import replace
import argparse
from collections import Counter
import json
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.evaluation_v18 import select_v18_dev_tasks, compare_v18_stages, audit_tool_call_format
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.tools_v18 import CustomerServiceToolsV18
from ecommerce_customer_service.verifier_v18 import verify_task_v18,score_v18


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-dir',type=Path,required=True)
    args=p.parse_args()
    folder=args.download_dir/'extracted/outputs/nine-tool-screen'
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    report=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    selected,plan=select_v18_dev_tasks(ROOT/'data/v18-train-dev-r1/tasks.jsonl',max_tasks_per_scenario=2)
    if plan['manifest_sha256']!=protocol['task_plan']['manifest_sha256']: raise ValueError('manifest_changed')
    tasks={TaskSpec.from_file(path).task_id:path for path in selected}
    for name,digest in protocol['source_sha256'].items():
        path=ROOT/('scripts/' if name=='evaluate_sft_v18.py' else 'src/ecommerce_customer_service/')/name
        if sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError('screen_verifier_source_changed:'+name)
    all_rows={}
    replay_failures=[]
    for stage in ('original','online'):
        rows=[json.loads(line) for line in (folder/stage/'rollouts.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(rows)!=60 or {row['task_id'] for row in rows}!=set(tasks): raise ValueError('unpaired_screen_tasks')
        all_rows[stage]={row['task_id']:row for row in rows}
        for row in rows:
            try:
                with tempfile.TemporaryDirectory(prefix='v18-dev-screen-replay-') as temporary:
                    original=TaskSpec.from_file(tasks[row['task_id']])
                    database=Path(temporary)/'state.sqlite'
                    shutil.copy2(original.database_path,database)
                    task=replace(original,database_path=database)
                    tools=CustomerServiceToolsV18(database,task)
                    tools.observe_user_message(task.user_prompt)
                    for call in row['tool_executions']:
                        if tools.execute(call['name'],call['arguments']).to_dict()!=call:
                            raise ValueError('tool_output_mismatch:'+call['name'])
                    calls=[call.to_dict() for call in tools.executions]
                    verification=verify_task_v18(task,row['final_response'],calls)
                    reward=score_v18(task,verification,calls,stopped_reason=row['stopped_reason'])
                    if (json.loads(json.dumps(verification.to_dict()))!=row['verification']
                            or json.loads(json.dumps(reward.to_dict()))!=row['reward_components']
                            or reward.strict_binary_v18!=row['reward'] or audit_tool_call_format(row['messages'])!=row['tool_format']):
                        raise ValueError('state_reward_or_format_mismatch')
            except Exception as error:
                replay_failures.append({'stage':stage,'task_id':row['task_id'],'error':str(error)})
    comparison=compare_v18_stages(all_rows['original'].values(),all_rows['online'].values())
    if comparison!=report['paired_comparisons_to_baseline']['online']: raise ValueError('reported_comparison_mismatch')
    changed=[]
    for task_id in sorted(tasks):
        before,after=all_rows['original'][task_id],all_rows['online'][task_id]
        if (before['reward']>=1)!=(after['reward']>=1):
            changed.append({'task_id':task_id,'scenario':before['scenario'],
                'change':'improved' if after['reward']>=1 else 'regressed',
                'original_failures':before['verification']['failures'],'online_failures':after['verification']['failures'],
                'original_final':before['final_response'],'online_final':after['final_response'],
                'original_tools':[{'name':c['name'],'ok':c['ok']} for c in before['tool_executions']],
                'online_tools':[{'name':c['name'],'ok':c['ok']} for c in after['tool_executions']]})
    overall={label:{key:report['stage_summaries'][label]['overall'][key] for key in (
        'strict_passes','rollouts','strict_pass_rate','hard_safety_failure_rollouts','structured_terminal_rate',
        'tool_format_stable_rate','successful_write_counts','worker_errors')} for label in ('original','online')}
    common_failures=[name for name in tasks if all_rows['original'][name]['reward']<1 and all_rows['online'][name]['reward']<1]
    result={'schema_version':'current_instance_dev_screen_audit_v1','cpu_replayed':120,'replay_failures':replay_failures,
        'comparison':comparison,'overall':overall,'changed_cases':changed,'both_failed_cases':sorted(common_failures),
        'limited_dev_screen':True,'formal_test_performed':False,'new_model_training_performed':False,
        'decision':'retain_original_nine_tool_adapter_and_continue_bounded_online_only_repair',
        'limitations':['two_dev_tasks_per_scenario_not_full_600','single_greedy_trial_nf4',
            'communication_checks_include_phrase_rules','hard_safety_flags_include_false_write_claims_and_blocked_attempts',
            'net_minus_two_not_statistical_proof_of_general_forgetting','native_nine_tools_differ_from_online_five_tool_runtime']}
    (args.download_dir/'analysis-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'overall':overall,'comparison':comparison,'cpu_replayed':120,'replay_failure_count':len(replay_failures),
                      'common_failures':len(common_failures)},ensure_ascii=False,indent=2))
    return 0 if not replay_failures else 2


if __name__=='__main__': raise SystemExit(main())
