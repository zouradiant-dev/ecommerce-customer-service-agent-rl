"""Describe real reward variation; never turn this report into a training approval."""
from collections import defaultdict
import math
from statistics import mean,pstdev


def describe(rows):
    if not rows:raise ValueError('reward_signal_requires_rollouts')
    groups=defaultdict(list);seen=set()
    for row in rows:
        if row.get('split')!='train' or row.get('stage')!='sft' or row.get('learned_model_result') is not True:
            raise ValueError('reward_signal_requires_real_train_only_sft_rollouts')
        if type(row.get('trial')) is not int or row['trial'] < 0:
            raise ValueError('invalid_trial')
        key=(row['task_id'],row['trial'])
        if key in seen:raise ValueError('duplicate_task_trial')
        seen.add(key)
        components=row['reward_components']
        for name in ('strict_binary_v18','train_shaped_v18'):
            value=components[name]
            if type(value) not in {float,int} or not math.isfinite(value) or not -1<=value<=1:
                raise ValueError('invalid_reward_value')
        if components['strict_binary_v18'] not in {0,1}:raise ValueError('invalid_binary_reward')
        groups[row['task_id']].append(row)
    if min(map(len, groups.values())) < 2 or len({len(group) for group in groups.values()}) != 1:
        raise ValueError('reward_signal_requires_equal_groups_with_at_least_two_trials')
    stats={}
    for mode in ('strict_binary_v18','train_shaped_v18'):
        mixed=[task for task,group in groups.items() if len(group)>=2 and
               len({row['reward_components'][mode] for row in group})>1]
        stats[mode]={'mean':mean(row['reward_components'][mode] for row in rows),
            'mixed_groups':len(mixed),'mixed_group_rate':len(mixed)/len(groups),'mixed_task_ids':sorted(mixed),
            'per_task_std':{task:pstdev(row['reward_components'][mode] for row in group)
                            for task,group in sorted(groups.items())}}
    unsafe_positive=[row.get('rollout_id',row['task_id']) for row in rows
        if row['verification'].get('hard_safety_failures') and row['reward_components']['train_shaped_v18']>0]
    return {'schema_version':'v18_reward_signal_diagnostic_v1','tasks':len(groups),'rollouts':len(rows),
            'rewards':stats,'unsafe_positive_ids':unsafe_positive,
            'human_review_complete':False,'reward_calibrated':False,'formal_rl_gate_passed':False,
            'policy_updates':0,'interpretation':'奖励分布诊断；组内差异不等于奖励正确，须复核正负例。'}
