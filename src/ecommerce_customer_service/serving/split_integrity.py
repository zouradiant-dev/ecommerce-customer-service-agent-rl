"""Audit actual generation inputs, excluding supervised targets and gold labels."""
from collections import defaultdict
from hashlib import sha256
import json

from ..training import canonicalize_tool_schemas, normalize_messages, chat_template_ids
from .context_builder import ContextBuilder, HFTokenCounter
from .policy import get_policy


def fingerprint(messages, tools, tokenizer):
    ids = chat_template_ids(tokenizer, normalize_messages(messages),
                            tools=canonicalize_tool_schemas(tools), add_generation_prompt=True)
    return sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def audit(rows_by_split, tokenizer, *, sealed_cases=(), policy_profile='binding_compact_v3', knowledge=None):
    index = defaultdict(lambda: defaultdict(list))
    counts = {}
    for split, rows in rows_by_split.items():
        if split not in {'train', 'dev'}: raise ValueError('supervision_audit_accepts_train_dev_only')
        counts[split] = len(rows)
        for row in rows:
            if row.get('split') != split or not row.get('messages') or row['messages'][-1].get('role') != 'assistant':
                raise ValueError('invalid_supervision_split_or_target')
            sha = fingerprint(row['messages'][:-1], row['tools'], tokenizer)
            index[sha][split].append(row.get('id', row.get('task_id')))
    policy = get_policy(policy_profile)
    if knowledge:
        from .knowledge import augment_policy
        policy = augment_policy(policy)
    context = ContextBuilder(counter=HFTokenCounter(tokenizer))
    counts['test_first_turns'] = len(sealed_cases)
    for case in sealed_cases:
        if case.get('split') != 'test': raise ValueError('sealed_case_must_be_test')
        # Read first user input only; never execute test references or supervise test.
        text = next(step['text'] for step in case['steps'] if step['kind'] == 'message')
        built = context.build(policy.prompt, [{'role':'user','content':text}], policy.tools,
                              {'conversation_status':'active','operations':[],'order_observations':[]})
        index[fingerprint(built.messages, policy.tools, tokenizer)]['test'].append(case['case_id'])
    collisions = [{'input_sha256':sha,'splits':dict(splits)} for sha,splits in sorted(index.items()) if len(splits)>1]
    return {'schema_version':'rendered_input_overlap_v1','policy_profile':policy_profile,'passed':not collisions,'counts':counts,
            'cross_split_collisions':collisions,'within_split_repeated_inputs':{
                split:sum(len(refs)-1 for splits in index.values() for name,refs in splits.items() if name==split)
                for split in ('train','dev','test')},
            'scope':'train_dev_all_generation_prefixes_and_test_first_user_turn_only',
            'test_reference_executed':False,'model_evaluation_performed':False,
            'limitations':['shared_scenario_families_remain','test_later_turns_not_audited_without_execution']}
