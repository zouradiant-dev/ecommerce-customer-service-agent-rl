"""Bounded closeout gates. Dev selects weights; sealed test cannot select them."""
from pathlib import Path
from hashlib import sha256
import json
import math
from .artifacts import BASE_REVISION, BASE_MODEL, FILE_HASHES, file_sha256
from .knowledge import KnowledgeIndex, augment_policy
from .policy import get_policy
from .repair_experiment import rows, training_arguments as old_training_arguments
from ..training import validate_training_row

ONLINE_SHA = '3a5dd3e29427d2bf5fc89e813ff16dbd30bdca6339ad2fd4169b6c2b6f024764'
CONFIG = {'schema_version':'current_instance_closeout_v1','seed':20261014,
          'smoke_steps':4,'pilot_steps':64,'max_sequence_length':3072,
          'learning_rate':2e-5,'gradient_accumulation':8,'policy_profile':'binding_guard_v4',
          'training_order':'seeded_shuffle_once',
          'knowledge_enabled_in_all_online_stages':True,'precision':'bf16',
          'selection_rule':'no_case_regression_any_suite_no_more_output_errors_and_at_least_one_improvement',
          'test_uses_selected_adapter_only':True,'automatic_production_promotion':False}
REQUIRED = {'adapter_model.safetensors','adapter_config.json','config.json','tokenizer.json','tokenizer_config.json'}


def verify_online_adapter(adapter):
    adapter=Path(adapter).resolve(strict=True)
    digest=file_sha256(adapter/'adapter_model.safetensors')
    if digest==ONLINE_SHA:
        receipt=json.loads((adapter/'repair-artifact.json').read_text(encoding='utf-8'))
        if receipt.get('source_adapter_sha256')!=FILE_HASHES['adapter_model.safetensors']:
            raise ValueError('online_adapter_lineage_changed')
    else:
        receipt=json.loads((adapter/'current-instance-artifact.json').read_text(encoding='utf-8'))
        if (receipt.get('source_adapter_sha256')!=ONLINE_SHA or receipt.get('training_config')!=CONFIG
                or receipt.get('original_v18_sha256')!=FILE_HASHES['adapter_model.safetensors']):
            raise ValueError('closeout_adapter_lineage_changed')
    if receipt.get('base_revision')!=BASE_REVISION or not REQUIRED<=set(receipt.get('files_sha256',{})):
        raise ValueError('online_adapter_receipt_incomplete')
    for name,expected in receipt['files_sha256'].items():
        path=adapter/name
        if Path(name).name!=name or path.is_symlink() or file_sha256(path)!=expected:
            raise ValueError('online_adapter_file_changed:'+name)
    config=json.loads((adapter/'adapter_config.json').read_text(encoding='utf-8'))
    if (config.get('base_model_name_or_path')!=BASE_MODEL or config.get('r')!=16
            or config.get('lora_alpha')!=32 or config.get('peft_type')!='LORA'
            or config.get('task_type')!='CAUSAL_LM'):
        raise ValueError('online_lora_configuration_changed')
    return {'adapter_sha256':digest,'model_id':'ecommerce-online-'+digest[:12],
            'base_revision':BASE_REVISION,'receipt_sha256':file_sha256(adapter/(
                'repair-artifact.json' if digest==ONLINE_SHA else 'current-instance-artifact.json'))}


def validate_data(root, data, knowledge_file, tokenizer):
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('schema_version')!='current_instance_five_tool_sft_v1'
            or manifest.get('training_ready') is not True or manifest.get('test_sft_rows')!=0
            or manifest.get('test_reference_executed') is not False
            or manifest.get('max_sequence_length')!=CONFIG['max_sequence_length']):
        raise ValueError('invalid_closeout_data_manifest')
    if any(path.name.startswith('test') for path in data.iterdir()):
        raise ValueError('test_must_be_outside_training_bundle')
    knowledge=KnowledgeIndex(knowledge_file)
    if (manifest['knowledge_sha256']!=knowledge.sha256
            or manifest['policy']!=augment_policy(get_policy(CONFIG['policy_profile'])).metadata()):
        raise ValueError('five_tool_training_protocol_changed')
    for mapping,base in ((manifest['files_sha256'],data),(manifest['source_files_sha256'],root),
                         (manifest['tokenizer_files_sha256'],tokenizer)):
        for name,digest in mapping.items():
            path=(base/name).resolve(strict=True)
            if not path.is_relative_to(base.resolve()) or file_sha256(path)!=digest:
                raise ValueError('closeout_frozen_input_changed:'+name)
    task_ids={}
    from .repair_sft import validate_tokenized_sample
    from .context_builder import load_token_counter
    counter=load_token_counter(str(tokenizer))
    for split in ('train','dev'):
        samples=rows(data/(split+'.jsonl'))
        if len(samples)!=manifest['splits'][split]['rows'] or not samples:
            raise ValueError('closeout_row_count_mismatch')
        seen=set()
        for row in samples:
            if row['split']!=split or row['id'] in seen or validate_training_row(row):
                raise ValueError('invalid_closeout_training_row')
            seen.add(row['id'])
            if {tool['function']['name'] for tool in row['tools']}!={'list_orders','get_order','preview_cancel','escalate_to_human','search_policies'}:
                raise ValueError('five_tool_schema_required')
            measured=validate_tokenized_sample(row,counter.tokenizer)
            if measured['input_tokens']>CONFIG['max_sequence_length']:
                raise ValueError('overlength_closeout_training_row')
            if row['validation']!={**measured,'reference_business_state_verified':True}:
                raise ValueError('closeout_token_measurement_changed')
        task_ids[split]={row['task_id'] for row in samples}
    if task_ids['train'] & task_ids['dev']: raise ValueError('closeout_split_leakage')
    return manifest


def training_arguments(data,base,adapter,output,*,smoke):
    arguments=old_training_arguments(data,base,adapter,output,smoke=smoke)
    for flag,value in (('--max-sequence-length',CONFIG['max_sequence_length']),('--seed',CONFIG['seed'])):
        arguments[arguments.index(flag)+1]=str(value)
    return [*arguments,'--shuffle-training-rows']


def choose_adapter(comparisons):
    if set(comparisons)!={'business','knowledge'}: raise ValueError('both_paired_dev_suites_required')
    regressions=[name for report in comparisons.values() for name in report['regressed_cases']]
    improvements=[name for report in comparisons.values() for name in report['improved_cases']]
    errors=any(report['candidate_output_failures']>report['baseline_output_failures'] for report in comparisons.values())
    choose_new=bool(improvements) and not regressions and not errors
    return {'selected':'pilot' if choose_new else 'old-online','selected_by':'paired_dev_only',
            'improvements':improvements,'regressions':regressions,'more_output_errors':errors,
            'production_promoted':False,'test_used_for_selection':False}


def claim_test(selection, cases_path, ledger):
    if selection.get('selected_by')!='paired_dev_only' or selection.get('test_used_for_selection') is not False:
        raise ValueError('test_requires_frozen_dev_selection')
    if file_sha256(cases_path)!=selection['sealed_test_sha256']:
        raise ValueError('sealed_test_changed')
    cases=rows(cases_path)
    if len(cases)!=30 or any(case.get('split')!='test' for case in cases):
        raise ValueError('unexpected_sealed_test_cases')
    if any('smoke_responses' not in case for case in cases): raise ValueError('sealed_case_schema_changed')
    # Exclusive creation persists after a crash: no silent second test attempt.
    with ledger.open('x',encoding='utf-8') as target:
        json.dump({'selection_sha256':sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest(),
                   'test_sha256':selection['sealed_test_sha256'],'status':'claimed_once'},target,indent=2)
    return cases
