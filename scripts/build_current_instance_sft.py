"""Build measured five-tool train/dev; never execute or supervise sealed test."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import argparse
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ecommerce_customer_service.serving.closeout_data import cases, SEED
from ecommerce_customer_service.serving.context_builder import load_token_counter
from ecommerce_customer_service.serving.repair_sft import execute_reference, validate_tokenized_sample
from ecommerce_customer_service.serving.split_integrity import audit
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex, augment_policy
from ecommerce_customer_service.serving.policy import get_policy


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_rows(path, values):
    with path.open('x', encoding='utf-8') as stream:
        for row in values: stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer-path', type=Path, required=True)
    p.add_argument('--sealed-cases', type=Path, required=True)
    p.add_argument('--knowledge-file', type=Path, default=ROOT/'data/demo-policies.jsonl')
    p.add_argument('--max-sequence-length', type=int, default=3072)
    args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    counter=load_token_counter(str(args.tokenizer_path))
    knowledge=KnowledgeIndex(args.knowledge_file)
    sealed=[json.loads(line) for line in args.sealed_cases.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(sealed)!=30 or any(row['split']!='test' for row in sealed): raise ValueError('unexpected_sealed_test')
    policy='binding_guard_v4'
    samples={}
    with tempfile.TemporaryDirectory(prefix='closeout-reference-') as temp:
        for split in ('train','dev'):
            generated=[]
            episodes=cases(split)
            write_rows(args.output/(split+'-cases.jsonl'), episodes)
            for index,case in enumerate(episodes):
                rows,result=execute_reference(case, Path(temp)/case['case_id'], counter,
                                              policy_profile=policy, knowledge=knowledge)
                for row in rows:
                    stats=validate_tokenized_sample(row,counter.tokenizer)
                    if stats['input_tokens']>args.max_sequence_length:
                        raise ValueError('sequence_budget_exceeded:'+row['id']+':'+str(stats['input_tokens']))
                    row['validation']={**stats,'reference_business_state_verified':True}
                generated.extend(rows)
                if (index+1)%20==0: print(split,index+1,'/',len(episodes),flush=True)
            samples[split]=generated
            write_rows(args.output/(split+'.jsonl'),generated)
    overlap=audit(samples,counter.tokenizer,sealed_cases=sealed,policy_profile=policy,knowledge=knowledge)
    write_json(args.output/'input-overlap-audit.json',overlap)
    if not overlap['passed']: raise ValueError('actual_generation_input_overlap')
    source_paths=[ROOT/'src/ecommerce_customer_service/serving'/name for name in (
        'closeout_data.py','repair_sft.py','context_builder.py','service.py','policy.py',
        'target_binding.py','split_integrity.py','online_evaluation.py','online_cases.py',
        'knowledge.py','backend.py','storage.py','telemetry.py','model.py')]
    source_paths.extend([Path(__file__),ROOT/'src/ecommerce_customer_service/training.py'])
    manifest={'schema_version':'current_instance_five_tool_sft_v1','seed':SEED,
        'policy':augment_policy(get_policy(policy)).metadata(),'knowledge_sha256':knowledge.sha256,
        'max_sequence_length':args.max_sequence_length,'training_ready':True,
        'source':'executed_synthetic_reference_not_real_customer_logs','model_training_performed':False,
        'test_reference_executed':False,'test_sft_rows':0,
        'sealed_cases_sha256':sha256(args.sealed_cases.read_bytes()).hexdigest(),
        'files_sha256':{path.name:sha256(path.read_bytes()).hexdigest() for path in args.output.iterdir() if path.is_file()},
        'source_files_sha256':{path.relative_to(ROOT).as_posix():sha256(path.read_bytes()).hexdigest() for path in source_paths},
        'tokenizer_files_sha256':{name:sha256((args.tokenizer_path/name).read_bytes()).hexdigest() for name in ('tokenizer.json','tokenizer_config.json')},
        'splits':{split:{'episodes':len(cases(split)),'rows':len(rows),
            'max_input_tokens':max(row['validation']['input_tokens'] for row in rows),
            'category_rows':dict(Counter(row['category'] for row in rows))} for split,rows in samples.items()},
        'limitations':['synthetic_shared_scenario_families','test_first_turn_audit_only',
            'repeated_knowledge_inputs_within_split','finite_target_grammar_unchanged',
            'source_checks_are_not_semantic_entailment','single_bounded_pilot']}
    write_json(args.output/'manifest.json',manifest)
    print(json.dumps(manifest['splits'],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
