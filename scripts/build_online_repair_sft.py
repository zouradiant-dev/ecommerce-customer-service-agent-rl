from pathlib import Path
import argparse
from hashlib import sha256
import json
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.repair_sft import KINDS,PRODUCTS,REQUESTS,episode,execute_reference,validate_tokenized_sample
from ecommerce_customer_service.serving.context_builder import load_token_counter
from ecommerce_customer_service.serving.policy import get_policy,PROFILES
from ecommerce_customer_service.serving.split_integrity import audit


def artifact_receipt(tokenizer_path):
    return {'tokenizer_files_sha256':{name:sha256((Path(tokenizer_path)/name).read_bytes()).hexdigest()
             for name in ('tokenizer.json','tokenizer_config.json')},
            'source_files_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                'src/ecommerce_customer_service/serving/repair_sft.py',
                'src/ecommerce_customer_service/serving/context_builder.py',
                'src/ecommerce_customer_service/serving/service.py',
                'src/ecommerce_customer_service/serving/policy.py',
                'src/ecommerce_customer_service/serving/target_binding.py',
                'src/ecommerce_customer_service/serving/split_integrity.py',
                'src/ecommerce_customer_service/training.py',
                'scripts/build_online_repair_sft.py')}}


def dump_lines(path,rows):
    with path.open('x',encoding='utf-8') as target:
        for row in rows: target.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')


def main():
    parser=argparse.ArgumentParser(description='Construct and replay synthetic four-tool SFT references; no GPU/model training')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokenizer-path',required=True,help='Local verified tokenizer; no download')
    parser.add_argument('--train-per-kind',type=int,default=8)
    parser.add_argument('--dev-per-kind',type=int,default=2)
    parser.add_argument('--test-per-kind',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--policy-profile',choices=PROFILES,default='binding_compact_v3')
    parser.add_argument('--max-sequence-length',type=int,default=2304,help='Training limit; overflow stops dataset publication')
    args=parser.parse_args()
    counts={'train':args.train_per_kind,'dev':args.dev_per_kind,'test':args.test_per_kind}
    if any(not 1<=count<=64 for count in counts.values()): parser.error('each split count must be 1..64')
    if not 512<=args.max_sequence_length<=8192:parser.error('max sequence length must be 512..8192')
    args.output.mkdir(parents=True,exist_ok=False)
    counter=load_token_counter(args.tokenizer_path)
    all_cases={split:[episode(split,kind,index,args.seed) for kind in KINDS for index in range(count)] for split,count in counts.items()}
    # Freeze held-out cases before creating supervision; never replay test or write test SFT rows.
    for split,rows in all_cases.items(): dump_lines(args.output/(split+'-cases.jsonl'),rows)
    manifest={'schema_version':'online_repair_sft_v1','source':'synthetic_executed_reference_not_real_customer_logs',
        'seed':args.seed,'policy':get_policy(args.policy_profile).metadata(),'gpu_used':False,'model_training_performed':False,
        'test_model_evaluation_performed':False,'test_reference_replayed':False,'test_sft_rows':0,
        'split_strategy':'disjoint_episode_ids_order_ids_product_catalogs_cancel_request_templates',
        'limitations':['synthetic_templates_not_real_customer_distribution','shared_business_scenario_families','human_review_required'],
        'splits':{},'products':{split:[product for product,alias in pool] for split,pool in PRODUCTS.items()},
        'request_templates':REQUESTS,'files_sha256':{},**artifact_receipt(args.tokenizer_path)}
    manifest['max_sequence_length']=args.max_sequence_length
    seen=set()
    for split in counts:
        ids={order['id'] for case in all_cases[split] for order in case['initial_orders']}
        if seen & ids: raise ValueError('cross_split_order_collision')
        seen.update(ids)
    samples_by_split={}
    with tempfile.TemporaryDirectory(prefix='online-repair-reference-') as temporary:
        for split in ('train','dev'):
            samples=[]; token_stats=[]
            for index,case in enumerate(all_cases[split]):
                rows,result=execute_reference(case,Path(temporary)/case['case_id'],counter,policy_profile=args.policy_profile)
                for row in rows:
                    stats=validate_tokenized_sample(row,counter.tokenizer)
                    if stats['input_tokens']>args.max_sequence_length:
                        raise ValueError('training_sequence_budget_exceeded:'+row['id'])
                    row['validation']={**stats,'reference_business_state_verified':True}
                    token_stats.append(stats); samples.append(row)
                if (index+1)%10==0: print(split,'verified episodes',index+1,'/',len(all_cases[split]),flush=True)
            dump_lines(args.output/(split+'.jsonl'),samples)
            samples_by_split[split]=samples
            manifest['splits'][split]={'episodes':len(all_cases[split]),'samples':len(samples),
                'reference_business_passed':len(all_cases[split]),'max_input_tokens':max(row['input_tokens'] for row in token_stats),
                'supervised_tokens':sum(row['supervised_tokens'] for row in token_stats)}
    manifest['splits']['test']={'episodes':len(all_cases['test']),'samples':0,'status':'frozen_not_replayed_not_evaluated'}
    overlap=audit(samples_by_split,counter.tokenizer,sealed_cases=all_cases['test'],policy_profile=args.policy_profile)
    (args.output/'input-overlap-audit.json').write_text(json.dumps(overlap,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not overlap['passed']: raise ValueError('cross_split_rendered_input_overlap; dataset_not_ready')
    manifest['rendered_input_overlap_passed']=True
    manifest['split_strategy']='disjoint_ids_catalogs_requests_train_dev_rendered_prefixes_and_test_first_turns'
    manifest['training_ready']=True
    for path in args.output.glob('*.jsonl'): manifest['files_sha256'][path.name]=sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=['# 新在线协议定向 SFT 数据准备','','来源：程序构造的参考轨迹，经过模拟业务执行验证；不是用户真实日志或模型采样。',
        '未训练模型，未运行正式 test。','','| split | 场景 | 监督样本 | 状态 |','|---|---:|---:|---|']
    for split in counts:
        item=manifest['splits'][split]
        report.append(f"| {split} | {item['episodes']} | {item['samples']} | {'执行与 token/mask 校验通过' if split!='test' else '已冻结，未执行'} |")
    report+=['','每个样本只监督最后一个 assistant 回答/工具请求；输入、工具结果和按钮产生的服务端结果不计算损失。',
        '需要人工抽查表达质量、场景分布和训练预算后，才启动小规模补 SFT。合成 test 不是实际生产泛化证据。']
    (args.output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(manifest['splits'],ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
