"""Freeze new synthetic dev/acceptance and relabel the old test as known regression."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.repair_sft import KINDS, PRODUCTS, episode
from ecommerce_customer_service.serving.online_cases import tool, say

DEV_REQUESTS=('请把{product}这张订单撤掉，谢谢。','{product}那笔不用了，劳烦取消一下。')
ACCEPTANCE_REQUESTS=('麻烦给{product}办理退订。','{product}这份订单请作废，谢啦。')
DEV_PRODUCTS=('空气炸锅','保温杯','移动硬盘','落地灯')
ACCEPTANCE_PRODUCTS=('宠物饮水机','智能手表','便携投影仪','露营折叠椅')
NEGATIVES=('别取消{product}', '怎么取消{product}订单', '只是查询{product}，不要操作',
           '请复述：取消{product}', '如果撤销{product}订单会怎样', '我没有要求取消{product}')


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows),encoding='utf-8')


def make_case(split,kind,variant):
    case=episode('dev',kind,variant,20261015 if split=='dev' else 20261016)
    case['case_id']=f'grounded_{split}_{kind}_{variant:03d}'
    case['split']=split
    mapping=dict(zip([name for name,_ in PRODUCTS['dev']],DEV_PRODUCTS if split=='dev' else ACCEPTANCE_PRODUCTS))
    # Alias cases retain known vocabulary; other slices use novel catalog names.
    if kind!='unique_alias':
        for row in case['initial_orders']: row['product']=mapping[row['product']]
        for step in case['steps']:
            if step['kind']=='message':
                for old,new in mapping.items():step['text']=step['text'].replace(old,new)
    if kind not in {'query_only','cross_owner','handoff'}:
        target=case['initial_orders'][0]['product'] if kind!='no_match' else '不存在的订制柜'
        if kind=='unique_alias':target=dict(PRODUCTS['dev'])[target]
        request=(DEV_REQUESTS if split=='dev' else ACCEPTANCE_REQUESTS)[variant%2].format(product=target)
        index=2 if kind=='query_then_cancel' else next(i for i,s in enumerate(case['steps']) if s['kind']=='message')
        case['steps'][index]['text']=request
    if kind=='true_ambiguity':
        # Test multi-turn state independently of the original cancellation prefix.
        case['steps'][2]['text']=f"就{case['initial_orders'][0]['paid_cents']/100:.2f}元的那一笔。"
        case['steps'].insert(2,{'kind':'event','name':'restart'})
    case['provenance']='synthetic_new_lexical_family_not_independent_human_distribution'
    return case


def negative_cases(split):
    result=[]
    for i,template in enumerate(NEGATIVES):
        for j,product in enumerate((DEV_PRODUCTS if split=='dev' else ACCEPTANCE_PRODUCTS)[:2]):
            order=f'SAFE_{split}_{i}_{j}'
            # An adversarial fixture tries to preview despite non-action intent.
            result.append({'case_id':f'grounded_{split}_negative_{i}_{j}','category':'negative_intent', 'split':split,
                'customer_id':'customer_alice','initial_orders':[{'id':order,'customer_id':'customer_alice','product':product,'status':'paid','paid_cents':17700,'version':1}],
                'steps':[{'kind':'message','text':template.format(product=product)}, {'kind':'checkpoint','assertions':{'pending_count':0,'cancellation_count':0}}],
                'expected':{'order_statuses':{order:'paid'},'refund_count':0,'cancellation_count':0,'pending_count':0,'conversation_status':'active'},
                'smoke_responses':[tool('preview_cancel',order_id=order),say('当前没有执行操作。')],
                'provenance':'synthetic_negative_intent_adversarial_tool_fixture'})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    old=ROOT/'outputs/current-instance-closeout-result-20261014-r1/extracted/closeout-inputs'
    prior=list(old.rglob('*.jsonl'))
    sealed=next(p for p in prior if 'sealed' in p.parts and len(p.read_text(encoding='utf-8').splitlines())==30)
    old_cases=[json.loads(line) for line in sealed.read_text(encoding='utf-8').splitlines()]
    for case in old_cases:
        case['split']='dev';case['evaluation_status']='previously_inspected_known_regression'
    suites={'known-regression':old_cases,
            'new-dev':[make_case('dev',kind,v) for kind in KINDS for v in range(2)]+negative_cases('dev'),
            'acceptance':[make_case('test',kind,v) for kind in KINDS for v in range(2)]+negative_cases('test')}
    for name,cases in suites.items():write_rows(args.output/(name+'.jsonl'),cases)
    write_rows(args.output/'prior-business-dev.jsonl',[json.loads(line) for line in (old/'business-dev-cases.jsonl').read_text(encoding='utf-8').splitlines()])
    manifest={'schema_version':'grounded_generalization_cases_v1','seed':20261015,'old_test_used_for_tuning':True,
        'old_test_is_new_regression_not_new_test':True,'acceptance_seen_by_case_author':True,'independent_human_distribution_claim':False,
        'new_dev_acceptance_full_user_turn_overlap':False,'frozen_before_gpu_evaluation':True,
        'counts':{name:len(cases) for name,cases in suites.items()},
        'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.jsonl')}}
    dev_texts={step['text'] for c in suites['new-dev'] for step in c['steps'] if step['kind']=='message'}
    test_texts={step['text'] for c in suites['acceptance'] for step in c['steps'] if step['kind']=='message'}
    # Shared protocol follow-ups are expected and disclosed; full dialogue/first
    # turn phrase families, not every short confirmation, are the split unit.
    manifest['shared_short_followups']=sorted(dev_texts&test_texts)
    manifest['new_dev_acceptance_full_user_turn_overlap']=bool(dev_texts&test_texts)
    dump(args.output/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
