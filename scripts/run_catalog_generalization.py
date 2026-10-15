"""Second declared controller iteration. Preserve first acceptance; use fresh cases."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.repair_sft import KINDS, PRODUCTS, episode
from ecommerce_customer_service.serving.online_evaluation import _run_suite
from ecommerce_customer_service.serving.context_builder import ContextBuilder, load_token_counter
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_rows(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def build_cases(output):
    output.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'outputs/grounded-generalization-cases-20261015-r1'
    consumed=read_rows(previous/'acceptance.jsonl')
    for case in consumed:case.update(split='dev',evaluation_status='first_acceptance_now_consumed_development_regression')
    fresh=[]
    for kind in KINDS:
        for variant in range(2):
            case=episode('train',kind,variant,20261017)
            case.update(case_id=f'catalog_acceptance_{kind}_{variant:03d}',split='test',provenance='new_synthetic_phrase_composition_known_catalog_not_independent_human_test')
            if kind not in {'query_only','cross_owner','handoff'}:
                product=case['initial_orders'][0]['product']
                if kind=='unique_alias':product=dict(PRODUCTS['train'])[product]
                if kind=='no_match':product='户外餐桌'
                request=('这笔{product}的购买请帮忙撤单。','针对{product}这单，请申请取消。')[variant].format(product=product)
                index=2 if kind=='query_then_cancel' else next(i for i,s in enumerate(case['steps']) if s['kind']=='message')
                case['steps'][index]['text']=request
            if kind=='true_ambiguity':
                case['steps'][2]['text']=f"选择{case['initial_orders'][0]['paid_cents']/100:.2f}元的那笔。"
                case['steps'].insert(2,{'kind':'event','name':'restart'})
            fresh.append(case)
    suites={'consumed-acceptance-dev':consumed,'prior-business-dev':read_rows(previous/'prior-business-dev.jsonl'),'fresh-acceptance':fresh}
    for name,rows in suites.items():
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows),encoding='utf-8')
    manifest={'version':'catalog_alias_iteration_v1','first_acceptance_result_preserved':'40/42','old_acceptance_now_dev':True,
              'fresh_acceptance_author_knows_templates':True,'catalog_vocabulary_seen_previously':True,
              'independent_distribution_claim':False,'training':False,'counts':{name:len(rows) for name,rows in suites.items()},
              'files_sha256':{path.name:sha(path) for path in output.glob('*.jsonl')}}
    write(output/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build-only',action='store_true')
    p.add_argument('--cpu-fixtures',action='store_true')
    args=p.parse_args()
    cases_dir=ROOT/'outputs/catalog-generalization-cases-20261015-r1'
    if not cases_dir.exists():build_cases(cases_dir)
    if args.build_only:print('NEW CASES FROZEN. No model calls.');return 0
    manifest=json.loads((cases_dir/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files_sha256'].items():
        if sha(cases_dir/name)!=digest:raise ValueError('frozen_cases_changed')
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((ROOT/'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    with urlopen('http://127.0.0.1:8647/healthz',timeout=10) as response:health=json.load(response)
    if not health['ready'] or health['adapter_sha256']!=receipt['selected_identity']['adapter_sha256'] or health['precision']!='bf16':raise ValueError('changed_model')
    source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/ecommerce_customer_service/serving').glob('*.py')}
    write(args.output/'protocol.json',{'profile':'binding_catalog_v6','frozen_source':source,'health':health,'cases_manifest_sha256':sha(cases_dir/'manifest.json'),
        'no_training':True,'same_model_as_first_iteration':True,'prompt_and_alias_tool_payload_changed':True,'production_promoted':False})
    model=OpenAIChatModel('http://127.0.0.1:8647/v1',health['model_id'])
    context=ContextBuilder(counter=load_token_counter(receipt['selected_adapter_dir']) if not args.cpu_fixtures else None,max_input_tokens=8192 if not args.cpu_fixtures else 32768)
    knowledge=KnowledgeIndex(Path(receipt['knowledge_file']))
    def run(name,split='dev'):
        if any(sha(ROOT/name)!=digest for name,digest in source.items()):raise ValueError('frozen_runtime_changed')
        print('STAGE '+name+' binding_catalog_v6',flush=True)
        return _run_suite(read_rows(cases_dir/(name+'.jsonl')),args.output/name,mode='smoke' if args.cpu_fixtures else 'model',
            model=None if args.cpu_fixtures else model,context=context,policy_profile='binding_catalog_v6',knowledge=knowledge,evaluation_split=split,progress=True)
    dev=run('consumed-acceptance-dev');prior=run('prior-business-dev')
    gate=all(report['complete'] and report['passed_cases']==report['evaluated_cases'] for report in (dev,prior))
    selection={'selected_by':'consumed_acceptance_dev_and_prior_regression','profile':'binding_catalog_v6','gate_passed':gate,
               'dev':{'passed':dev['passed_cases'],'total':dev['evaluated_cases']},'prior':{'passed':prior['passed_cases'],'total':prior['evaluated_cases']},
               'acceptance_used_for_selection':False,'acceptance_performed':False,'no_training':True,'production_promoted':False}
    write(args.output/'dev-selection.json',selection)
    if gate and not args.cpu_fixtures:
        with (args.output/'acceptance-claim.json').open('x',encoding='utf-8') as target:json.dump({'selection_sha256':sha(args.output/'dev-selection.json'),'cases_sha256':manifest['files_sha256']['fresh-acceptance.jsonl'],'one_attempt':True},target)
        report=run('fresh-acceptance','test')
        selection.update(acceptance_performed=True,acceptance_result={key:report[key] for key in ('complete','passed_cases','evaluated_cases','case_pass_rate')})
    write(args.output/'completed.json',selection)
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)
    return 0 if gate else 2


if __name__=='__main__':raise SystemExit(main())
