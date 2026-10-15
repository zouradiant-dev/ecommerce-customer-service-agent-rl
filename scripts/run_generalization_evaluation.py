"""Same-weight, same-prompt controller comparison; no training is performed."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.context_builder import ContextBuilder, load_token_counter
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.online_evaluation import _run_suite
from ecommerce_customer_service.serving.policy import get_policy
from ecommerce_customer_service.serving.storage import encode


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--api-base',default='http://127.0.0.1:8647/v1')
    p.add_argument('--cpu-fixtures',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.cases_dir/'manifest.json').read_text(encoding='utf-8'))
    for name,sha in manifest['files_sha256'].items():
        if digest(args.cases_dir/name)!=sha:raise ValueError('case_manifest_changed:'+name)
    receipt=json.loads((ROOT/'outputs/current-instance-closeout-result-20261014-r1/verification-receipt.json').read_text(encoding='utf-8'))
    source={str(path.relative_to(ROOT)):digest(path) for path in (ROOT/'src/ecommerce_customer_service/serving').glob('*.py')}
    model=None; health=None
    if not args.cpu_fixtures:
        with urlopen(args.api_base.removesuffix('/v1')+'/healthz',timeout=10) as response:health=json.load(response)
        if (not health.get('ready') or health['adapter_sha256']!=receipt['selected_identity']['adapter_sha256']
                or health['model_id']!=receipt['selected_identity']['model_id'] or health['precision']!='bf16'):
            raise ValueError('inference_identity_mismatch')
        model=OpenAIChatModel(args.api_base,health['model_id'])
    context=ContextBuilder(counter=load_token_counter(receipt['selected_adapter_dir']) if not args.cpu_fixtures else None,
                           max_input_tokens=8192 if not args.cpu_fixtures else 32768,recent_turns=6)
    knowledge=KnowledgeIndex(Path(receipt['knowledge_file']))
    protocol={'schema_version':'grounded_controller_comparison_v1','mode':'engineering_fixtures' if args.cpu_fixtures else 'real_model_http',
              'no_training':True,'frozen_source':source,'cases_manifest_sha256':digest(args.cases_dir/'manifest.json'),
              'same_weights':True,'same_prompt_and_tools':True,'same_knowledge_sha256':knowledge.sha256,'health':health,
              'acceptance_used_for_selection':False,'production_promotion':False}
    assert get_policy('binding_guard_v4').prompt==get_policy('binding_grounded_v5').prompt
    (args.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')

    def run(name,profile,split='dev'):
        for path,sha in source.items():
            if digest(ROOT/path)!=sha:raise ValueError('runtime_changed_during_frozen_evaluation:'+path)
        cases=[json.loads(line) for line in (args.cases_dir/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        print('STAGE '+name+' '+profile,flush=True)
        return _run_suite(cases,args.output/(name+'-'+profile),mode='smoke' if args.cpu_fixtures else 'model',
            model=model,context=context,policy_profile=profile,knowledge=knowledge,evaluation_split=split,progress=True)

    baseline=run('new-dev','binding_guard_v4')
    candidate=run('new-dev','binding_grounded_v5')
    known=run('known-regression','binding_grounded_v5')
    prior=run('prior-business-dev','binding_grounded_v5')
    reports={'new-dev-baseline':baseline,'new-dev-candidate':candidate,'known-regression':known,'prior-business-dev':prior}
    def results(name):
        return {p.parent.name:json.loads(p.read_text(encoding='utf-8')) for p in (args.output/name/'cases').glob('*/result.json')}
    old=results('new-dev-binding_guard_v4');new=results('new-dev-binding_grounded_v5')
    regressions=[key for key in old if old[key]['passed'] and not new[key]['passed']]
    new_negative=[row for row in new.values() if row['category']=='negative_intent']
    gate=all(report['complete'] for report in reports.values()) and not regressions and prior['passed_cases']==prior['evaluated_cases']
    gate=gate and all(row['passed'] for row in new_negative) and known['passed_cases']>10 and candidate['passed_cases']>baseline['passed_cases']
    selection={'selected_by':'dev_and_known_regression_only','gate_passed':gate,'regressed_dev_cases':regressions,
               'case_results':{name:{key:report[key] for key in ('passed_cases','evaluated_cases','complete','model_output_failures','model_infrastructure_failures')} for name,report in reports.items()},
               'acceptance_performed':False,'production_promoted':False,'cpu_fixtures_not_model_quality':args.cpu_fixtures}
    (args.output/'dev-selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    # Fixtures validate engineering only. Do not open/execute acceptance with them.
    if gate and not args.cpu_fixtures:
        with (args.output/'acceptance-claim.json').open('x',encoding='utf-8') as target:
            json.dump({'selection_sha256':digest(args.output/'dev-selection.json'),'cases_sha256':manifest['files_sha256']['acceptance.jsonl'],'one_attempt':True},target)
        acceptance=run('acceptance','binding_grounded_v5','test')
        selection['acceptance_performed']=True
        selection['acceptance_result']={key:acceptance[key] for key in ('passed_cases','evaluated_cases','complete','case_pass_rate')}
    (args.output/'completed.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)
    return 0 if gate else 2


if __name__=='__main__':raise SystemExit(main())
