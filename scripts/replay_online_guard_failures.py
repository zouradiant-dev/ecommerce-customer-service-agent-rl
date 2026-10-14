"""Replay recorded model responses against a new runtime; NOT a model evaluation."""
import argparse
from copy import deepcopy
import json
from hashlib import sha256
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.context_builder import ContextBuilder
from ecommerce_customer_service.serving.online_evaluation import run_case
from ecommerce_customer_service.serving.policy import get_policy


class RecordedResponses:
    model='recorded_responses_not_live_llm'
    def __init__(self,history):self.responses=[deepcopy(m) for m in history['messages'] if m['role']=='assistant']
    def complete(self,messages,tools):return self.responses.pop(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cases={c['case_id']:c for c in [json.loads(line) for line in (args.backup/'experiment-data/paired-dev-cases.jsonl').read_text(encoding='utf-8').splitlines()]}
    candidate=args.backup/'outputs/online-repair-eval-resume-3cc5e3c9/candidate/cases'
    results=[]
    for directory in sorted(candidate.iterdir()):
        old=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if old['passed']:continue
        result=run_case(cases[directory.name],args.output/'cases'/directory.name,
            RecordedResponses(json.loads((directory/'history.json').read_text(encoding='utf-8'))),
            ContextBuilder(max_input_tokens=32768),policy_profile='binding_guard_v4')
        results.append(result)
    receipt={'kind':'recorded_model_response_engineering_replay','live_model_called':False,
        'model_quality_claim':False,'historical_scores_modified':False,'cases':results,
        'policy':get_policy('binding_guard_v4').metadata(),
        'source_sha256':{name:sha256((ROOT/'src/ecommerce_customer_service/serving'/name).read_bytes()).hexdigest()
                         for name in ('service.py','target_binding.py','policy.py','online_evaluation.py')},
        'limitations':['recorded_responses_do_not_adapt_to_new_tool_errors','not_new_SFT_or_agent_success_rate']}
    (args.output/'report.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{'case':r['case_id'],'passed':r['passed'],'failures':r['failures']} for r in results]))
