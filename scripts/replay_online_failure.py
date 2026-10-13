"""Replay one inference request; never execute its returned business tools."""
from pathlib import Path
import argparse
import json
import os
import sqlite3
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.context_builder import ContextBuilder, historical_observations, load_token_counter
from ecommerce_customer_service.serving.diagnostics import DIAGNOSTICS_VERSION
from ecommerce_customer_service.serving.model import ModelServiceError, OpenAIChatModel
from ecommerce_customer_service.serving.policy import PROFILES, get_policy
from ecommerce_customer_service.training import canonicalize_tool_schemas, normalize_messages


def reconstruct(case_dir, profile, counter):
    history=json.loads((case_dir/'history.json').read_text(encoding='utf-8'))
    if history['operations'] or history['status']!='active':
        raise ValueError('case_with_operations_requires_exact_captured_request')
    messages=history['messages']
    if not messages or messages[-1].get('content')!='本轮未完成，请稍后重新发送，或通过订单卡片操作。':
        raise ValueError('unsupported_failure_history_use_exact_captured_request')
    # Read-only snapshot; no Store/CustomerService is constructed and no recovery runs.
    with sqlite3.connect((case_dir/'sessions.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        audit=db.execute('SELECT tool,result,created_at FROM audit WHERE conversation_id=? ORDER BY id DESC LIMIT 50',
                         (history['conversation_id'],)).fetchall()
    state={'conversation_status':history['status'],'operations':[],
           'order_observations':historical_observations(list(reversed(audit)))}
    policy=get_policy(profile)
    built=ContextBuilder(counter=counter).build(policy.prompt,messages[:-1],policy.tools,state)
    return {'model':'ecommerce-v18-sft','messages':normalize_messages(built.messages),
            'tools':canonicalize_tool_schemas(policy.tools),'temperature':0,'max_tokens':512,'stream':False,'tool_choice':'auto'}


def private_json(path,data):
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w',encoding='utf-8') as output:
        json.dump(data,output,ensure_ascii=False,indent=2); output.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--case-dir',type=Path)
    source.add_argument('--request-json',type=Path,help='Previously saved exact replay request')
    parser.add_argument('--policy-profile',choices=PROFILES,default='binding_compact_v3')
    parser.add_argument('--tokenizer-path',default=str(ROOT/'models/v18-sft-verified'))
    parser.add_argument('--api-base',default='http://127.0.0.1:8000/v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    payload=json.loads(args.request_json.read_text(encoding='utf-8')) if args.request_json else reconstruct(
        args.case_dir,args.policy_profile,load_token_counter(args.tokenizer_path))
    if (payload.get('model')!='ecommerce-v18-sft' or payload.get('temperature')!=0
            or payload.get('max_tokens')!=512 or payload.get('stream') is not False):
        raise ValueError('replay_requires_original_greedy_512_token_protocol')
    args.output.mkdir(parents=True,exist_ok=False,mode=0o700)
    private_json(args.output/'request.json',payload)
    headers={'Authorization':'Bearer '+os.environ.get('MODEL_API_KEY','')}
    base=args.api_base.rstrip('/')
    def fetch(url):
        with urlopen(Request(url,headers=headers),timeout=15) as response: return json.load(response)
    health=fetch(base.removesuffix('/v1')+'/healthz')
    private_json(args.output/'endpoint.json',health)
    report={'business_tools_executed':False,'inference_requests':0,
            'request_provenance':'provided_request_json' if args.request_json else 'reconstructed_post_failure_snapshot',
            'limitation':'A supplied JSON reproduces that payload; its provenance must be checked separately.' if args.request_json else 'Historical request was not captured; post-failure state reconstruction is not proof of byte-identical replay.'}
    if health.get('diagnostics_version')!=DIAGNOSTICS_VERSION or not health.get('diagnostics_enabled'):
        report['status']='server_update_required'
    else:
        report['inference_requests']=1
        try:
            response=OpenAIChatModel(base,payload['model'],os.environ.get('MODEL_API_KEY','')).complete(payload['messages'],payload.get('tools',[]))
            private_json(args.output/'response.json',response)
            report['status']='generation_accepted_no_tools_executed'
        except ModelServiceError as error:
            report.update(status=error.code,**error.metadata)
            if error.metadata.get('diagnostic_id'):
                diagnostic=fetch(base+'/diagnostics/'+error.metadata['diagnostic_id'])
                private_json(args.output/'diagnostic.json',diagnostic)
                report['parse_reason']=diagnostic.get('parse_reason')
        except Exception as error:
            report.update(status='unclassified_transport_failure',exception_type=type(error).__name__)
    private_json(args.output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 2 if report['status']=='server_update_required' else 0


if __name__=='__main__': raise SystemExit(main())
