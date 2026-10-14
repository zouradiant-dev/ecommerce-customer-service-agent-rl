"""Real inference HTTP -> browser API -> isolated mock business -> restart check."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import os
import secrets
import sys
import threading
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.api import OnlineServer
from ecommerce_customer_service.serving.backend import MockBusinessBackend
from ecommerce_customer_service.serving.closeout_experiment import verify_online_adapter
from ecommerce_customer_service.serving.context_builder import ContextBuilder, load_token_counter
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.model import OpenAIChatModel
from ecommerce_customer_service.serving.service import CustomerService
from ecommerce_customer_service.serving.storage import Store


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--knowledge-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--api-base',default='http://127.0.0.1:8000/v1')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads(args.selection.read_text(encoding='utf-8'))
    identity=verify_online_adapter(Path(selection['selected_adapter_dir']))
    if identity['adapter_sha256']!=selection['selected_adapter_sha256']: raise ValueError('selected_adapter_changed')
    key=os.environ.get('MODEL_API_KEY','')
    with urlopen(Request(args.api_base.rstrip('/')[:-3]+'/healthz',headers={'Authorization':'Bearer '+key}),timeout=15) as response:
        health=json.load(response)
    if health.get('adapter_sha256')!=identity['adapter_sha256'] or health.get('model_id')!=identity['model_id']:
        raise ValueError('http_inference_identity_mismatch')
    model=OpenAIChatModel(args.api_base,identity['model_id'],key,timeout=180)
    token=secrets.token_urlsafe(32)
    auth={sha256(token.encode()).hexdigest():'customer_alice'}
    knowledge=KnowledgeIndex(args.knowledge_file)
    if knowledge.sha256!=selection['knowledge_sha256']: raise ValueError('http_knowledge_changed')
    context=ContextBuilder(counter=load_token_counter(selection['selected_adapter_dir']),max_input_tokens=8192,recent_turns=6)
    backend=MockBusinessBackend(args.output/'business.sqlite')
    server=thread=None
    evidence={}
    def start():
        service=CustomerService(Store(args.output/'sessions.sqlite'),backend,model,context_builder=context,
                                policy_profile='binding_guard_v4',knowledge=knowledge)
        service.recover()
        current=OnlineServer(('127.0.0.1',0),service,auth)
        worker=threading.Thread(target=current.serve_forever,daemon=True);worker.start()
        return current,worker
    def stop(current,worker):
        current.shutdown();current.server_close();worker.join()
    def request(path,body=None):
        data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None
        with urlopen(Request(f'http://127.0.0.1:{server.server_port}'+path,data=data,
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}),timeout=240) as response:
            return response.read()
    def call(path,body=None): return json.loads(request(path,body))
    status=False;error=None
    try:
        server,thread=start()
        page=request('/').decode('utf-8')
        evidence['html_served']=('在线客服' in page and '发送' in page)
        if not evidence['html_served']: raise ValueError('webpage_not_served')
        evidence['initial_orders']=call('/api/orders')
        conversation=call('/api/conversations',{})['conversation_id']
        path='/api/conversations/'+conversation+'/messages'
        query=call(path,{'request_id':'http-query','text':'先查询我的订单，不要操作。'})
        evidence['query']=query
        history=call('/api/conversations/'+conversation)
        if history['operations']: raise ValueError('query_created_operation')
        cancel={'request_id':'http-preview','text':'帮我取消无线耳机订单。'}
        preview=call(path,cancel);evidence['preview']=preview
        operation=preview.get('operation')
        if not operation or operation.get('order_id')!='order_A': raise ValueError('correct_preview_missing')
        if preview.get('error'): raise ValueError('preview_model_turn_failed')
        before=call('/api/orders/order_A')
        evidence['before_confirm']=before
        if before.get('order',before).get('status')!='paid': raise ValueError('cancelled_before_button_confirmation')
        repeated=call(path,cancel)
        if repeated.get('operation',{}).get('id')!=operation['id']: raise ValueError('message_idempotency_failed')
        stop(server,thread);server=thread=None
        server,thread=start()
        restored=call('/api/conversations/'+conversation)
        evidence['restart_restored_messages']=len(restored['messages'])
        if not any(item['id']==operation['id'] for item in restored['operations']): raise ValueError('preview_lost_after_restart')
        confirmation=call('/api/operations/'+operation['id']+'/confirm',{})
        duplicate=call('/api/operations/'+operation['id']+'/confirm',{})
        evidence['confirmation']=confirmation;evidence['duplicate_confirmation']=duplicate
        with backend.transaction() as db:
            final=db.execute('SELECT status FROM orders WHERE id=?',('order_A',)).fetchone()[0]
            counts={name:db.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in ('cancellations','refunds')}
            refunds=[dict(row) for row in db.execute('SELECT * FROM refunds')]
        evidence.update(final_order_status=final,counts=counts,refunds=refunds)
        if final!='cancelled' or counts!={'cancellations':1,'refunds':1} or refunds[0]['status']!='pending':
            raise ValueError('confirmed_state_or_idempotency_failed')
        status=True
    except Exception as caught:
        error={'exception_type':type(caught).__name__,'message':str(caught)}
    finally:
        if server: stop(server,thread)
        report={'schema_version':'closeout_http_integration_v1','passed':status,'real_model_http_execution':True,
            'adapter_sha256':identity['adapter_sha256'],'browser_visual_checked':False,
            'browser_js_interaction_checked':False,'same_browser_api_routes_checked':True,
            'simulated_business_only':True,'error':error,'evidence':evidence}
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'http_integration_passed':status,'browser_visual_checked':False,'error':error},ensure_ascii=False),flush=True)
    return 0 if status else 2


if __name__=='__main__': raise SystemExit(main())
