"""Serve exactly the dev-selected online adapter, retaining truthful identity."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.closeout_experiment import verify_online_adapter
from ecommerce_customer_service.serving.artifacts import verify_local_base, file_sha256
from ecommerce_customer_service.serving.inference import InferenceServer, BoundedGenerator
from ecommerce_customer_service.serving.policy import get_policy
from ecommerce_customer_service.serving.knowledge import augment_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--base-model-dir',type=Path,required=True)
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    selection=json.loads(args.selection.read_text(encoding='utf-8'))
    adapter=Path(selection['selected_adapter_dir'])
    identity=verify_online_adapter(adapter)
    if identity['adapter_sha256']!=selection['selected_adapter_sha256']:
        raise ValueError('selected_weights_changed')
    verify_local_base(args.base_model_dir,adapter)
    identity.update(tokenizer_sha256=file_sha256(adapter/'tokenizer.json'),
        tokenizer_config_sha256=file_sha256(adapter/'tokenizer_config.json'),
        training_protocol='online_five_tool_repair' if selection['selected']=='pilot' else 'online_four_tool_repair',
        selection_sha256=file_sha256(args.selection),online_quality_verified=False)
    identity.update(augment_policy(get_policy('binding_guard_v4')).metadata(),
                    knowledge_index_sha256=selection['knowledge_sha256'])
    if args.check_only:
        print(json.dumps(identity,indent=2));return 0
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    from ecommerce_customer_service.local_hf import TransformersCompleter
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA_BF16_required')
    tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True,trust_remote_code=False)
    base=AutoModelForCausalLM.from_pretrained(args.base_model_dir,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,adapter,is_trainable=False,local_files_only=True);model.eval()
    config=GenerationConfig(do_sample=False,max_new_tokens=512,eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    completer=TransformersCompleter(model,tokenizer,torch,config,device='cuda:0',max_input_tokens=8192)
    server=InferenceServer(('127.0.0.1',args.port),BoundedGenerator(completer),
        diagnostics_dir=args.selection.parent/'selected-service-diagnostics',capture_invalid_output=True,
        api_key=os.environ.get('MODEL_API_KEY',''),artifact_identity=identity)
    print('SELECTED SERVICE READY '+identity['model_id']+' http://127.0.0.1:'+str(server.server_port)+'/v1',flush=True)
    print('Dev-selected demonstration adapter. Production quality is not certified.',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0


if __name__=='__main__': raise SystemExit(main())
