"""Evaluate one verified online adapter; ordinary entry accepts dev only."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evaluate_online_local_adapter import LocalChatModel
from ecommerce_customer_service.local_hf import TransformersCompleter
from ecommerce_customer_service.serving.artifacts import BASE_REVISION, verify_adapter, verify_local_base, file_sha256
from ecommerce_customer_service.serving.closeout_experiment import CONFIG, verify_online_adapter
from ecommerce_customer_service.serving.context_builder import ContextBuilder, HFTokenCounter
from ecommerce_customer_service.serving.diagnostics import DiagnosticStore
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.online_evaluation import _run_suite
from ecommerce_customer_service.serving.repair_experiment import rows


def evaluate(args, *, sealed=False):
    identity=verify_online_adapter(args.adapter_dir)
    verify_adapter(args.original_adapter_dir)
    verify_local_base(args.base_model_dir,args.adapter_dir)
    cases=rows(args.cases)
    split='test' if sealed else 'dev'
    if any(case.get('split')!=split for case in cases): raise ValueError('wrong_evaluation_split')
    if args.output.exists(): raise FileExistsError('evaluation_output_exists')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA_BF16_required')
    tokenizer=AutoTokenizer.from_pretrained(args.adapter_dir,local_files_only=True,trust_remote_code=False)
    original=AutoTokenizer.from_pretrained(args.original_adapter_dir,local_files_only=True,trust_remote_code=False)
    if tokenizer.get_vocab()!=original.get_vocab() or tokenizer.chat_template!=original.chat_template:
        raise ValueError('closeout_tokenizer_changed')
    base=AutoModelForCausalLM.from_pretrained(args.base_model_dir,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,args.adapter_dir,is_trainable=False,local_files_only=True)
    model.eval()
    generation=GenerationConfig(do_sample=False,max_new_tokens=512,eos_token_id=tokenizer.eos_token_id,
                               pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    client=LocalChatModel(TransformersCompleter(model,tokenizer,torch,generation,device='cuda:0',max_input_tokens=8192),
                         identity['model_id'],DiagnosticStore(args.output.parent/(args.output.name+'-private-diagnostics'),capture_raw=True))
    summary=_run_suite(cases,args.output,mode='model',model=client,
        context=ContextBuilder(counter=HFTokenCounter(tokenizer),max_input_tokens=8192,recent_turns=6),
        policy_profile=CONFIG['policy_profile'],knowledge=KnowledgeIndex(args.knowledge_file),
        evaluation_split=split,progress=True)
    artifact={'base_revision':BASE_REVISION,'precision':'bf16',**identity,
        'template_sha256':sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        'vocab_sha256':sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        'generation':{'do_sample':False,'max_new_tokens':512,'context_budget':8192},
        'formal_test_performed':sealed,'synthetic_test_only':sealed}
    (args.output/'artifact.json').write_text(json.dumps(artifact,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:summary[key] for key in ('complete','passed_cases','evaluated_cases','case_pass_rate')},indent=2),flush=True)
    return 0 if summary['complete'] else 2


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-model-dir','adapter-dir','original-adapter-dir','cases','output','knowledge-file'):
        p.add_argument('--'+name,type=Path,required=True)
    return p


if __name__=='__main__': raise SystemExit(evaluate(parser().parse_args()))
