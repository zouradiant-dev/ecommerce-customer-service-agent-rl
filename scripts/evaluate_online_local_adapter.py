"""Real BF16 adapter evaluation on isolated online dev cases, without a HTTP server."""
from pathlib import Path
import argparse
from hashlib import sha256
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.local_hf import TransformersCompleter
from ecommerce_customer_service.serving.artifacts import BASE_REVISION, file_sha256, verify_adapter, verify_local_base
from ecommerce_customer_service.serving.context_builder import ContextBuilder, HFTokenCounter
from ecommerce_customer_service.serving.diagnostics import DiagnosticStore, OUTPUT_ERRORS, classify_failure
from ecommerce_customer_service.serving.model import ModelOutputError, ModelServiceError
from ecommerce_customer_service.serving.online_evaluation import run_suite
from ecommerce_customer_service.serving.repair_experiment import rows
from ecommerce_customer_service.serving.policy import PROFILES
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex


class LocalChatModel:
    def __init__(self, completer, model_id, diagnostics):
        self.completer,self.model,self.diagnostics=completer,model_id,diagnostics

    def complete(self,messages,tools):
        try:
            result=self.completer(messages,tools)
            if result.get('local_finish_reason')=='length':
                from ecommerce_customer_service.local_hf import ToolCallParseError
                error=ToolCallParseError('generation_truncated',result.get('local_raw_generation',''))
                error.generation_metadata=result
                raise error
        except Exception as error:
            status,code,reason,metadata,raw=classify_failure(error)
            identifier=self.diagnostics.record(code=code,exception_type=type(error).__name__,metadata=metadata,raw=raw,parse_reason=reason)
            details={'diagnostic_id':identifier}
            for source,target in (('local_input_tokens','input_tokens'),('local_generated_tokens','output_tokens')):
                if source in metadata: details[target]=metadata[source]
            kind=ModelOutputError if code in OUTPUT_ERRORS else ModelServiceError
            raise kind(code,status,details) from error
        return {key:result[key] for key in ('role','content','tool_calls') if key in result} | {
            '_provider_metadata':{'response_model':self.model,'input_tokens':result['local_input_tokens'],
                                  'output_tokens':result['local_generated_tokens'],'finish_reason':result['local_finish_reason']}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model-dir',type=Path,required=True)
    parser.add_argument('--adapter-dir',type=Path,required=True)
    parser.add_argument('--original-adapter-dir',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('baseline','candidate'),required=True)
    parser.add_argument('--policy-profile',choices=PROFILES,default='binding_compact_v3')
    parser.add_argument('--knowledge-file',type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('evaluation_output_exists')
    cases=rows(args.cases)
    if any(case.get('split')=='test' for case in cases): raise ValueError('formal_test_not_allowed_in_repair_pilot')
    verify_adapter(args.original_adapter_dir)
    if args.stage=='baseline': verify_adapter(args.adapter_dir)
    else:
        receipt=json.loads((args.adapter_dir/'repair-artifact.json').read_text(encoding='utf-8'))
        if receipt.get('formal_test_performed') is not False or receipt.get('base_revision')!=BASE_REVISION:
            raise ValueError('candidate_artifact_receipt_mismatch')
        for name,digest in receipt['files_sha256'].items():
            if Path(name).name!=name or file_sha256(args.adapter_dir/name)!=digest:
                raise ValueError('candidate_artifact_hash_mismatch')
    verify_local_base(args.base_model_dir,args.adapter_dir)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA_BF16_required')
    original_tokenizer=AutoTokenizer.from_pretrained(args.original_adapter_dir,local_files_only=True,trust_remote_code=False)
    tokenizer=AutoTokenizer.from_pretrained(args.adapter_dir,local_files_only=True,trust_remote_code=False)
    if tokenizer.get_vocab()!=original_tokenizer.get_vocab() or tokenizer.chat_template!=original_tokenizer.chat_template:
        raise ValueError('candidate_tokenizer_template_or_vocab_changed')
    base=AutoModelForCausalLM.from_pretrained(args.base_model_dir,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,args.adapter_dir,is_trainable=False,local_files_only=True)
    model.eval()
    generation=GenerationConfig(do_sample=False,max_new_tokens=512,eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    adapter_hash=file_sha256(args.adapter_dir/'adapter_model.safetensors')
    identifier='ecommerce-v18-sft' if args.stage=='baseline' else 'ecommerce-online-repair-'+adapter_hash[:12]
    diagnostic_dir=args.output.parent/(args.output.name+'-private-diagnostics')
    client=LocalChatModel(TransformersCompleter(model,tokenizer,torch,generation,device='cuda:0',max_input_tokens=8192),
        identifier,DiagnosticStore(diagnostic_dir,capture_raw=True))
    result=run_suite(cases,args.output,mode='model',model=client,
        context=ContextBuilder(counter=HFTokenCounter(tokenizer),max_input_tokens=8192,recent_turns=6),
        policy_profile=args.policy_profile,knowledge=KnowledgeIndex(args.knowledge_file) if args.knowledge_file else None)
    outcomes={path.parent.name:json.loads(path.read_text(encoding='utf-8'))['passed'] for path in args.output.glob('cases/*/result.json')}
    groups=sorted({case.get('regression_group','unspecified') for case in cases})
    result['by_regression_group']={group:{
        'cases':sum(case.get('regression_group','unspecified')==group for case in cases),
        'evaluated':sum(case.get('regression_group','unspecified')==group and case['case_id'] in outcomes for case in cases),
        'passed':sum(case.get('regression_group','unspecified')==group and outcomes.get(case['case_id'],False) for case in cases)} for group in groups}
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    artifact={'base_revision':BASE_REVISION,'precision':'bf16','adapter_sha256':adapter_hash,'model_id':identifier,
        'template_sha256':sha256(str(tokenizer.chat_template).encode('utf-8')).hexdigest(),
        'vocab_sha256':sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest(),
        'generation':{'do_sample':False,'max_new_tokens':512,'context_budget':8192},'formal_test_performed':False}
    (args.output/'artifact.json').write_text(json.dumps(artifact,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('complete','passed_cases','evaluated_cases','case_pass_rate','model_output_failures','model_infrastructure_failures')},indent=2),flush=True)
    # Low task success is a measured baseline/candidate result; infrastructure failure aborts the experiment.
    return 0 if result['complete'] else 2


if __name__=='__main__': raise SystemExit(main())
