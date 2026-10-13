"""Sequential baseline, 4-step smoke, 64-step repair SFT and paired online dev evaluation."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.artifacts import BASE_REVISION, FILE_HASHES, file_sha256, verify_adapter, verify_local_base
from ecommerce_customer_service.serving.repair_experiment import CONFIG, compare_runs, parameter_delta, rows, training_arguments, validate_bundle
from ecommerce_customer_service.serving.context_builder import load_token_counter
from ecommerce_customer_service.serving.repair_sft import validate_tokenized_sample


def write(path,data):
    with path.open('x',encoding='utf-8') as output: json.dump(data,output,ensure_ascii=False,indent=2); output.write('\n')


def child(arguments,log):
    with log.open('x',encoding='utf-8') as output:
        process=subprocess.Popen([sys.executable,'-u',*arguments],cwd=ROOT,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        try:
            for line in process.stdout:
                output.write(line);output.flush();print(line,end='',flush=True)
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
        finally:
            process.stdout.close()
    if code: raise RuntimeError('experiment_child_failed:'+str(code)+':'+log.name)


def preflight(args):
    receipt=validate_bundle(args.data_dir,args.adapter_dir)
    verify_local_base(args.base_model_dir,args.adapter_dir)
    cases=rows(args.data_dir/'paired-dev-cases.jsonl')
    if len(cases)!=57 or len({case['case_id'] for case in cases})!=57 or any(case.get('split')=='test' for case in cases):
        raise ValueError('paired_dev_case_manifest_invalid')
    frozen=json.loads((args.data_dir/'experiment.json').read_text(encoding='utf-8'))
    if frozen.get('config')!=CONFIG: raise ValueError('experiment_configuration_mismatch')
    if file_sha256(args.data_dir/'paired-dev-cases.jsonl')!=frozen['paired_dev_cases_sha256']:
        raise ValueError('paired_dev_case_hash_mismatch')
    if shutil.disk_usage(args.output).free<6*1024**3: raise RuntimeError('insufficient_experiment_disk_space')
    counter=load_token_counter(str(args.adapter_dir))
    for split in ('train','dev'):
        for row in rows(args.data_dir/(split+'.jsonl')): validate_tokenized_sample(row,counter.tokenizer)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
    import torch, peft, bitsandbytes, transformers
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA_BF16_required')
    free,total=torch.cuda.mem_get_info()
    if free<16*1024**3: raise RuntimeError('GPU_not_free_stop_existing_inference_first')
    if total<22*1024**3: raise RuntimeError('24GB_memory_class_required')
    receipt.update(gpu=torch.cuda.get_device_name(0),free_memory_gib=free/1024**3,total_memory_gib=total/1024**3,
        versions={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','bitsandbytes','accelerate')},
        experiment=CONFIG,paired_dev_cases=57)
    # Avoid retaining a CUDA context in the orchestration process across GPU children.
    return receipt


def verify_smoke(path,expected_steps,original_adapter):
    metrics=json.loads((path/'metrics.json').read_text(encoding='utf-8'))
    if (metrics['optimizer_steps']!=expected_steps or metrics['train_skipped'] or metrics['dev_skipped']
            or metrics['adapter_initialization']!='existing_adapter'
            or set(metrics['trainable_parameter_elements_by_dtype'])!={'torch.float32'}):
        raise ValueError('repair_training_metrics_gate_failed')
    if not all(math.isfinite(row['train_loss']) and math.isfinite(row['gradient_norm']) for row in metrics['trace']):
        raise ValueError('repair_nonfinite_metrics')
    if any(not math.isfinite(metrics[key]) for key in ('initial_dev_loss','final_dev_loss')):
        raise ValueError('repair_nonfinite_dev_loss')
    if file_sha256(path/'adapter_model.safetensors')==FILE_HASHES['adapter_model.safetensors']:
        raise ValueError('repair_adapter_has_no_update')
    write(path/'weight-update.json',parameter_delta(original_adapter/'adapter_model.safetensors',path/'adapter_model.safetensors'))
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--base-model-dir',type=Path,required=True)
    parser.add_argument('--adapter-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--restore-original-service',action='store_true',help='After experiment, serve the original pinned adapter again')
    parser.add_argument('--preflight-child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    for name in ('data_dir','base_model_dir','adapter_dir','output'): setattr(args,name,getattr(args,name).resolve())
    if args.output==args.adapter_dir or args.output.is_relative_to(args.adapter_dir): raise ValueError('cannot_overwrite_original_adapter')
    if args.preflight_child:
        write(args.output/'preflight.json',preflight(args));return 0
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'experiment-config.json',CONFIG)
    status=1
    comparison=None
    error_type=None
    phase_started='preflight'
    phase_completed=None
    try:
        child([str(Path(__file__).resolve()),'--preflight-child','--data-dir',str(args.data_dir),
               '--base-model-dir',str(args.base_model_dir),'--adapter-dir',str(args.adapter_dir),'--output',str(args.output)],args.output/'preflight.log')
        print('PREFLIGHT PASSED',flush=True)
        phase_completed='preflight'
        if args.preflight_only: return 0
        def evaluate(stage,adapter):
            child([str(ROOT/'scripts/evaluate_online_local_adapter.py'),'--stage',stage,'--adapter-dir',str(adapter),
                '--original-adapter-dir',str(args.adapter_dir),'--base-model-dir',str(args.base_model_dir),
                '--cases',str(args.data_dir/'paired-dev-cases.jsonl'),'--output',str(args.output/stage)],args.output/(stage+'.log'))
        phase_started='baseline'
        print('STAGE baseline BF16 paired dev',flush=True);evaluate('baseline',args.adapter_dir)
        phase_completed='baseline'
        for phase,smoke in (('smoke',True),('pilot',False)):
            print('STAGE '+phase+' QLoRA training',flush=True)
            phase_started=phase
            child([str(ROOT/'scripts/train_sft.py'),*training_arguments(args.data_dir,args.base_model_dir,args.adapter_dir,args.output/phase,smoke=smoke)],args.output/(phase+'.log'))
            verify_smoke(args.output/phase,4 if smoke else 64,args.adapter_dir)
            phase_completed=phase
        candidate=args.output/'pilot'
        files={path.name:file_sha256(path) for path in candidate.iterdir() if path.is_file() and path.suffix in {'.json','.safetensors','.txt','.jinja'} and path.name!='metrics.json'}
        write(candidate/'repair-artifact.json',{'base_revision':BASE_REVISION,'source_adapter_sha256':FILE_HASHES['adapter_model.safetensors'],
            'files_sha256':files,'formal_test_performed':False,'training_config':CONFIG,'data_receipt':json.loads((args.output/'preflight.json').read_text(encoding='utf-8'))['files_sha256']})
        phase_started='candidate'
        print('STAGE candidate BF16 paired dev',flush=True);evaluate('candidate',candidate)
        phase_completed='candidate'
        phase_started='comparison'
        comparison=compare_runs(args.output/'baseline',args.output/'candidate')
        write(args.output/'comparison.json',comparison)
        print(json.dumps(comparison,ensure_ascii=False,indent=2),flush=True)
        write(args.output/'completed.json',{'completed':True,'formal_test_performed':False,'candidate_promoted':False})
        phase_completed='comparison'
        status=0
    except Exception as error:
        error_type=type(error).__name__
        write(args.output/'failure.json',{'completed':False,'exception_type':type(error).__name__,'message':str(error),'formal_test_performed':False})
        traceback.print_exc()
    if args.restore_original_service and not args.preflight_only and (args.output/'preflight.json').exists():
        public={'schema_version':CONFIG['schema_version'],'experiment_complete':status==0,
                'candidate_promoted':False,'formal_test_performed':False,'comparison':comparison,'error_type':error_type,
                'last_started_stage':phase_started,'last_completed_stage':phase_completed}
        for phase in ('smoke','pilot'):
            path=args.output/phase/'metrics.json'
            if path.exists():
                metrics=json.loads(path.read_text(encoding='utf-8'))
                public[phase]={key:metrics[key] for key in ('optimizer_steps','train_rows','dev_rows','train_skipped',
                    'dev_skipped','initial_dev_loss','final_dev_loss','elapsed_seconds','max_memory_gib')}
                update=args.output/phase/'weight-update.json'
                if update.exists(): public[phase]['weight_update']=json.loads(update.read_text(encoding='utf-8'))
        report_path=args.output/'public-experiment-report.json'
        write(report_path,public)
        # All GPU work was in exited children; no training optimizer survives into serving.
        print('Restoring ORIGINAL pinned inference service. Experiment status='+str(status),flush=True)
        os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'scripts/serve_v18_sft.py'),
            '--adapter-dir',str(args.adapter_dir),'--base-model-dir',str(args.base_model_dir),'--precision','bf16',
            '--diagnostics-dir',str(args.output/'restored-service-diagnostics'),'--capture-invalid-output',
            '--repair-experiment-report',str(report_path)])
    return status


if __name__=='__main__': raise SystemExit(main())
