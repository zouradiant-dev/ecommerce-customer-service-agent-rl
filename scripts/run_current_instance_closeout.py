"""One 4-step smoke, one 64-step pilot, paired dev selection and one sealed test."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
from hashlib import sha256
from collections import Counter
import random
import os
import subprocess
import sys
import traceback
import time
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.artifacts import BASE_REVISION, FILE_HASHES, verify_adapter, verify_local_base, file_sha256
from ecommerce_customer_service.serving.closeout_experiment import CONFIG, ONLINE_SHA, validate_data, verify_online_adapter, training_arguments, choose_adapter
from ecommerce_customer_service.serving.repair_experiment import compare_runs, parameter_delta, rows


def write(path,value):
    with path.open('x',encoding='utf-8') as target:
        json.dump(value,target,ensure_ascii=False,indent=2);target.write('\n')


def child(argv,log):
    with log.open('x',encoding='utf-8') as target:
        process=subprocess.Popen([sys.executable,'-u',*map(str,argv)],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        try:
            for line in process.stdout:
                target.write(line);target.flush();print(line,end='',flush=True)
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
        finally: process.stdout.close()
    if code: raise RuntimeError('closeout_child_failed:'+log.name+':'+str(code))


def gpu_free():
    # Never import torch in the parent orchestrator; children own their CUDA context.
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if active.strip(): raise RuntimeError('GPU_has_compute_processes_no_automatic_kill')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free<18000: raise RuntimeError('less_than_18GB_GPU_free')
    return free


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','base-model-dir','original-adapter','online-adapter','screen-output','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume-after-smoke',action='store_true',help='Reuse only the audited smoke stopped by the known double-shuffle mismatch')
    p.add_argument('--trainer-script',type=Path,default=ROOT/'scripts/train_sft.py')
    p.add_argument('--serve-selected-after',action='store_true',help='After acceptance, keep the selected inference service in this SSH foreground')
    args=p.parse_args()
    args.output=args.output.resolve()
    for adapter in (args.original_adapter,args.online_adapter):
        if args.output.is_relative_to(adapter.resolve()): raise ValueError('cannot_overwrite_saved_weights')
    verify_adapter(args.original_adapter)
    if verify_online_adapter(args.online_adapter)['adapter_sha256']!=ONLINE_SHA: raise ValueError('must_initialize_saved_online_candidate')
    verify_local_base(args.base_model_dir,args.online_adapter)
    knowledge=args.data_dir.parent/'demo-policies.jsonl'
    manifest=validate_data(ROOT,args.data_dir,knowledge,args.original_adapter)
    plan=json.loads((args.data_dir.parent/'closeout-plan.json').read_text(encoding='utf-8'))
    if plan['config']!=CONFIG: raise ValueError('closeout_config_changed')
    for name,digest in plan['files_sha256'].items():
        path=(args.data_dir.parent/name).resolve(strict=True)
        if not path.is_relative_to(args.data_dir.parent.resolve()) or file_sha256(path)!=digest:
            raise ValueError('closeout_evaluation_inputs_changed:'+name)
    training_rows=rows(args.data_dir/'train.jsonl')
    random.Random(CONFIG['seed']).shuffle(training_rows)
    ordered_ids=[row['id'] for row in training_rows]
    order_hash=sha256(json.dumps(ordered_ids,separators=(',',':')).encode()).hexdigest()
    if order_hash!=plan['training_order_sha256']:
        raise ValueError('closeout_training_order_changed')
    prefix_counts=dict(Counter(row['category'] for row in training_rows[:64*8]))
    if prefix_counts!=plan['pilot_prefix_category_rows'] or len(prefix_counts)!=16:
        raise ValueError('bounded_pilot_does_not_cover_all_categories')
    screen=json.loads((args.screen_output/'summary.json').read_text(encoding='utf-8'))
    if screen.get('complete') is not True or set(screen['stage_summaries'])!={'original','online'}:
        raise ValueError('complete_paired_nine_tool_screen_required')
    if screen['protocol'].get('task_plan',{}).get('selected_tasks')!=60:
        raise ValueError('screen_must_have_60_tasks_per_stage')
    for label,expected in (('original',FILE_HASHES['adapter_model.safetensors']),('online',ONLINE_SHA)):
        if screen['protocol'].get('local_adapter_sha256',{}).get(label)!=expected:
            raise ValueError('screen_adapter_identity_changed')
    saved_smoke=None
    if args.resume_after_smoke:
        from ecommerce_customer_service.serving.closeout_recovery import validate_saved_smoke
        saved_smoke=validate_saved_smoke(ROOT,args.data_dir,args.output,args.online_adapter)
        if args.trainer_script.resolve()==(ROOT/'scripts/train_sft.py').resolve():
            raise ValueError('recovery_requires_separate_corrected_trainer')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/('recovery-preflight.json' if args.resume_after_smoke else 'preflight.json'),{'cpu_valid':True,'config':CONFIG,'data_manifest_sha256':file_sha256(args.data_dir/'manifest.json'),
        'screen_summary_sha256':file_sha256(args.screen_output/'summary.json'),
        'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','peft','bitsandbytes')},
        'execute_requested':args.execute,'test_opened':False,'saved_smoke_audit':saved_smoke,
        'trainer_script':str(args.trainer_script),'trainer_sha256':file_sha256(args.trainer_script)})
    if not args.execute:
        print('CLOSEOUT CPU PREFLIGHT PASSED; no GPU/training/test started.',flush=True);return 0
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',PYTHONUTF8='1')
    phase='baseline'
    try:
        def evaluate(label,adapter,suite):
            gpu_free()
            child([ROOT/'scripts/evaluate_closeout_adapter.py','--adapter-dir',adapter,'--original-adapter-dir',args.original_adapter,
                '--base-model-dir',args.base_model_dir,'--knowledge-file',knowledge,
                '--cases',args.data_dir.parent/(suite+'-dev-cases.jsonl'),'--output',args.output/(suite+'-'+label)],
                args.output/(suite+'-'+label+'.log'))
        if args.resume_after_smoke:
            print('REUSED: completed business/knowledge baselines and audited four-step smoke. No model calls repeated.',flush=True)
        else:
            for suite in ('business','knowledge'):
                print('STAGE 1/5 old online '+suite+' five-tool dev',flush=True)
                evaluate('baseline',args.online_adapter,suite)
        for label,smoke in (('smoke',True),('pilot',False)):
            if smoke and args.resume_after_smoke: continue
            phase=label;gpu_free()
            print('STAGE '+('2' if smoke else '3')+'/5 '+label+' (fresh optimizer from saved online weights)',flush=True)
            target=args.output/label
            if target.exists(): raise FileExistsError('no_repeat_training')
            child([args.trainer_script,*training_arguments(args.data_dir,args.base_model_dir,args.online_adapter,target,smoke=smoke)],args.output/(label+'.log'))
            metrics=json.loads((target/'metrics.json').read_text(encoding='utf-8'))
            if (metrics['optimizer_steps']!=(4 if smoke else 64) or metrics['train_skipped'] or metrics['dev_skipped']
                    or metrics['adapter_initialization']!='existing_adapter' or metrics['resume_checkpoint'] is not None
                    or Path(metrics['initial_adapter']).resolve()!=args.online_adapter.resolve()
                    or set(metrics['trainable_parameter_elements_by_dtype'])!={'torch.float32'}
                    or metrics.get('training_rows_shuffled_once') is not True
                    or metrics.get('training_order_sha256')!=order_hash):
                raise ValueError('training_gate_failed')
            if any(not math.isfinite(row[key]) for row in metrics['trace'] for key in ('train_loss','gradient_norm')):
                raise ValueError('nonfinite_training')
            if not all(math.isfinite(metrics[key]) for key in ('initial_dev_loss','final_dev_loss')):
                raise ValueError('nonfinite_validation_loss')
            delta=parameter_delta(args.online_adapter/'adapter_model.safetensors',target/'adapter_model.safetensors')
            write(target/'weight-update.json',delta)
        candidate=args.output/'pilot'
        files={path.name:file_sha256(path) for path in candidate.iterdir()
               if path.is_file() and path.suffix in {'.json','.safetensors','.txt','.jinja'} and path.name!='metrics.json'}
        write(candidate/'current-instance-artifact.json',{'base_revision':BASE_REVISION,'source_adapter_sha256':ONLINE_SHA,
            'original_v18_sha256':FILE_HASHES['adapter_model.safetensors'],'files_sha256':files,
            'training_config':CONFIG,'data_manifest_sha256':file_sha256(args.data_dir/'manifest.json'),'formal_test_performed':False})
        verify_online_adapter(candidate)
        phase='candidate_dev'
        comparisons={}
        for suite in ('business','knowledge'):
            print('STAGE 4/5 pilot '+suite+' paired dev',flush=True)
            evaluate('candidate',candidate,suite)
            comparisons[suite]=compare_runs(args.output/(suite+'-baseline'),args.output/(suite+'-candidate'))
            write(args.output/(suite+'-comparison.json'),comparisons[suite])
        selection=choose_adapter(comparisons)
        selected=candidate if selection['selected']=='pilot' else args.online_adapter
        selection.update(selected_adapter_dir=str(selected.resolve()),
            selected_adapter_sha256=verify_online_adapter(selected)['adapter_sha256'],
            knowledge_sha256=file_sha256(knowledge),sealed_test_sha256=plan['sealed_test_sha256'],
            protocol_config=CONFIG,paired_comparisons_sha256={suite:file_sha256(args.output/(suite+'-comparison.json')) for suite in comparisons})
        write(args.output/'selection.json',selection)
        print('DEV SELECTION '+json.dumps(selection,ensure_ascii=False),flush=True)
        phase='sealed_test';gpu_free()
        print('STAGE 5/5 selected adapter sealed synthetic acceptance ONCE',flush=True)
        child([ROOT/'scripts/evaluate_closeout_sealed_test.py','--selection',args.output/'selection.json',
            '--ledger',args.output/'sealed-test-claim.json','--adapter-dir',selected,'--original-adapter-dir',args.original_adapter,
            '--base-model-dir',args.base_model_dir,'--knowledge-file',knowledge,
            '--cases',args.data_dir.parent/'sealed/test-cases.jsonl','--output',args.output/'sealed-test'],args.output/'sealed-test.log')
        phase='http_integration';gpu_free()
        print('STAGE HTTP inference -> webpage API -> confirmation -> restart integration',flush=True)
        serve_args=[str(ROOT/'scripts/serve_closeout_adapter.py'),'--selection',str(args.output/'selection.json'),
                    '--base-model-dir',str(args.base_model_dir)]
        with (args.output/'selected-service.log').open('x',encoding='utf-8') as service_log:
            service=subprocess.Popen([sys.executable,'-u',*serve_args],cwd=ROOT,stdout=service_log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+120
                while True:
                    if service.poll() is not None: raise RuntimeError('selected_service_start_failed')
                    try:
                        request=Request('http://127.0.0.1:8000/healthz',headers={'Authorization':'Bearer '+os.environ.get('MODEL_API_KEY','')})
                        with urlopen(request,timeout=2) as response: health=json.load(response)
                        if health.get('adapter_sha256')!=selection['selected_adapter_sha256']:
                            raise ValueError('selected_inference_identity_mismatch')
                        break
                    except (OSError,TimeoutError):
                        if time.monotonic()>=deadline: raise RuntimeError('selected_service_start_timeout')
                        time.sleep(1)
                child([ROOT/'scripts/check_closeout_http.py','--selection',args.output/'selection.json',
                    '--knowledge-file',knowledge,'--output',args.output/'http-integration'],args.output/'http-integration.log')
            finally:
                service.terminate()
                try: service.wait(timeout=20)
                except subprocess.TimeoutExpired: service.kill();service.wait()
        write(args.output/'completed.json',{'complete':True,'training_rounds':1,'grpo_started':False,
            'sealed_synthetic_test_performed':True,'production_promoted':False,'http_integration_pending':False,
            'browser_visual_pending':True,'backup_pending':True})
        print('GPU ACCEPTANCE COMPLETE. Keep this instance: browser visual check and verified download remain.',flush=True)
        if args.serve_selected_after:
            print('Keeping dev-selected inference in this SSH foreground for browser validation.',flush=True)
            os.execv(sys.executable,[sys.executable,'-u',*serve_args])
        return 0
    except BaseException as error:
        write(args.output/('recovery-failure.json' if args.resume_after_smoke else 'failure.json'),{'phase':phase,'exception_type':type(error).__name__,'message':str(error),
            'no_automatic_repeat':True,'original_weights_preserved':True})
        traceback.print_exc();return 2


if __name__=='__main__': raise SystemExit(main())
