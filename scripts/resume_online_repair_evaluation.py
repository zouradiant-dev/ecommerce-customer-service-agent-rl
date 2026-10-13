"""Resume candidate evaluation after a tokenizer export check failure; never train."""
from pathlib import Path
import argparse
from hashlib import sha256
import importlib.util
import json
import os
import sys
import traceback
from uuid import uuid4

EXPECTED_OLD_ARTIFACTS_SHA256='f7402097c4c1582d7c9ec00a925d34a323310987d95fc665ee2c2c7865690e37'


def install_patch(repo: Path, source: Path, *, expected_old: str = EXPECTED_OLD_ARTIFACTS_SHA256) -> dict:
    target=repo/'src/ecommerce_customer_service/serving/artifacts.py'
    before=target.read_bytes();after=source.read_bytes()
    old,new=sha256(before).hexdigest(),sha256(after).hexdigest()
    if old!=new:
        normalized_old=sha256(before.replace(b'\r\n',b'\n')).hexdigest()
        if old!=expected_old and normalized_old!=expected_old:
            raise ValueError('unexpected_existing_artifacts_source_no_overwrite')
        backup=target.with_name('artifacts.pre-tokenizer-format-fix.py')
        if backup.exists(): raise FileExistsError('compatibility_backup_already_exists')
        with backup.open('xb') as output: output.write(before)
        temporary=target.with_name('artifacts.patch-'+uuid4().hex+'.tmp')
        with temporary.open('xb') as output: output.write(after)
        temporary.replace(target)
    return {'before_sha256':old,'after_sha256':new,'only_changed_file':'serving/artifacts.py',
            'before_lf_normalized_sha256':sha256(before.replace(b'\r\n',b'\n')).hexdigest(),
            'training_or_weights_modified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True,help='EXISTING experiment repository containing completed pilot/baseline')
    parser.add_argument('--patch-source',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--base-model-dir',type=Path,default=Path('/home/featurize/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'))
    parser.add_argument('--original-adapter-dir',type=Path,default=Path('/home/featurize/work/v18-sft-48gb-r1-fp32-grad-r2/outputs/v18-sft-full-newgpu-r1'))
    parser.add_argument('--restore-original-service',action='store_true')
    args=parser.parse_args();args.repo=args.repo.resolve(strict=True)
    job=args.repo/'outputs/online-repair-pilot-r1'
    pilot=job/'pilot';baseline=job/'baseline';data=args.repo/'experiment-data'
    metrics=json.loads((pilot/'metrics.json').read_text(encoding='utf-8'))
    before=json.loads((baseline/'summary.json').read_text(encoding='utf-8'))
    if metrics['optimizer_steps']!=64 or metrics['train_skipped'] or metrics['dev_skipped'] or not before['complete']:
        raise ValueError('completed_pilot_and_baseline_required_no_retraining')
    output=args.output or args.repo/'outputs'/('online-repair-eval-resume-'+uuid4().hex[:8])
    output=output.resolve()
    if output==pilot.resolve() or output.is_relative_to(pilot.resolve()) or output==baseline.resolve() or output.is_relative_to(baseline.resolve()):
        raise ValueError('cannot_overwrite_completed_weights_or_baseline')
    output.mkdir(parents=True,exist_ok=False)
    patch_receipt=install_patch(args.repo,args.patch_source.resolve(strict=True))
    sys.path.insert(0,str(args.repo/'src'))
    from ecommerce_customer_service.serving.repair_experiment import CONFIG, compare_runs
    spec=importlib.util.spec_from_file_location('existing_repair_pipeline',args.repo/'scripts/run_online_repair_pilot.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.write(output/'resume-receipt.json',{**patch_receipt,'existing_pilot':str(pilot),'existing_baseline':str(baseline),
        'training_steps_in_resume':0,'baseline_evaluation_repeated':False,'formal_test_performed':False})
    status=1;comparison=None;error_type=None;phase='preflight'
    try:
        helper.child([str(args.repo/'scripts/run_online_repair_pilot.py'),'--preflight-child',
            '--data-dir',str(data),'--base-model-dir',str(args.base_model_dir),'--adapter-dir',str(args.original_adapter_dir),
            '--output',str(output)],output/'preflight.log')
        phase='candidate'
        print('RESUME CANDIDATE EVALUATION ONLY; training steps=0; original baseline retained',flush=True)
        helper.child([str(args.repo/'scripts/evaluate_online_local_adapter.py'),'--stage','candidate',
            '--adapter-dir',str(pilot),'--original-adapter-dir',str(args.original_adapter_dir),
            '--base-model-dir',str(args.base_model_dir),'--cases',str(data/'paired-dev-cases.jsonl'),
            '--output',str(output/'candidate')],output/'candidate.log')
        phase='comparison'
        comparison=compare_runs(baseline,output/'candidate')
        helper.write(output/'comparison.json',comparison)
        print(json.dumps(comparison,ensure_ascii=False,indent=2),flush=True)
        status=0
    except Exception as error:
        error_type=type(error).__name__
        helper.write(output/'failure.json',{'completed':False,'exception_type':error_type,'message':str(error),
            'stage':phase,'training_steps_in_resume':0})
        traceback.print_exc()
    public={'schema_version':CONFIG['schema_version'],'experiment_complete':status==0,'candidate_promoted':False,
        'formal_test_performed':False,'comparison':comparison,'error_type':error_type,'last_started_stage':phase,
        'last_completed_stage':'comparison' if status==0 else 'pilot','training_steps_in_resume':0,
        'compatibility_patch':patch_receipt}
    for name in ('smoke','pilot'):
        values=json.loads((job/name/'metrics.json').read_text(encoding='utf-8'))
        public[name]={key:values[key] for key in ('optimizer_steps','train_rows','dev_rows','train_skipped',
            'dev_skipped','initial_dev_loss','final_dev_loss','elapsed_seconds','max_memory_gib')}
        public[name]['weight_update']=json.loads((job/name/'weight-update.json').read_text(encoding='utf-8'))
    helper.write(output/'public-experiment-report.json',public)
    print('RESUME OUTPUT: '+str(output),flush=True)
    if args.restore_original_service and (output/'preflight.json').exists():
        os.execv(sys.executable,[sys.executable,'-u',str(args.repo/'scripts/serve_v18_sft.py'),
            '--adapter-dir',str(args.original_adapter_dir),'--base-model-dir',str(args.base_model_dir),'--precision','bf16',
            '--diagnostics-dir',str(output/'restored-service-diagnostics'),'--capture-invalid-output',
            '--repair-experiment-report',str(output/'public-experiment-report.json')])
    return status


if __name__=='__main__': raise SystemExit(main())
