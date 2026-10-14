"""Validate by default; --execute runs evaluation/sampling only, sequentially on one GPU."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.dual_track import commands, validate, write_json
from ecommerce_customer_service.local_hf import sha256_file
from ecommerce_customer_service.serving.artifacts import verify_local_base
from ecommerce_customer_service.serving.repair_experiment import compare_runs


def gpu_preflight(track):
    command = ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits']
    status = subprocess.run(command, capture_output=True, text=True, check=True)
    uuid, free = [field.strip() for field in status.stdout.splitlines()[0].split(',')]
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                               capture_output=True, text=True, check=True)
    if any(line.split(',')[0].strip() == uuid for line in processes.stdout.splitlines()):
        raise RuntimeError('GPU_0_has_compute_processes; stop_your_service_manually_before_evaluation')
    required = 18000 if track == 'agent' else 12000
    if int(free) < required:
        raise RuntimeError(f'insufficient_free_GPU_memory:{free}<{required}_MiB')
    return {'gpu_uuid': uuid, 'free_MiB': int(free)}


def child(command, log, env):
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f'child_failed:{code}:{log.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--track', choices=('agent', 'rl'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original-adapter', type=Path, default=ROOT / 'models/v18-sft-verified')
    parser.add_argument('--candidate-adapter', type=Path, default=ROOT / 'outputs/online-repair-backup-20261013-r1/extracted/outputs/online-repair-pilot-r1/pilot')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/v18-train-dev-r1/tasks.jsonl')
    parser.add_argument('--base-model-dir', type=Path)
    parser.add_argument('--hf-home', type=Path, help='RL named-revision loader uses this existing HF cache; no downloads')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {None, '', '0'}:
        raise ValueError('this_runner_requires_physical_GPU_0')
    if args.output.exists():
        raise FileExistsError('new_check_output_required')
    validate(ROOT, args.plan_dir, manifest=args.manifest, original=args.original_adapter,
             candidate=args.candidate_adapter, track=args.track)
    if args.execute and args.track == 'agent' and not args.base_model_dir:
        raise ValueError('agent_execution_requires_base_model_dir')
    if args.base_model_dir:
        verify_local_base(args.base_model_dir, args.original_adapter)
        if args.track == 'agent':
            verify_local_base(args.base_model_dir, args.candidate_adapter)
    argv = commands(ROOT, args.plan_dir, args.output, python=sys.executable, manifest=args.manifest,
                    original=args.original_adapter, candidate=args.candidate_adapter,
                    base=args.base_model_dir or Path('BASE_SNAPSHOT_REQUIRED'))[args.track]
    args.output.mkdir(parents=True)
    receipt = {'schema_version': 'dual_track_check_receipt_v1', 'track': args.track,
        'plan_sha256': sha256_file(args.plan_dir / 'plan.json'), 'commands': argv,
        'cpu_preflight_passed': True, 'gpu_execution_requested': args.execute,
        'gpu_execution_performed': False, 'complete': False, 'training_performed': False,
        'formal_test_performed': False, 'automatic_promotion': False,
        'started_at': datetime.now(timezone.utc).isoformat()}
    write_json(args.output / 'preflight.json', receipt)
    if not args.execute:
        print('CPU PREFLIGHT PASSED. No GPU, evaluation, sampling or training was started.')
        return 0
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='0')
    if args.hf_home:
        env['HF_HOME'] = str(args.hf_home.resolve())
    try:
        receipt['gpu_checks'] = []
        for index, command in enumerate(argv):
            receipt['gpu_checks'].append(gpu_preflight(args.track))
            receipt['gpu_execution_performed'] = True
            child(command, args.output / f'job-{index}.log', env)
        if args.track == 'agent':
            for suite in ('core', 'knowledge'):
                comparison = compare_runs(args.output / f'{suite}-baseline', args.output / f'{suite}-candidate')
                write_json(args.output / f'{suite}-comparison.json', comparison)
            receipt['metrics_must_remain_separate'] = True
        else:
            child([sys.executable, str(ROOT / 'scripts/report_v18_reward_signal.py'),
                   '--rollouts', str(args.output / 'sampling/rollouts.jsonl'),
                   '--output', str(args.output / 'reward-signal.json')], args.output / 'reward-report.log', env)
            receipt['human_review_required'] = True
        receipt['complete'] = True
    except BaseException as error:
        receipt['error'] = str(error)
        raise
    finally:
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'execution.json', receipt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
