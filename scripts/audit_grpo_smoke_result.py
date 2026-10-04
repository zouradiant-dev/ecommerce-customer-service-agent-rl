"""Read-only smoke audit and isolated CPU tool replay; never change rewards or weights."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.local_hf import sha256_file
from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.online_env import EcommerceCustomerServiceEnv, _final_response
from ecommerce_customer_service.tasks import load_task_paths


def tensor_comparison(left, right):
    """Compare exact safetensors payloads without loading torch/pickle or allocating the model."""
    with left.open('rb') as a, right.open('rb') as b:
        na, nb = struct.unpack('<Q', a.read(8))[0], struct.unpack('<Q', b.read(8))[0]
        ha, hb = json.loads(a.read(na)), json.loads(b.read(nb))
        keys = set(ha) - {'__metadata__'}
        if keys != set(hb) - {'__metadata__'}:
            raise ValueError('tensor keys differ')
        changed, byte_differences, cast_tensors = [], 0, 0
        for key in sorted(keys):
            x, y = ha[key], hb[key]
            normalize = x['dtype'] == 'F32' and y['dtype'] == 'BF16'
            if x['shape'] != y['shape'] or (x['dtype'] != y['dtype'] and not normalize):
                raise ValueError('unsupported tensor dtype/shape difference: ' + key)
            cast_tensors += int(normalize)
            sa, ea = x['data_offsets']
            sb, eb = y['data_offsets']
            if ea - sa != (eb - sb) * (2 if normalize else 1):
                raise ValueError('tensor byte lengths differ')
            a.seek(8 + na + sa)
            b.seek(8 + nb + sb)
            remaining, different = ea - sa, False
            while remaining:
                count = min(remaining, 1024 * 1024)
                expected_b = count // 2 if normalize else count
                ca, cb = a.read(count), b.read(expected_b)
                if len(ca) != count or len(cb) != expected_b:
                    raise ValueError('truncated safetensors file')
                if normalize:
                    bits = array('I')
                    bits.frombytes(ca)
                    if sys.byteorder != 'little':
                        bits.byteswap()
                    # Match finite F32 -> BF16 round-to-nearest-even. Refuse
                    # nonfinite original weights instead of normalizing them away.
                    if any((value & 0x7f800000) == 0x7f800000 for value in bits):
                        raise ValueError('nonfinite original SFT weight')
                    rounded = array('H', (((value + 0x7fff + ((value >> 16) & 1)) >> 16) & 0xffff
                                          for value in bits))
                    if sys.byteorder != 'little':
                        rounded.byteswap()
                    ca = rounded.tobytes()
                if ca != cb:
                    different = True
                    byte_differences += sum(u != v for u, v in zip(ca, cb))
                remaining -= count
            if different:
                changed.append(key)
        return {'tensors': len(keys), 'changed_tensors': len(changed),
                'changed_payload_bytes': byte_differences,
                'f32_to_bf16_normalized_tensors': cast_tensors,
                'comparison_after_dtype_alignment_equal': not changed,
                'exact_payload_equal': not changed and not cast_tensors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--sft', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists() or output == run or run in output.parents:
        raise ValueError('audit must use a new directory outside the original run')
    training = run / 'training'
    preflight = json.loads((training / 'preflight.json').read_text(encoding='utf-8'))
    pilot = json.loads((args.pilot / 'protocol.json').read_text(encoding='utf-8'))
    manifest = ROOT / 'data/qwen15b-v2/tasks.jsonl'
    if sha256_file(manifest) != preflight['manifest_sha256']:
        raise ValueError('manifest changed')
    for relative in ('src/ecommerce_customer_service/communication.py',
                     'src/ecommerce_customer_service/verifier.py',
                     'src/ecommerce_customer_service/tools.py'):
        if sha256_file(ROOT / relative) != preflight['source_sha256'][relative]:
            raise ValueError('frozen tool/reward source changed: ' + relative)
    if sha256_file(args.sft / 'adapter_model.safetensors') != preflight['adapter_sha256']:
        raise ValueError('initial SFT changed')
    rows = [json.loads(line) for line in (training / 'online_rewards.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    paths = {}
    for path in load_task_paths(manifest, split='train'):
        task = TaskSpec.from_file(path)
        if task.task_id in preflight['smoke_task_ids']:
            if sha256_file(task.database_path) != pilot['source_database_sha256'][task.task_id]:
                raise ValueError('source database changed: ' + task.task_id)
            paths[task.task_id] = path
    replay, failures, batches, tasks = [], Counter(), defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows, 1):
        env = EcommerceCustomerServiceEnv(task_path=paths[row['task_id']])
        try:
            env.reset()
            for call in row['tool_executions']:
                observed = env._tools.execute(call['name'], call['arguments']).to_dict()
                if observed != call:
                    raise ValueError(f'tool replay mismatch at row {index}: {call["name"]}')
            final = _final_response(row['completion'])
            reward = env._evaluate(final)
            verifier = json.loads(json.dumps(env._last_verification.to_dict()))
            if reward != row['reward'] or verifier != row['verifier']:
                raise ValueError(f'reward/state replay mismatch at row {index}')
            replay.append({'row': index, 'task_id': row['task_id'], 'batch': row['batch'],
                           'reward': reward, 'verifier': verifier, 'final': final,
                           'tool_replay_matches': True, 'state_and_reward_replay_matches': True})
        finally:
            env._cleanup()
        failures.update(row['verifier']['failures'])
        batches[row['batch']].append(row)
        tasks[row['task_id']].append(row)
    for task_id, path in paths.items():
        if sha256_file(TaskSpec.from_file(path).database_path) != pilot['source_database_sha256'][task_id]:
            raise ValueError('audit mutated source database')
    groups = []
    for batch, batch_rows in sorted(batches.items()):
        if len(batch_rows) != 8:
            raise ValueError('incomplete generation batch')
        for offset in (0, 4):
            group = batch_rows[offset:offset + 4]
            if len({r['task_id'] for r in group}) != 1:
                raise ValueError('mixed task identities within a relative-reward group')
            values = [r['reward'] for r in group]
            groups.append({'batch': batch, 'task_id': group[0]['task_id'], 'rewards': values,
                           'mixed': len(set(values)) > 1})
    history = json.loads((training / 'checkpoint-10/trainer_state.json').read_text(encoding='utf-8'))['log_history']
    metrics = [r for r in history if 'kl' in r]
    report = {
        'method': 'original v11 replay; no rescoring or training',
        'rollouts': len(rows), 'successes': sum(r['reward'] == 1 for r in rows),
        'recorded_success_rate': sum(r['reward'] for r in rows) / len(rows),
        'groups': len(groups), 'mixed_groups': sum(g['mixed'] for g in groups),
        'mixed_batches': len({g['batch'] for g in groups if g['mixed']}),
        'kl_min': min(r['kl'] for r in metrics), 'kl_max': max(r['kl'] for r in metrics),
        'all_tool_and_reward_replays_match': True, 'source_databases_unchanged': True,
        'communication_only_zero_rows': [r['row'] for r in replay if not r['reward']
            and all(r['verifier'][k] for k in ('state_passed', 'policy_passed', 'required_tools_passed'))],
        'state_passes': sum(r['verifier']['state_passed'] for r in rows),
        'policy_passes': sum(r['verifier']['policy_passed'] for r in rows),
        'required_tools_passes': sum(r['verifier']['required_tools_passed'] for r in rows),
        'communication_passes': sum(r['verifier']['communication_passed'] for r in rows),
        'sft_vs_saved_reference': tensor_comparison(args.sft / 'adapter_model.safetensors', training / 'ref/adapter_model.safetensors'),
        'sft_vs_trained_policy': tensor_comparison(args.sft / 'adapter_model.safetensors', training / 'adapter_model.safetensors'),
        'checkpoint_vs_final_policy': tensor_comparison(training / 'checkpoint-10/adapter_model.safetensors', training / 'adapter_model.safetensors'),
        'source_files_sha256': {str(p.relative_to(run)): sha256_file(p) for p in (
            run / 'run.log', training / 'preflight.json', training / 'online_rewards.jsonl',
            training / 'gradient_audit.jsonl', training / 'smoke_complete.json',
            training / 'adapter_model.safetensors', training / 'ref/adapter_model.safetensors')},
        'per_task': {task: {'rollouts': len(values), 'successes': sum(r['reward'] == 1 for r in values)}
                     for task, values in sorted(tasks.items())},
        'failure_counts': dict(failures), 'groups_detail': groups, 'replay_rows': replay,
        'formal_rl_gate_passed': False, 'held_out_improvement_evaluated': False,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'audit.json').open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in (
        'replay_rows', 'groups_detail', 'source_files_sha256', 'failure_counts')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
