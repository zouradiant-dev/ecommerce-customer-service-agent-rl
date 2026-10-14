"""Freeze two separate dev protocols; never authorize training or promotion."""
import json
from pathlib import Path

from .grpo_v18 import select_v18_train_tasks, EXPECTED_SFT_ADAPTER_SHA256, BASE_REVISION
from .local_hf import sha256_file
from .serving.artifacts import verify_adapter
from .serving.knowledge import KnowledgeIndex, augment_policy
from .serving.knowledge_cases import cases as knowledge_cases
from .serving.policy import get_policy

REPAIR_SHA = '3a5dd3e29427d2bf5fc89e813ff16dbd30bdca6339ad2fd4169b6c2b6f024764'
SCENARIOS = ('cross_order_item_mismatch', 'duplicate_refund_request', 'mixed_order_bulk_cancel',
             'multi_order_disambiguation', 'order_not_found', 'partial_quantity_refund',
             'policy_bypass_request', 'refund_policy_tool_transient_error', 'refund_quantity_exceeds',
             'update_address')
SELECTION_SEED = 20261014


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_rows(path, values):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def runtime_hashes(root):
    files = sorted([*(root / 'src').rglob('*.py'), *(root / 'scripts').rglob('*.py'),
                    root / 'configs/v18-reward.json'])
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in files}


def adapter_files(directory, *, candidate=False):
    if not candidate:
        return verify_adapter(directory)['files_sha256']
    receipt = read_json(directory / 'repair-artifact.json')
    if (receipt.get('source_adapter_sha256') != EXPECTED_SFT_ADAPTER_SHA256
            or receipt.get('base_revision') != BASE_REVISION
            or receipt.get('formal_test_performed') is not False
            or receipt['files_sha256'].get('adapter_model.safetensors') != REPAIR_SHA):
        raise ValueError('unexpected_online_repair_candidate')
    for name, digest in receipt['files_sha256'].items():
        if Path(name).name != name or (directory / name).is_symlink() or sha256_file(directory / name) != digest:
            raise ValueError('candidate_artifact_hash_mismatch')
    return {**receipt['files_sha256'], 'repair-artifact.json': sha256_file(directory / 'repair-artifact.json')}


def freeze(root, output, *, manifest, original, candidate, dev_cases, historical_audit, git_head=None):
    if output.exists():
        raise FileExistsError('new_dual_track_plan_required')
    original_hashes = adapter_files(original)
    candidate_hashes = adapter_files(candidate, candidate=True)
    audit = read_json(historical_audit)
    if (audit.get('archive_sha256') != '3bf624cdc283d2d27baa2e056ff89bf5d8a1b051879a5ce3ba7c9bfda61a08fd'
            or audit.get('cpu_replayed') != 80 or audit.get('replay_failures') != []):
        raise ValueError('historical_signal_not_verified')
    excluded = audit['historical_task_ids']
    _, selection = select_v18_train_tasks(manifest, tasks_per_scenario=2, scenarios=SCENARIOS,
                                          selection_seed=SELECTION_SEED, excluded_task_ids=excluded)
    core = (read_rows(root / 'data/online-agent-cases-v1.jsonl')
            + read_rows(root / 'data/online-binding-dev-v2.jsonl') + read_rows(dev_cases))
    if len(core) != 57 or len({case['case_id'] for case in core}) != 57 or any(case['split'] != 'dev' for case in core):
        raise ValueError('unexpected_online_core_dev_suite')
    KnowledgeIndex(root / 'data/demo-policies.jsonl')
    policy = get_policy('binding_guard_v4')
    augmented = augment_policy(policy)
    output.mkdir(parents=True)
    write_rows(output / 'agent-core-dev.jsonl', core)
    write_rows(output / 'agent-knowledge-dev.jsonl', knowledge_cases())
    (output / 'demo-policies.jsonl').write_bytes((root / 'data/demo-policies.jsonl').read_bytes())
    write_json(output / 'rl-excluded-task-ids.json', excluded)
    write_json(output / 'historical-signal-audit.json', audit)
    plan = {'schema_version': 'dual_track_dev_plan_v1', 'base_revision': BASE_REVISION,
        'git_head_at_freeze': git_head, 'working_tree_bytes_bound_by_source_sha256': True,
        'source_sha256': runtime_hashes(root),
        'payload_sha256': {path.name: sha256_file(path) for path in sorted(output.iterdir())},
        'artifacts': {'original': original_hashes, 'candidate': candidate_hashes},
        'agent': {'policy_profile': policy.name, 'policy': policy.metadata(),
                  'knowledge_policy': augmented.metadata(), 'core_cases': 57, 'knowledge_cases': 6,
                  'generation': {'precision': 'bf16', 'do_sample': False, 'max_new_tokens': 512,
                                 'context_budget': 8192, 'recent_turns': 6},
                  'historically_inspected_dev': True, 'independent_test_claim': False,
                  'demo_policy_only': True, 'citation_check_is_not_semantic_entailment': True},
        'rl': {**selection, 'trials_per_task': 4, 'rollouts': 80, 'temperature': 0.8, 'seed': 1807,
               'max_steps': 20, 'max_tokens': 512, 'max_input_tokens': 8192,
               'precision': 'nf4_double_quant', 'policy_controller': None,
               'selection_reason': 'seeded_balanced_train_selection_excluding_historical_20',
               'proven_harder': False, 'new_tasks_are_heldout_test': False},
        'training_authorized': False, 'automatic_promotion': False, 'gpu_execution_performed': False,
        'limitations': ['Two protocols have different tool surfaces and precision; do not pool their metrics.',
                        'Online candidate was trained for four tools; the fifth knowledge tool is unverified.',
                        'A source change requires freezing a new plan. No online formal test is consumed.']}
    write_json(output / 'plan.json', plan)
    return plan


def validate(root, plan_dir, *, manifest, original, candidate, track):
    plan = read_json(plan_dir / 'plan.json')
    if plan.get('schema_version') != 'dual_track_dev_plan_v1' or plan.get('training_authorized') is not False:
        raise ValueError('invalid_dual_track_plan')
    if (plan.get('base_revision') != BASE_REVISION
            or plan.get('agent', {}).get('policy_profile') != 'binding_guard_v4'
            or plan.get('rl', {}).get('selection_seed') != SELECTION_SEED
            or plan['rl']['scenarios'] != sorted(SCENARIOS)):
        raise ValueError('unsupported_dual_track_protocol')
    if runtime_hashes(root) != plan['source_sha256']:
        raise ValueError('frozen_runtime_changed; create_a_new_plan')
    for name, digest in plan['payload_sha256'].items():
        if Path(name).name != name or sha256_file(plan_dir / name) != digest:
            raise ValueError('frozen_payload_changed')
    if adapter_files(original) != plan['artifacts']['original']:
        raise ValueError('original_adapter_changed')
    if track == 'agent' and adapter_files(candidate, candidate=True) != plan['artifacts']['candidate']:
        raise ValueError('candidate_adapter_changed')
    if track == 'rl':
        _, selected = select_v18_train_tasks(manifest, tasks_per_scenario=plan['rl']['tasks_per_scenario'],
            scenarios=plan['rl']['scenarios'], selection_seed=plan['rl']['selection_seed'],
            excluded_task_ids=plan['rl']['excluded_task_ids'])
        if any(selected[key] != plan['rl'][key] for key in ('manifest_sha256', 'task_ids', 'database_sha256')):
            raise ValueError('rl_task_selection_changed')
    return plan


def commands(root, plan_dir, output, *, python, manifest, original, candidate, base):
    """Return argv lists; no shell interpolation and no optimizer commands."""
    agent = []
    for suite in ('core', 'knowledge'):
        for stage, adapter in (('baseline', original), ('candidate', candidate)):
            command = [str(python), '-u', str(root / 'scripts/evaluate_online_local_adapter.py'),
                '--base-model-dir', str(base), '--adapter-dir', str(adapter), '--original-adapter-dir', str(original),
                '--cases', str(plan_dir / f'agent-{suite}-dev.jsonl'), '--output', str(output / f'{suite}-{stage}'),
                '--stage', stage, '--policy-profile', 'binding_guard_v4']
            if suite == 'knowledge':
                command += ['--knowledge-file', str(plan_dir / 'demo-policies.jsonl')]
            agent.append(command)
    rl = [str(python), '-u', str(root / 'scripts/sample_grpo_v18_calibration.py'), '--manifest', str(manifest),
          '--adapter', str(original), '--output', str(output / 'sampling'), '--selection-seed', str(SELECTION_SEED),
          '--exclude-task-ids', str(plan_dir / 'rl-excluded-task-ids.json'), '--tasks-per-scenario', '2',
          '--trials-per-task', '4', '--temperature', '0.8', '--seed', '1807', '--max-steps', '20',
          '--max-tokens', '512', '--max-input-tokens', '8192']
    for scenario in SCENARIOS:
        rl += ['--scenario', scenario]
    return {'agent': agent, 'rl': [rl]}
