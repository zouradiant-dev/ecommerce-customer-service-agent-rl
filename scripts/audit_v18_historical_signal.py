"""Verify the preserved October 5 sampling archive and replay its tools on CPU."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.grpo_v18 import read_rows, replay_v18_row, select_v18_train_tasks
from ecommerce_customer_service.local_hf import sha256_file
from ecommerce_customer_service.reward_signal import describe

ARCHIVE_SHA = '3bf624cdc283d2d27baa2e056ff89bf5d8a1b051879a5ce3ba7c9bfda61a08fd'
PREFIX = 'outputs/v18-grpo-calibration-r1/'
PINNED = {
    'protocol.json': 'edfe673b2520f5da3e67fdc8d0d964df6aef39cfd36f4cbde98a71d961cfcc63',
    'rollouts.jsonl': '6d942419197e4f8aaabea6e0f9cd50740c4a220eb1121a47976c7df382725290',
    'review-pending.jsonl': '9261cf8e840124f7bb4228b4fe5230cf8b76aae31aed726e7c884fbf8a792cd1',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('new_audit_output_required')
    if sha256_file(args.archive) != ARCHIVE_SHA:
        raise ValueError('historical_archive_hash_mismatch')
    payloads = {}
    with tarfile.open(args.archive, 'r:gz') as archive:
        for member in archive:
            if member.name not in {PREFIX + name for name in PINNED}:
                continue
            name = member.name[len(PREFIX):]
            if name in payloads or not member.isfile() or member.size > 16 * 1024 * 1024:
                raise ValueError('invalid_historical_member')
            payloads[name] = archive.extractfile(member).read()
    if set(payloads) != set(PINNED) or any(sha256(raw).hexdigest() != PINNED[name]
                                         for name, raw in payloads.items()):
        raise ValueError('historical_member_hash_mismatch')
    protocol = json.loads(payloads['protocol.json'])
    selected, plan = select_v18_train_tasks(args.manifest, tasks_per_scenario=2,
                                           scenarios=protocol['scenarios'])
    if plan['task_ids'] != protocol['task_ids'] or plan['database_sha256'] != protocol['database_sha256']:
        raise ValueError('historical_task_selection_mismatch')
    args.output.mkdir(parents=True)
    for name, raw in payloads.items():
        (args.output / name).write_bytes(raw)
    rows = read_rows(args.output / 'rollouts.jsonl')
    report = describe(rows)
    if report['tasks'] != 20 or report['rollouts'] != 80:
        raise ValueError('historical_rollout_count_mismatch')
    task_paths = {task.task_id: path for path, task in selected}
    replay_failures = []
    for row in rows:
        try:
            replay_v18_row(row, task_paths[row['task_id']])
        except Exception as error:
            replay_failures.append({'rollout_id': row['rollout_id'], 'error': str(error)})
    source_changes = [name for name, digest in protocol['source_sha256'].items()
                      if not (ROOT / name).is_file() or sha256_file(ROOT / name) != digest]
    report.update(archive_sha256=ARCHIVE_SHA, files_sha256=PINNED,
                  historical_task_ids=plan['task_ids'], cpu_replayed=len(rows),
                  replay_failures=replay_failures, source_changes_since_sampling=source_changes,
                  current_gpu_reproduction=False, human_review_complete=False,
                  current_training_authorization=False,
                  replay_source_sha256={p.relative_to(ROOT).as_posix(): sha256_file(p)
                                       for p in sorted((ROOT / 'src').rglob('*.py'))})
    (args.output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('tasks', 'rollouts', 'cpu_replayed', 'replay_failures')}, ensure_ascii=False))
    return 0 if not replay_failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
