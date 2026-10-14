"""Package frozen source/dev inputs and required train DBs; exclude all model weights."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.dual_track import read_json, read_rows, runtime_hashes, write_json
from ecommerce_customer_service.local_hf import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/v18-train-dev-r1/tasks.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + '.json').exists():
        raise FileExistsError('new_bundle_required')
    plan = read_json(args.plan_dir / 'plan.json')
    if runtime_hashes(ROOT) != plan['source_sha256']:
        raise ValueError('frozen_source_changed')
    files = {name: ROOT / name for name in plan['source_sha256']}
    for name in ('pyproject.toml', 'docs/dual-track-agent-rl-20261014.zh-CN.md'):
        files[name] = ROOT / name
    for name in ['plan.json', *plan['payload_sha256']]:
        source = args.plan_dir / name
        if name != 'plan.json' and sha256_file(source) != plan['payload_sha256'][name]:
            raise ValueError('frozen_payload_changed')
        files['experiment/' + name] = source
    if sha256_file(args.manifest) != plan['rl']['manifest_sha256']:
        raise ValueError('frozen_rl_manifest_changed')
    files['data/v18-train-dev-r1/tasks.jsonl'] = args.manifest
    # The selector revalidates ALL train rows in the requested scenarios, not just the selected 20.
    # Copy that frozen subset plus the exact original manifest; dev/test DBs are never packaged here.
    for row in read_rows(args.manifest):
        if row['split'] == 'train' and row['scenario'] in plan['rl']['scenarios']:
            task = args.manifest.parent / row['task_file']
            db = task.parent / read_json(task)['database']
            if sha256_file(task) != row['task_file_sha256'] or sha256_file(db) != row['database_sha256']:
                raise ValueError('frozen_rl_task_changed')
            for path in (task, db):
                files['data/v18-train-dev-r1/' + path.relative_to(args.manifest.parent).as_posix()] = path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'x:gz') as archive:
        for name, source in sorted(files.items()):
            if source.is_symlink():
                raise ValueError('linked_bundle_source')
            archive.add(source, arcname=name, recursive=False)
    receipt = {'bundle_sha256': sha256_file(args.output), 'bytes': args.output.stat().st_size,
               'files': len(files), 'model_weights_included': False, 'dev_test_databases_included': False,
               'files_sha256': {name: sha256_file(path) for name, path in sorted(files.items())}}
    write_json(args.output.with_suffix(args.output.suffix + '.json'), receipt)
    print(json.dumps({key: receipt[key] for key in ('bundle_sha256', 'bytes', 'files', 'model_weights_included')}))


if __name__ == '__main__':
    main()
