"""Freeze CPU-validated v18 dev screening inputs; contains no model weights."""
from pathlib import Path
from hashlib import sha256
import json
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.evaluation_v18 import select_v18_dev_tasks


def main():
    target = ROOT / 'packages/current-instance-screen-20261014-r1.tar.gz'
    if target.exists():
        raise FileExistsError(target)
    manifest = ROOT / 'data/v18-train-dev-r1/tasks.jsonl'
    selected, selection = select_v18_dev_tasks(manifest, max_tasks_per_scenario=2)
    if selection['selected_tasks'] != 60 or len(selection['scenario_counts']) != 30:
        raise ValueError('screen_must_have_30_scenarios_times_2')
    files = {manifest}
    # The evaluator validates every frozen dev payload before selecting the 60 rows.
    for row in map(json.loads, manifest.read_text(encoding='utf-8').splitlines()):
        if row['split'] == 'dev':
            files.update(manifest.parent / row[key] for key in ('task_file', 'database'))
    for folder in ('src', 'scripts', 'configs'):
        files.update(path for path in (ROOT / folder).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts
                     and path.suffix in {'.py', '.json', '.yaml', '.yml'})
    files.add(ROOT / 'packages/featurize-46739-screen.sh')
    receipt = {'schema_version': 'current_instance_screen_v1', 'selection': selection,
               'original_adapter_sha256': 'f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c',
               'online_adapter_sha256': '3a5dd3e29427d2bf5fc89e813ff16dbd30bdca6339ad2fd4169b6c2b6f024764',
               'training_started': False, 'test_included': False,
               'files_sha256': {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
                                for path in sorted(files)}}
    receipt_path = ROOT / 'packages/current-instance-screen-20261014-r1.json'
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with tarfile.open(target, 'w:gz') as archive:
        for path in sorted(files | {receipt_path}):
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    digest = sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + '.sha256').write_text(digest + '  ' + target.name + '\n', encoding='ascii')
    print(json.dumps({'archive': str(target), 'sha256': digest, 'bytes': target.stat().st_size,
                      'dev_selected_per_model': len(selected), 'files': len(files)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
