"""Freeze one bounded five-tool experiment, without model/base weights."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json
import random
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.closeout_experiment import CONFIG, ONLINE_SHA


def digest(path): return sha256(path.read_bytes()).hexdigest()


def main():
    staging=ROOT/'outputs/current-instance-closeout-inputs-20261014-r1'
    target=ROOT/'packages/current-instance-closeout-20261014-r1.tar.gz'
    if target.exists() or staging.exists(): raise FileExistsError('new_freeze_required')
    staging.mkdir()
    source=ROOT/'outputs/current-instance-five-tool-sft-20261014-r3'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['training_ready'] is not True: raise ValueError('data_not_ready')
    shutil.copytree(source,staging/'train-dev')
    (staging/'sealed').mkdir()
    sealed=ROOT/'outputs/online-repair-sft-20261014-v4-r2/test-cases.jsonl'
    if digest(sealed)!=manifest['sealed_cases_sha256']: raise ValueError('sealed_test_changed')
    shutil.copyfile(sealed,staging/'sealed/test-cases.jsonl')
    old_plan=ROOT/'outputs/dual-track-agent-rl-20261014-r1'
    for source_name,target_name in (('agent-core-dev.jsonl','business-dev-cases.jsonl'),
                                    ('agent-knowledge-dev.jsonl','knowledge-dev-cases.jsonl')):
        shutil.copyfile(old_plan/source_name,staging/target_name)
    shutil.copyfile(ROOT/'data/demo-policies.jsonl',staging/'demo-policies.jsonl')
    samples=[json.loads(line) for line in (source/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    random.Random(CONFIG['seed']).shuffle(samples)
    plan={'schema_version':'current_instance_closeout_plan_v1','config':CONFIG,'initial_online_sha256':ONLINE_SHA,
          'sealed_test_sha256':digest(sealed),'source_data_manifest_sha256':digest(source/'manifest.json'),
          'training_order_sha256':sha256(json.dumps([row['id'] for row in samples],separators=(',',':')).encode()).hexdigest(),
          'pilot_prefix_category_rows':dict(Counter(row['category'] for row in samples[:512])),
          'smoke_prefix_category_rows':dict(Counter(row['category'] for row in samples[:32])),
          'files_sha256':{path.relative_to(staging).as_posix():digest(path) for path in staging.rglob('*') if path.is_file()},
          'model_training_performed':False,'sealed_test_evaluated':False,'production_promotion_allowed':False}
    (staging/'closeout-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files={}
    for folder in ('src','scripts','configs'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.json','.yaml','.yml','.html'}:
                files[path.relative_to(ROOT).as_posix()]=path
    for path in staging.rglob('*'):
        if path.is_file(): files['closeout-inputs/'+path.relative_to(staging).as_posix()]=path
    for name in ('pyproject.toml','requirements-training.txt','requirements-inference-v18.txt',
                 'docs/current-instance-closeout-plan-20261014.zh-CN.md'):
        files[name]=ROOT/name
    receipt={'schema_version':'current_instance_closeout_bundle_v1','model_weights_included':False,
             'files_sha256':{name:digest(path) for name,path in sorted(files.items())}}
    receipt_path=staging/'bundle-receipt.json'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    files['bundle-receipt.json']=receipt_path
    with tarfile.open(target,'x:gz') as archive:
        for name,path in sorted(files.items()):
            if path.is_symlink(): raise ValueError('no_linked_bundle_source')
            archive.add(path,arcname=name,recursive=False)
    checksum=digest(target)
    target.with_suffix(target.suffix+'.sha256').write_text(checksum+'  '+target.name+'\n',encoding='ascii')
    print(json.dumps({'archive':str(target),'sha256':checksum,'bytes':target.stat().st_size,'files':len(files)},ensure_ascii=False))


if __name__=='__main__': main()
