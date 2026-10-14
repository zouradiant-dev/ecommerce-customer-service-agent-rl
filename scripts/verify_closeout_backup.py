"""Verify every archive member, restore adapters and write local artifact paths."""
from pathlib import Path,PurePosixPath
from hashlib import sha256
import argparse
import json
import shutil
import sqlite3
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.serving.closeout_experiment import verify_online_adapter
from ecommerce_customer_service.serving.artifacts import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-dir',type=Path,required=True)
    args=p.parse_args()
    base=args.download_dir.resolve(strict=True)
    archive=base/'current-instance-closeout-20261014-r1-results.tar.gz'
    expected=Path(str(archive)+'.sha256').read_text().split()[0]
    if len(expected)!=64 or file_sha256(archive)!=expected: raise ValueError('backup_archive_hash_mismatch')
    destination=base/'extracted'
    with tarfile.open(archive) as bundle:
        members=bundle.getmembers()
        if len(members)>15000 or sum(member.size for member in members)>1024**3: raise ValueError('backup_size_limit')
        manifest=json.load(bundle.extractfile('export-receipt.json'))
        if manifest.get('schema_version')!='current_instance_closeout_export_v1': raise ValueError('wrong_backup_schema')
        seen=set()
        for member in members:
            name=PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or '..' in name.parts or '\\' in member.name
                    or member.name in seen): raise ValueError('unsafe_backup_member')
            seen.add(member.name)
            if member.name=='export-receipt.json': continue
            h=sha256()
            with bundle.extractfile(member) as source:
                for block in iter(lambda:source.read(1024*1024),b''): h.update(block)
            if h.hexdigest()!=manifest['files_sha256'].get(member.name): raise ValueError('backup_member_mismatch:'+member.name)
        if seen!={'export-receipt.json',*manifest['files_sha256']}: raise ValueError('backup_member_set_mismatch')
        destination.mkdir(exist_ok=True)
        if destination.is_symlink(): raise ValueError('linked_extraction_destination')
        for member in members:
            path=destination.joinpath(*PurePosixPath(member.name).parts)
            if not path.resolve().is_relative_to(destination.resolve()): raise ValueError('existing_extraction_parent_escapes_destination')
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                expected_member=(manifest['files_sha256'].get(member.name) if member.name!='export-receipt.json'
                                 else sha256(bundle.extractfile(member).read()).hexdigest())
                if path.is_symlink() or file_sha256(path)!=expected_member: raise ValueError('existing_extraction_changed_no_overwrite')
                continue
            with bundle.extractfile(member) as source,path.open('xb') as target: shutil.copyfileobj(source,target)
    experiment=destination/'outputs/closeout'
    selection=json.loads((experiment/'selection.json').read_text(encoding='utf-8'))
    candidate=experiment/'pilot'
    candidate_identity=verify_online_adapter(candidate)
    selected=(candidate if selection['selected']=='pilot' else
              ROOT/'outputs/online-repair-backup-20261013-r1/extracted/outputs/online-repair-pilot-r1/pilot')
    identity=verify_online_adapter(selected)
    if identity['adapter_sha256']!=selection['selected_adapter_sha256']: raise ValueError('selected_weights_not_restored')
    knowledge=destination/'closeout-inputs/demo-policies.jsonl'
    if file_sha256(knowledge)!=selection['knowledge_sha256']: raise ValueError('restored_knowledge_changed')
    databases=[path for path in experiment.rglob('*.sqlite') if path.is_file()]
    for path in databases:
        database=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        try:
            if database.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:
                raise ValueError('restored_sqlite_integrity_failed:'+str(path))
        finally: database.close()
    receipt={'schema_version':'verified_closeout_backup_v1','archive_sha256':expected,
        'verified_members':len(members),'selected_adapter_dir':str(selected),'selected_identity':identity,
        'candidate_identity':candidate_identity,'selection_file':str(experiment/'selection.json'),
        'knowledge_file':str(knowledge),'experiment_dir':str(experiment),'source_repo':manifest['source_repo'],
        'smoke_model_sha256':file_sha256(experiment/'smoke/adapter_model.safetensors'),
        'sqlite_databases_verified':len(databases),
        'gpu_used':False,'browser_visual_checked':False,'base_weights_or_image_restored':False}
    (base/'verification-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
