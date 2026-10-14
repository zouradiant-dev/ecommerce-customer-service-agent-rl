"""Export completed screening evidence only; never starts model inference."""
from pathlib import Path
from hashlib import sha256
import io
import json
import tarfile

ORIGINAL='f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c'
ONLINE='3a5dd3e29427d2bf5fc89e813ff16dbd30bdca6339ad2fd4169b6c2b6f024764'


def digest(path):
    h=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    work=Path('/home/featurize/work').resolve(strict=True)
    pointer=work/'current-instance-screen-20261014-r1.path'
    repo=Path(pointer.read_text().strip()).resolve(strict=True)
    if repo.parent!=work or not repo.name.startswith('current-instance-screen-20261014.'):
        raise ValueError('unexpected_screen_directory')
    if (repo/'screen.exit-code').read_text().strip()!='0': raise ValueError('screen_not_successfully_completed')
    result=json.loads((repo/'outputs/nine-tool-screen/summary.json').read_text())
    if result.get('complete') is not True or result['protocol']['task_plan']['selected_tasks']!=60:
        raise ValueError('incomplete_screen')
    if result['protocol']['local_adapter_sha256']!={'original':ORIGINAL,'online':ONLINE}:
        raise ValueError('screen_adapter_identity_changed')
    output=work/'current-instance-screen-20261014-r1-results.tar.gz'
    sidecar=Path(str(output)+'.sha256')
    if output.exists():
        if not sidecar.is_file() or digest(output)!=sidecar.read_text().split()[0]:
            raise ValueError('existing_export_not_verified_no_overwrite')
        print(json.dumps({'archive':str(output),'sha256':digest(output),'reused':True}));return
    files=[p for p in (repo/'outputs/nine-tool-screen').rglob('*') if p.is_file()]
    files.extend(repo/name for name in ('screen-preflight.json','screen.exit-code','screen.log',
                                      'packages/current-instance-screen-20261014-r1.json'))
    if len(files)>10000 or sum(p.stat().st_size for p in files)>512*1024**2:
        raise ValueError('screen_export_size_limit')
    if any(p.is_symlink() or not p.resolve().is_relative_to(repo) for p in files):
        raise ValueError('linked_or_external_screen_payload')
    hashes={p.relative_to(repo).as_posix():digest(p) for p in files}
    receipt=json.dumps({'schema_version':'screen_export_v1','source_repo':str(repo),
                        'files_sha256':hashes,'model_weights_included':False},indent=2).encode()
    with output.open('xb') as target,tarfile.open(fileobj=target,mode='w:gz') as bundle:
        for path in sorted(files): bundle.add(path,arcname=path.relative_to(repo).as_posix(),recursive=False)
        item=tarfile.TarInfo('export-receipt.json');item.size=len(receipt)
        bundle.addfile(item,io.BytesIO(receipt))
    if any(digest(repo/name)!=expected for name,expected in hashes.items()):
        raise ValueError('screen_changed_during_export_no_verified_sidecar_written')
    checksum=digest(output)
    sidecar.write_text(checksum+'  '+output.name+'\n',encoding='ascii')
    print(json.dumps({'archive':str(output),'sha256':checksum,'files':len(files),'bytes':output.stat().st_size}))


if __name__=='__main__': main()
