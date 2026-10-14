"""Export closed experiment evidence and root adapters while inference stays live."""
from pathlib import Path
from hashlib import sha256
import io
import json
import tarfile


def digest(path):
    h=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def export(work):
    work=Path(work).resolve(strict=True)
    root=Path((work/'current-instance-closeout-20261014-r1.path').read_text().strip()).resolve(strict=True)
    if root.parent!=work or not root.name.startswith('current-instance-closeout-20261014.'):
        raise ValueError('unexpected_closeout_directory')
    result=root/'outputs/closeout'
    completed=json.loads((result/'completed.json').read_text())
    test=json.loads((result/'sealed-test/summary.json').read_text())
    http=json.loads((result/'http-integration/report.json').read_text())
    if (completed.get('complete') is not True or test.get('complete') is not True or http.get('passed') is not True):
        raise ValueError('closeout_not_complete_or_http_not_passed')
    selection=json.loads((result/'selection.json').read_text())
    candidate=result/'pilot'
    receipt=json.loads((candidate/'current-instance-artifact.json').read_text())
    for name,expected in receipt['files_sha256'].items():
        if Path(name).name!=name or digest(candidate/name)!=expected: raise ValueError('candidate_artifact_changed:'+name)
    candidate_sha=digest(candidate/'adapter_model.safetensors')
    if selection['selected']=='pilot' and selection['selected_adapter_sha256']!=candidate_sha:
        raise ValueError('selected_adapter_changed')
    output=work/'current-instance-closeout-20261014-r1-results.tar.gz'
    sidecar=Path(str(output)+'.sha256')
    if output.exists():
        if not sidecar.is_file() or digest(output)!=sidecar.read_text().split()[0]:
            raise ValueError('existing_unverified_export_no_overwrite')
        return {'archive':str(output),'sha256':digest(output),'bytes':output.stat().st_size,'reused':True}
    files=[]
    for folder in ('outputs/closeout','closeout-inputs','src','scripts','configs'):
        for path in (root/folder).rglob('*'):
            if not path.is_file(): continue
            relative=path.relative_to(root)
            if ('selected-service-diagnostics' not in relative.parts and path.name.endswith('-wal')
                    and path.stat().st_size):
                raise ValueError('closed_evidence_has_uncheckpointed_wal:'+str(relative))
            if ('__pycache__' in relative.parts or 'selected-service-diagnostics' in relative.parts
                    or any(part.startswith('checkpoint-') for part in relative.parts)
                    or path.suffix in {'.pt','.pyc'} or path.name.endswith(('-wal','-shm'))):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root): raise ValueError('external_or_linked_backup_source')
            files.append(path)
    for name in ('bundle-receipt.json','shuffle-recovery-r1-installation.json','pyproject.toml',
                 'requirements-training.txt','requirements-inference-v18.txt'):
        path=root/name
        if path.is_file(): files.append(path)
    if len(files)>15000 or sum(path.stat().st_size for path in files)>1024**3:
        raise ValueError('closeout_export_size_limit')
    required={result/'pilot/adapter_model.safetensors',result/'smoke/adapter_model.safetensors',
              result/'selection.json',result/'sealed-test-claim.json',result/'recovery-preflight.json'}
    if not required<=set(files): raise ValueError('required_final_evidence_missing')
    hashes={path.relative_to(root).as_posix():digest(path) for path in files}
    manifest={'schema_version':'current_instance_closeout_export_v1','source_repo':str(root),
              'files_sha256':hashes,'root_smoke_and_pilot_weights_included':True,
              'selected_adapter_sha256':selection['selected_adapter_sha256'],
              'pilot_adapter_sha256':candidate_sha,'sealed_test_passed':test['passed_cases'],
              'sealed_test_cases':test['evaluated_cases'],'http_integration_passed':http['passed'],
              'browser_visual_checked':False,'excluded':['optimizer_training_state_pt','intermediate_checkpoints',
                  'live_inference_diagnostics','python_bytecode','sqlite_wal_shm'],
              'base_weights_included':False,'environment_image_included':False}
    raw=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
    with output.open('xb') as target,tarfile.open(fileobj=target,mode='w:gz') as bundle:
        for path in sorted(files): bundle.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
        member=tarfile.TarInfo('export-receipt.json');member.size=len(raw)
        bundle.addfile(member,io.BytesIO(raw))
    if any(digest(root/name)!=expected for name,expected in hashes.items()):
        raise ValueError('evidence_changed_during_export_no_sidecar_written')
    checksum=digest(output)
    sidecar.write_text(checksum+'  '+output.name+'\n',encoding='ascii')
    return {'archive':str(output),'sha256':checksum,'bytes':output.stat().st_size,'files':len(files),
            'selected_adapter_sha256':selection['selected_adapter_sha256'],
            'sealed_test_passed':test['passed_cases'],'sealed_test_cases':test['evaluated_cases']}


if __name__=='__main__': print(json.dumps(export(Path('/home/featurize/work')),ensure_ascii=False))
