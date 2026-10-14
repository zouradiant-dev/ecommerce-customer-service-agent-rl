"""Package additive recovery files; preserve the original frozen experiment."""
from pathlib import Path
from hashlib import sha256
import io
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    archive=ROOT/'packages/current-instance-shuffle-recovery-20261014-r1.tar.gz'
    if archive.exists(): raise FileExistsError('recovery_package_exists')
    files={
        'scripts/train_sft_closeout_single_shuffle.py':ROOT/'scripts/train_sft.py',
        'scripts/run_current_instance_closeout_recovered.py':ROOT/'scripts/run_current_instance_closeout.py',
        'src/ecommerce_customer_service/serving/closeout_recovery.py':ROOT/'src/ecommerce_customer_service/serving/closeout_recovery.py',
    }
    with tarfile.open(ROOT/'packages/current-instance-closeout-20261014-r1.tar.gz') as old:
        original_receipt=old.extractfile('bundle-receipt.json').read()
    manifest={'schema_version':'closeout_shuffle_recovery_v1',
        'original_bundle_receipt_sha256':sha256(original_receipt).hexdigest(),
        'files_sha256':{name:sha256(path.read_bytes()).hexdigest() for name,path in files.items()},
        'original_files_must_remain_unchanged':True,'reuse_completed_baselines':True,
        'reuse_four_step_smoke_only_after_cpu_audit':True,'pilot_steps':64,
        'pilot_initializes_from_saved_online_not_smoke':True,'sealed_test_not_opened_for_repair':True}
    raw=json.dumps(manifest,indent=2).encode()
    with tarfile.open(archive,'x:gz') as target:
        for name,path in files.items(): target.add(path,arcname=name,recursive=False)
        member=tarfile.TarInfo('recovery-package.json');member.size=len(raw)
        target.addfile(member,io.BytesIO(raw))
    checksum=sha256(archive.read_bytes()).hexdigest()
    (ROOT/'packages/current-instance-shuffle-recovery-20261014-r1.json').write_bytes(raw+b'\n')
    Path(str(archive)+'.sha256').write_text(checksum+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps({'archive':str(archive),'sha256':checksum,'bytes':archive.stat().st_size,
                      'original_receipt_sha256':manifest['original_bundle_receipt_sha256']},ensure_ascii=False))


if __name__=='__main__': main()
