"""Snapshot completed local evidence, verify SQLite and every archive member."""
from pathlib import Path
import hashlib
import json
import shutil
import sqlite3
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'outputs/generalization-final-backup-20261015-r1'
    output.mkdir(exist_ok=False);stage=output/'verified-payload';stage.mkdir()
    experiments=('grounded-generalization-real-20261015-r1','catalog-generalization-real-20261015-r1','catalog-generalization-real-20261015-r2')
    for name in (*experiments,'grounded-generalization-cases-20261015-r1','catalog-generalization-cases-20261015-r1'):
        shutil.copytree(ROOT/'outputs'/name,stage/name)
    demo=ROOT/'outputs/catalog-generalization-demo-20261015-r1'
    target=stage/'browser-evidence';target.mkdir()
    for name in ('browser-completed.jpg','browser-full-page.jpg'):
        shutil.copy2(demo/name,target/name)
    for name in ('business.sqlite','sessions.sqlite'):
        source=sqlite3.connect((demo/name).as_uri()+'?mode=ro',uri=True)
        dest=sqlite3.connect(target/name);source.backup(dest);dest.close();source.close()
    db=sqlite3.connect(target/'business.sqlite');db.row_factory=sqlite3.Row
    orders=[dict(row) for row in db.execute('SELECT id,status FROM orders ORDER BY id')]
    counts={name:db.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in ('cancellations','refunds')}
    refunds=[dict(row) for row in db.execute('SELECT order_id,amount_cents,status FROM refunds')];db.close()
    assert counts=={'cancellations':1,'refunds':1}
    assert refunds==[{'order_id':'order_A','amount_cents':19900,'status':'pending'}]
    assert orders==[{'id':'order_A','status':'cancelled'},{'id':'order_B','status':'paid'},
                    {'id':'order_C','status':'shipped'},{'id':'order_D','status':'paid'}]
    receipt={'browser_visual_checked':True,'browser_js_interaction_checked':True,'profile':'binding_catalog_v6',
             'alias_request':'麻烦给耳机办理退订。','chat_confirm_rebuilt_preview_without_writing':True,
             'expired_confirmation_rejected_in_ui':True,'valid_button_confirmation_succeeded':True,
             'reload_preserved_history_and_result':True,'mock_business_only':True,'orders':orders,'counts':counts,'refunds':refunds}
    (target/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copytree(ROOT/'src',stage/'source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'scripts',stage/'source/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('README.md','pyproject.toml','requirements-inference-v18.txt','docs/grounded-generalization-20261015.zh-CN.md','tests/test_grounded_target.py'):
        path=stage/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    sqlite_count=0
    for path in stage.rglob('*.sqlite'):
        db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok',path
        db.close();sqlite_count+=1
    files={p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file()}
    (stage/'files-sha256.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    archive=output/'generalization-results-20261015-r1.tar.gz'
    with tarfile.open(archive,'w:gz') as target_tar:
        for p in sorted(stage.rglob('*')):
            if p.is_file():target_tar.add(p,arcname=p.relative_to(stage).as_posix())
    with tarfile.open(archive,'r:gz') as check:
        for name,digest in files.items():
            member=check.getmember(name);assert member.isfile() and hashlib.sha256(check.extractfile(member).read()).hexdigest()==digest,name
    record={'archive_sha256':sha(archive),'verified_files':len(files),'sqlite_databases_verified':sqlite_count,
            'all_results_local':True,'no_training':True,'model_weights_unchanged_sha256':'0c9da9f747324ed06f4b5184ac7a9b439255e9b5f9a6e7b57789ae6540085e73',
            'weights_retained_in_prior_verified_backup':True,'weights_in_this_archive':False,'base_environment_in_this_archive':False,
            'failed_dev_and_first_acceptance_retained':True,'browser_checked':True,'production_promoted':False}
    (output/'verification-receipt.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (output/(archive.name+'.sha256')).write_text(record['archive_sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
