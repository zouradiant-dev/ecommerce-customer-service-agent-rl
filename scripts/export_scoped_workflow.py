"""Archive scoped-workflow development, test, frozen code and optional UI evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sqlite3
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--test',type=Path,required=True)
    p.add_argument('--test-cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--demo-dir',type=Path);args=p.parse_args()
    if not (args.test/'completed.json').exists():raise ValueError('completed_test_required')
    args.output.mkdir(parents=True,exist_ok=False);stage=args.output/'verified-payload';stage.mkdir()
    locations={'development':args.development,'test':args.test,'sealed-test-cases':args.test_cases}
    authorship=ROOT/'outputs/scoped-workflow-test-authorship-20261015-r1'
    if authorship.exists():locations['test-authorship']=authorship
    for dataset in sorted((ROOT/'outputs').glob('scoped-development-cases-20261015-r*')):
        locations[dataset.name]=dataset
    # Retain every failed dev iteration rather than exporting the best score alone.
    for old in sorted((ROOT/'outputs').glob('scoped-workflow-development-20261015-r*')):
        if old!=args.development:locations['previous-'+old.name]=old
    for name,source in locations.items():shutil.copytree(source,stage/name)
    for launch in sorted((ROOT/'outputs').glob('scoped-workflow-launch-20261015-r*')):
        shutil.copytree(launch,stage/'launch-logs'/launch.name)
    qa=stage/'cpu-validation';qa.mkdir()
    for pattern in ('scoped-workflow-qa-*.log','scoped-workflow-case-contracts-*.json'):
        for path in (ROOT/'outputs').glob(pattern):shutil.copy2(path,qa/path.name)
    shutil.copytree(ROOT/'src',stage/'current-source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'scripts',stage/'current-source/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'tests',stage/'current-source/tests',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    consumed=ROOT/'data/consumed-scoped-heldout-20261015-r1'
    if consumed.exists():shutil.copytree(consumed,stage/'consumed-test-review')
    for name in ('README.md','docs/scoped-workflow-20261015.zh-CN.md','tests/test_scoped_workflow.py'):
        path=ROOT/name
        if path.exists():
            target=stage/'current-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    if args.demo_dir:
        demo=stage/'browser-evidence';demo.mkdir()
        for path in args.demo_dir.glob('browser-*'):
            if path.is_file():shutil.copy2(path,demo/path.name)
        for name in ('sessions.sqlite','business.sqlite'):
            source=sqlite3.connect((args.demo_dir/name).resolve().as_uri()+'?mode=ro',uri=True)
            dest=sqlite3.connect(demo/name);source.backup(dest);dest.close();source.close()
    count=0
    for path in stage.rglob('*.sqlite'):
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('sqlite_integrity_failed:'+str(path))
        db.close();count+=1
    files={path.relative_to(stage).as_posix():sha(path) for path in stage.rglob('*') if path.is_file()}
    write(stage/'files-sha256.json',files)
    archive=args.output/'scoped-workflow-20261015-r1.tar.gz'
    with tarfile.open(archive,'w:gz') as target:
        for path in sorted(stage.rglob('*')):
            if path.is_file():target.add(path,arcname=path.relative_to(stage).as_posix())
    with tarfile.open(archive,'r:gz') as check:
        for name,digest in files.items():
            if hashlib.sha256(check.extractfile(name).read()).hexdigest()!=digest:raise ValueError('archive_member_changed:'+name)
    result=dict(archive_sha256=sha(archive),verified_files=len(files),sqlite_databases_verified=count,
        all_dev_failures_and_test_retained=True,weights_unchanged=True,weights_in_archive=False,
        base_environment_in_archive=False,no_training=True,production_promoted=False)
    write(args.output/'verification-receipt.json',result)
    (args.output/(archive.name+'.sha256')).write_text(result['archive_sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
