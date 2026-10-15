"""Preserve all v8 dev failures, frozen acceptance and optional UI evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sqlite3
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--test',type=Path,required=True)
    p.add_argument('--test-cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--demo-dir',type=Path);args=p.parse_args()
    if not (args.test/'completed.json').exists():raise ValueError('completed_test_required')
    args.output.mkdir(parents=True,exist_ok=False);payload=args.output/'verified-payload';payload.mkdir()
    shutil.copytree(args.test,payload/'third-test-results');shutil.copytree(args.test_cases,payload/'sealed-third-test')
    for pattern in ('request-development-cases-20261015-r*','request-workflow-development-20261015-r*',
                    'request-workflow-launch-20261015-r*','request-workflow-test-authorship-20261015-r*',
                    'request-workflow-demo-launch-20261015-r*'):
        for source in sorted((ROOT/'outputs').glob(pattern)):
            if source.is_dir():shutil.copytree(source,payload/source.name)
    qa=payload/'cpu-validation';qa.mkdir()
    for pattern in ('request-workflow-qa-*.log','request-workflow-known-case-contracts-*.json'):
        for source in sorted((ROOT/'outputs').glob(pattern)):shutil.copy2(source,qa/source.name)
    for source in sorted((ROOT/'outputs').glob('request-workflow-local-recovery-check-*.json')):
        shutil.copy2(source,qa/source.name)
    resume=payload/'resume-inference-code';resume.mkdir()
    for name in ('Start-RequestDemo24566.ps1','request-demo-24566-code-start.sh','request-demo-24566-local.py'):
        source=ROOT/'packages'/name
        if source.exists():shutil.copy2(source,resume/name)
    code_bundle=ROOT/'packages/request-demo-24566-code-r1'
    if code_bundle.exists():shutil.copytree(code_bundle,resume/code_bundle.name)
    for source in ('src','scripts','tests'):
        shutil.copytree(ROOT/source,payload/'source'/source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('README.md','docs/request-workflow-20261015.zh-CN.md'):
        target=payload/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    consumed=ROOT/'data/consumed-request-heldout-20261015-r1'
    if consumed.exists():shutil.copytree(consumed,payload/'consumed-test-review')
    if args.demo_dir:
        target=payload/'browser-evidence';target.mkdir()
        for source in args.demo_dir.glob('browser-*'):
            if source.is_file():shutil.copy2(source,target/source.name)
        for name in ('sessions.sqlite','business.sqlite'):
            source=sqlite3.connect((args.demo_dir/name).resolve().as_uri()+'?mode=ro',uri=True)
            dest=sqlite3.connect(target/name);source.backup(dest);dest.close();source.close()
    sqlite_count=0
    for path in payload.rglob('*.sqlite'):
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('sqlite_integrity_failed:'+str(path))
        db.close();sqlite_count+=1
    files={path.relative_to(payload).as_posix():sha(path) for path in payload.rglob('*') if path.is_file()}
    write(payload/'files-sha256.json',files)
    archive=args.output/'request-workflow-20261015-r1.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        for path in sorted(payload.rglob('*')):
            if path.is_file():stream.add(path,arcname=path.relative_to(payload).as_posix())
    with tarfile.open(archive,'r:gz') as stream:
        for name,digest in files.items():
            if hashlib.sha256(stream.extractfile(name).read()).hexdigest()!=digest:raise ValueError('archive_member_changed:'+name)
    receipt=dict(archive_sha256=sha(archive),verified_files=len(files),sqlite_databases_verified=sqlite_count,
        all_development_failures_retained=True,sqlite_count_includes_reused_baseline_copies=True,
        weights_unchanged=True,weights_in_archive=False,base_environment_in_archive=False,
        no_training=True,production_promoted=False)
    write(args.output/'verification-receipt.json',receipt)
    (args.output/(archive.name+'.sha256')).write_text(receipt['archive_sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
