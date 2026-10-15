"""Prepare a verified inference-only runtime; no training or evaluation is started."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.serving.closeout_experiment import verify_online_adapter


def main():
    output = ROOT / 'packages/generalization-64710-runtime-20261015-r1'
    output.mkdir(exist_ok=False)
    source = ROOT / 'outputs/current-instance-closeout-result-20261014-r1/extracted'
    adapter = source / 'outputs/closeout/pilot'
    identity = verify_online_adapter(adapter)
    assert identity['adapter_sha256'] == '0c9da9f747324ed06f4b5184ac7a9b439255e9b5f9a6e7b57789ae6540085e73'
    stage = output / 'payload'
    shutil.copytree(ROOT / 'src', stage / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (stage / 'scripts').mkdir()
    shutil.copy2(ROOT / 'scripts/serve_closeout_adapter.py', stage / 'scripts/serve_closeout_adapter.py')
    shutil.copytree(adapter, stage / 'models/selected')
    shutil.copy2(source / 'closeout-inputs/demo-policies.jsonl', stage / 'demo-policies.jsonl')
    files = {p.relative_to(stage).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in stage.rglob('*') if p.is_file()}
    (stage / 'runtime-files.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    archive = output / 'runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        for p in sorted(stage.rglob('*')):
            if p.is_file(): target.add(p, arcname=p.relative_to(stage).as_posix())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    shell = '''#!/usr/bin/env bash
set -euo pipefail
task_work=/home/featurize/work
printf '%s  %s\n' ARCHIVE_SHA "$task_work/generalization-64710-runtime.tar.gz" | sha256sum --check --status
task_repo=$(mktemp -d "$task_work/generalization-runtime-20261015.XXXXXX")
tar -xzf "$task_work/generalization-64710-runtime.tar.gz" -C "$task_repo"
cd "$task_repo"
printf '%s\n' "$task_repo" > "$task_work/generalization-64710-runtime.path"
export HF_HOME=/home/featurize/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
task_python=/home/featurize/grpo-env-v12/bin/python
task_base=/home/featurize/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
"$task_python" - "$task_base" <<'PY'
from pathlib import Path
import hashlib,json,socket,sys
sys.path.insert(0,'src')
from ecommerce_customer_service.serving.closeout_experiment import verify_online_adapter
from ecommerce_customer_service.serving.knowledge import KnowledgeIndex
from ecommerce_customer_service.serving.artifacts import verify_local_base
root=Path.cwd(); files=json.loads((root/'runtime-files.json').read_text())
for name,expected in files.items():
    p=(root/name).resolve(strict=True)
    if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('changed_runtime:'+name)
adapter=root/'models/selected';identity=verify_online_adapter(adapter)
verify_local_base(Path(sys.argv[1]),adapter)
with socket.socket() as probe:probe.bind(('127.0.0.1',8000))
selection={'selected':'pilot','selected_adapter_dir':str(adapter),'selected_adapter_sha256':identity['adapter_sha256'],
           'knowledge_sha256':KnowledgeIndex(root/'demo-policies.jsonl').sha256,'production_promoted':False}
(root/'selection.json').write_text(json.dumps(selection,indent=2))
print('RUNTIME VERIFIED. Frozen adapter; inference only; no training.',flush=True)
PY
"$task_python" -u scripts/serve_closeout_adapter.py --selection "$task_repo/selection.json" --base-model-dir "$task_base" 2>&1 | tee runtime.log
'''.replace('ARCHIVE_SHA', digest)
    (output / 'start.sh').write_text(shell, encoding='utf-8', newline='\n')
    ps = '''param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
$task_project = Split-Path -Parent $PSScriptRoot
$task_bundle = Join-Path $PSScriptRoot 'generalization-64710-runtime-20261015-r1'
$task_known = Join-Path $task_project 'outputs/featurize-64710/known_hosts'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $task_known) | Out-Null
$task_archive = Join-Path $task_bundle 'runtime.tar.gz'
if ((Get-FileHash -LiteralPath $task_archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ARCHIVE_SHA') { throw 'Runtime archive checksum mismatch' }
if ($ValidateOnly) { Write-Host 'LOCAL VALIDATION PASSED. No SSH or GPU work started.'; exit 0 }
if (Get-NetTCPConnection -State Listen -LocalPort 8647 -ErrorAction SilentlyContinue) { throw 'Local port 8647 is busy; do not start a duplicate tunnel.' }
Write-Host 'Upload frozen adapter and inference runtime. Enter SSH credentials only in this terminal.'
Write-Host 'For this new instance, SSH may ask you to confirm its first-use host fingerprint.'
$task_opts = @('-o',"UserKnownHostsFile=$task_known",'-o','StrictHostKeyChecking=ask')
scp -P 64710 @task_opts $task_archive 'featurize@workspace.featurize.cn:/home/featurize/work/generalization-64710-runtime.tar.gz'
if ($LASTEXITCODE -ne 0) { throw 'Runtime upload failed; inference was not started.' }
scp -P 64710 @task_opts (Join-Path $task_bundle 'start.sh') 'featurize@workspace.featurize.cn:/home/featurize/work/generalization-64710-start.sh'
if ($LASTEXITCODE -ne 0) { throw 'Launcher upload failed; inference was not started.' }
Write-Host 'Keep this window connected. Inference endpoint: http://127.0.0.1:8647/v1. No training is started.'
ssh -t -p 64710 @task_opts -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes -L 127.0.0.1:8647:127.0.0.1:8000 featurize@workspace.featurize.cn 'bash /home/featurize/work/generalization-64710-start.sh'
if ($LASTEXITCODE -ne 0) { throw 'Remote inference stopped; preserve the terminal output for diagnosis.' }
'''.replace('ARCHIVE_SHA', digest)
    (ROOT / 'packages/Start-Generalization64710.ps1').write_text(ps, encoding='utf-8-sig')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest,
                      'model': identity['model_id'], 'inference_only': True}, indent=2))


if __name__ == '__main__': main()
