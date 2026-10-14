"""CPU-only preparation for separate Agent dev evaluation and train-only RL sampling."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecommerce_customer_service.dual_track import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/v18-train-dev-r1/tasks.jsonl')
    parser.add_argument('--original-adapter', type=Path, default=ROOT / 'models/v18-sft-verified')
    parser.add_argument('--candidate-adapter', type=Path, default=ROOT / 'outputs/online-repair-backup-20261013-r1/extracted/outputs/online-repair-pilot-r1/pilot')
    parser.add_argument('--dev-cases', type=Path, default=ROOT / 'outputs/online-repair-sft-20261014-v4-r2/dev-cases.jsonl')
    parser.add_argument('--historical-audit', type=Path, default=ROOT / 'outputs/v18-historical-signal-audit-20261014/audit.json')
    args = parser.parse_args()
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    plan = freeze(ROOT, args.output.resolve(), manifest=args.manifest, original=args.original_adapter,
                  candidate=args.candidate_adapter, dev_cases=args.dev_cases,
                  historical_audit=args.historical_audit, git_head=git.stdout.strip() if git.returncode == 0 else None)
    print(json.dumps({'plan': str(args.output / 'plan.json'), 'agent_core': plan['agent']['core_cases'],
                      'agent_knowledge': plan['agent']['knowledge_cases'], 'rl_tasks': plan['rl']['selected_tasks'],
                      'rl_rollouts': plan['rl']['rollouts'], 'training_authorized': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
