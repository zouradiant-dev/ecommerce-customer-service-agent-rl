"""Read-only tokenizer audit; never runs a model or executes test references."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ecommerce_customer_service.serving.context_builder import load_token_counter
from ecommerce_customer_service.serving.split_integrity import audit
from ecommerce_customer_service.serving.policy import PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--dev',type=Path,required=True)
    parser.add_argument('--test-cases',type=Path)
    parser.add_argument('--tokenizer-path',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy-profile',choices=PROFILES,default='binding_compact_v3')
    args = parser.parse_args()
    def read(path): return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = audit({'train':read(args.train),'dev':read(args.dev)},load_token_counter(args.tokenizer_path).tokenizer,
                   sealed_cases=read(args.test_cases) if args.test_cases else (),policy_profile=args.policy_profile)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out: json.dump(result,out,ensure_ascii=False,indent=2)
    print(json.dumps({'passed':result['passed'],'collision_groups':len(result['cross_split_collisions']),
                      'counts':result['counts']},ensure_ascii=False))
    return 0 if result['passed'] else 2


if __name__=='__main__': raise SystemExit(main())
