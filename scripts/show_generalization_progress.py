"""Read local saved evaluation progress without issuing model requests."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/catalog-generalization-real-20261015-r2')
    args=p.parse_args()
    expected=[('new-dev-binding_guard_v4',42),('new-dev-binding_grounded_v5',42),
              ('known-regression-binding_grounded_v5',30),('prior-business-dev-binding_grounded_v5',57),
              ('acceptance-binding_grounded_v5',42)]
    protocol=args.output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')).get('profile')=='binding_catalog_v6':
        expected=[('consumed-acceptance-dev',42),('prior-business-dev',57),('fresh-acceptance',30)]
    done=0
    for name,total in expected:
        directory=args.output/name
        results=list((directory/'cases').glob('*/result.json'))
        count=len(results);passed=sum(json.loads(r.read_text(encoding='utf-8'))['passed'] for r in results)
        done+=count
        print(f'{name}: {count}/{total}, passed={passed}')
    print(f'Completed trajectories: {done}/{sum(total for _,total in expected)} maximum; acceptance runs only after the dev gate.')
    completed=args.output/'completed.json'
    if completed.exists():print(completed.read_text(encoding='utf-8'))


if __name__=='__main__':main()
