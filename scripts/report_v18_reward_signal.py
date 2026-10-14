from hashlib import sha256
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ecommerce_customer_service.reward_signal import describe

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Read-only strict/shaped reward variation analysis; no model or optimizer')
    parser.add_argument('--rollouts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raw=args.rollouts.read_bytes()
    report=describe([json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()])
    report['rollouts_sha256']=sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:json.dump(report,out,ensure_ascii=False,indent=2)
    print(json.dumps({key:report[key] for key in ('tasks','rollouts','formal_rl_gate_passed')}))
