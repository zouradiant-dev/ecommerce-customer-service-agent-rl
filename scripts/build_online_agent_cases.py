from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.serving.online_cases import defaults
from ecommerce_customer_service.serving.storage import encode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write synthetic dev regression cases; never overwrite an existing manifest")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "online-agent-cases-v1.jsonl")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for case in defaults():
            stream.write(encode(case)+"\n")
    print("Wrote", len(defaults()), "synthetic dev cases")
