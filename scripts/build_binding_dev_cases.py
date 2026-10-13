from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.serving.binding_dev_cases import cases

if __name__ == "__main__":
    output = ROOT / "data" / "online-binding-dev-v2.jsonl"
    with output.open("x", encoding="utf-8") as target:
        for case in cases():
            target.write(json.dumps(case, ensure_ascii=False) + "\n")
    print("Created", len(cases()), "synthetic dev guards; not independent test")
