from pathlib import Path
import argparse
import json
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.serving.telemetry import summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read-only trace report; never infer task accuracy from logs alone")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database.resolve().as_uri()+"?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [{**dict(row), "attributes": json.loads(row["attributes"])} for row in connection.execute("SELECT * FROM spans ORDER BY started_at,id")]
    finally:
        connection.close()
    result = summarize(rows)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
