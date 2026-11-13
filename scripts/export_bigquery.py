import argparse
import json
from pathlib import Path

import pandas as pd

from contractlens.bq import load_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Load evaluation results and request logs into BigQuery.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", default="contractlens")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--request-log", default="logs/requests.jsonl")
    args = parser.parse_args()

    files = [p for p in Path(args.results_dir).glob("*.csv") if not p.stem.endswith("_by_category")]
    if files:
        results = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        rows = load_dataframe(results, args.project, args.dataset, "eval_results")
        print(f"eval_results: loaded {rows} rows from {len(files)} runs")

    log_path = Path(args.request_log)
    if log_path.exists():
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        if records:
            logs = pd.DataFrame(records)
            logs["timestamp"] = pd.to_datetime(logs["timestamp"])
            rows = load_dataframe(logs, args.project, args.dataset, "request_logs")
            print(f"request_logs: loaded {rows} rows")


if __name__ == "__main__":
    main()
