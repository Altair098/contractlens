import argparse

import mlflow

COLUMNS = [
    "accuracy",
    "retrieval_hit_rate",
    "answer_f1",
    "coverage",
    "abstention_accuracy",
    "false_answer_rate",
    "unsupported_answer_rate",
    "escalation_rate",
    "error_rate",
    "latency_ms_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a markdown table comparing MLflow evaluation runs.")
    parser.add_argument("--experiment", default="contractlens-eval")
    args = parser.parse_args()

    runs = mlflow.search_runs(experiment_names=[args.experiment], order_by=["start_time ASC"])
    if runs.empty:
        raise SystemExit("no runs found")
    names = runs["tags.mlflow.runName"]
    failed = runs[runs["status"] != "FINISHED"]
    for name in failed["tags.mlflow.runName"]:
        print(f"skipping unfinished run: {name}")
    runs = runs[(runs["status"] == "FINISHED") & ~names.str.endswith("-threshold-sweep")]
    runs = runs.drop_duplicates("tags.mlflow.runName", keep="last")
    table = runs[["tags.mlflow.runName"] + [f"metrics.{c}" for c in COLUMNS if f"metrics.{c}" in runs]]
    table.columns = ["run"] + [c.removeprefix("metrics.") for c in table.columns[1:]]
    print(table.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
