import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from contractlens.evaluation import threshold_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the verifier confidence threshold on a finished run.")
    parser.add_argument("results_csv", help="Per-question CSV written by run_eval.py")
    parser.add_argument("--experiment", default="contractlens-eval")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    sweep = threshold_sweep(df, np.round(np.arange(0.0, 1.0, 0.05), 2))
    best = sweep.loc[sweep["accuracy"].idxmax()]

    fig, ax = plt.subplots(figsize=(7, 4))
    for col in ("accuracy", "coverage", "false_answer_rate"):
        ax.plot(sweep["threshold"], sweep[col], marker="o", markersize=3, label=col)
    ax.axvline(best["threshold"], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("confidence threshold")
    ax.set_ylim(0, 1)
    ax.set_title("Verifier threshold: accuracy vs coverage vs false answers")
    ax.legend()
    fig.tight_layout()

    source = Path(args.results_csv).stem
    out_png = Path(args.results_csv).with_name(f"{source}_threshold_sweep.png")
    fig.savefig(out_png, dpi=150)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{source}-threshold-sweep"):
        mlflow.log_param("source_run", source)
        for i, row in sweep.iterrows():
            mlflow.log_metrics({k: row[k] for k in ("accuracy", "coverage", "false_answer_rate", "answer_f1")}, step=i)
        mlflow.log_metrics({"best_threshold": best["threshold"], "best_accuracy": best["accuracy"]})
        mlflow.log_artifact(str(out_png))

    print(sweep.round(3).to_markdown(index=False))
    print(f"\nbest threshold {best['threshold']:.2f} -> accuracy {best['accuracy']:.3f}, coverage {best['coverage']:.3f}, "
          f"false answers {best['false_answer_rate']:.3f}")


if __name__ == "__main__":
    main()
