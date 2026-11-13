import argparse
import asyncio
import time
from pathlib import Path

import mlflow
import pandas as pd
from tqdm import tqdm

from contractlens.config import get_settings
from contractlens.data import load_eval_set
from contractlens.evaluation import score_item, summarize, summarize_by_category
from contractlens.pipeline import ContractLens
from contractlens.prompts import PROMPTS
from contractlens.tracking import enable_tracing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ContractLens on the CUAD subset and log to MLflow.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment", default="contractlens-eval")
    parser.add_argument("--model")
    parser.add_argument("--escalation-model")
    parser.add_argument("--prompt-version", choices=sorted(PROMPTS))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--threshold", type=float, dest="confidence_threshold")
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions.")
    parser.add_argument("--trace", action="store_true", help="Record MLflow traces for every question.")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def error_result(exc: Exception) -> dict:
    return {
        "decision": "error", "answer": "", "quotes": [], "raw_answer": "", "raw_quotes": [], "parse_ok": False,
        "citations": [], "confidence": 0.0, "verification": None, "escalated": False, "llm_calls": 0,
        "retrieved": [], "latency_ms": 0.0, "error": f"{type(exc).__name__}: {exc}"[:300],
    }


async def ask_with_retry(app: ContractLens, question: str, doc_id: str, attempts: int = 2) -> dict:
    for attempt in range(attempts):
        try:
            return await app.ask(question, doc_id)
        except Exception as exc:
            if attempt == attempts - 1:
                print(f"\nquestion failed after {attempts} attempts: {type(exc).__name__}: {exc}")
                return error_result(exc)
            await asyncio.sleep(2)


async def evaluate(app: ContractLens, items) -> pd.DataFrame:
    rows = []
    for item in tqdm(items, desc="questions"):
        result = await ask_with_retry(app, item.question, item.doc_id)
        row = score_item(item, result)
        row["error"] = result.get("error", "")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    settings = get_settings(
        model=args.model,
        escalation_model=args.escalation_model,
        prompt_version=args.prompt_version,
        top_k=args.top_k,
        confidence_threshold=args.confidence_threshold,
        use_verifier=False if args.no_verifier else None,
    )
    _, items = load_eval_set(settings.data_dir)
    if args.limit:
        items = items[: args.limit]

    mlflow.set_experiment(args.experiment)
    enable_tracing(args.trace)
    app = ContractLens.from_settings(settings)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(settings.as_params())
        mlflow.log_text(PROMPTS[settings.prompt_version], f"prompts/{settings.prompt_version}.txt")

        started = time.time()
        df = asyncio.run(evaluate(app, items))
        metrics = summarize(df)
        metrics["error_rate"] = float((df["decision"] == "error").mean())
        metrics["wall_time_s"] = time.time() - started
        mlflow.log_metrics(metrics)

        df["run_id"] = run.info.run_id
        df["run_name"] = args.run_name
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_question = out_dir / f"{args.run_name}.csv"
        by_category = out_dir / f"{args.run_name}_by_category.csv"
        df.to_csv(per_question, index=False)
        summarize_by_category(df).to_csv(by_category, index=False)
        mlflow.log_artifact(str(per_question), "eval")
        mlflow.log_artifact(str(by_category), "eval")

    print(f"\nrun {args.run_name} ({run.info.run_id})")
    for key, value in metrics.items():
        print(f"  {key:26s} {value:.3f}")


if __name__ == "__main__":
    main()
