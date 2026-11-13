from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .data import EvalItem

_TOKEN_RE = re.compile(r"[a-z0-9]+")
F1_CORRECT_THRESHOLD = 0.3


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = _tokens(prediction), _tokens(gold)
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, golds: list[str]) -> float:
    return max((token_f1(prediction, g) for g in golds), default=0.0)


def retrieval_hit(retrieved: list[dict], spans: list[tuple[int, int]], min_coverage: float = 0.5) -> bool:
    """True if any retrieved chunk covers at least min_coverage of any gold span."""
    for start, end in spans:
        length = max(1, end - start)
        for r in retrieved:
            overlap = min(end, r["end"]) - max(start, r["start"])
            if overlap / length >= min_coverage:
                return True
    return False


def score_item(item: EvalItem, result: dict) -> dict:
    answered = result["decision"] == "answered"
    prediction = " ".join(result["quotes"]) or result["answer"]
    raw_prediction = " ".join(result["raw_quotes"]) or result["raw_answer"]
    verification = result.get("verification") or {}
    f1 = best_f1(prediction, item.answers) if answered and item.answerable else 0.0
    row = {
        "qid": item.qid,
        "doc_id": item.doc_id,
        "category": item.category,
        "answerable": item.answerable,
        "decision": result["decision"],
        "answered": answered,
        "escalated": result["escalated"],
        "llm_calls": result["llm_calls"],
        "confidence": result["confidence"],
        "quote_support": verification.get("quote_support", 0.0),
        "nli_support": verification.get("nli_support", 0.0),
        "relevance": verification.get("relevance", 0.0),
        "model_found": result["decision"] in ("answered", "abstained"),
        "parse_ok": result["parse_ok"],
        "retrieval_hit": retrieval_hit(result["retrieved"], item.spans) if item.answerable else None,
        "f1": f1,
        "raw_f1": best_f1(raw_prediction, item.answers) if item.answerable else 0.0,
        "latency_ms": result.get("latency_ms", 0.0),
        "question": item.question,
        "prediction": prediction if answered else "",
        "gold": " || ".join(item.answers),
    }
    row["unsupported"] = answered and row["quote_support"] < 0.9
    row["correct"] = (item.answerable and answered and f1 >= F1_CORRECT_THRESHOLD) or (not item.answerable and not answered)
    return row


def summarize(df: pd.DataFrame) -> dict[str, float]:
    ans = df[df["answerable"]]
    unans = df[~df["answerable"]]
    answered = df[df["answered"]]

    def mean(series) -> float:
        return float(series.mean()) if len(series) else 0.0

    return {
        "n_questions": float(len(df)),
        "accuracy": mean(df["correct"]),
        "retrieval_hit_rate": mean(ans["retrieval_hit"].astype(float)),
        "answer_f1": mean(ans["f1"]),
        "answered_f1": mean(ans[ans["answered"]]["f1"]),
        "coverage": mean(ans["answered"]),
        "abstention_accuracy": mean(~unans["answered"]),
        "false_answer_rate": mean(unans["answered"]),
        "unsupported_answer_rate": mean(answered["unsupported"]),
        "escalation_rate": mean(df["escalated"]),
        "llm_calls_per_question": mean(df["llm_calls"]),
        "json_parse_rate": mean(df["parse_ok"]),
        "latency_ms_mean": mean(df["latency_ms"]),
        "latency_ms_p95": float(np.percentile(df["latency_ms"], 95)) if len(df) else 0.0,
    }


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    overall = df.groupby("category").agg(n=("qid", "count"), accuracy=("correct", "mean"))
    answerable = df[df["answerable"]].groupby("category").agg(
        n_answerable=("qid", "count"), answer_f1=("f1", "mean"), coverage=("answered", "mean")
    )
    return overall.join(answerable, how="left").fillna(0).round(3).reset_index()


def threshold_sweep(df: pd.DataFrame, thresholds) -> pd.DataFrame:
    """Re-score a run at different confidence thresholds without new LLM calls.

    Uses each question's pre-abstention draft (model_found, raw_f1, confidence).
    """
    rows = []
    for t in thresholds:
        answered = df["model_found"] & (df["confidence"] >= t)
        f1 = df["raw_f1"].where(answered, 0.0)
        correct = (df["answerable"] & answered & (f1 >= F1_CORRECT_THRESHOLD)) | (~df["answerable"] & ~answered)
        ans = df["answerable"]
        rows.append({
            "threshold": float(t),
            "accuracy": float(correct.mean()),
            "answer_f1": float(f1[ans].mean()) if ans.any() else 0.0,
            "coverage": float(answered[ans].mean()) if ans.any() else 0.0,
            "false_answer_rate": float(answered[~ans].mean()) if (~ans).any() else 0.0,
        })
    return pd.DataFrame(rows)
