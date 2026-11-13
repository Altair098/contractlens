from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Callable

from rapidfuzz import fuzz

from .index import tokenize

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_WS_RE = re.compile(r"\s+")


@dataclass
class Draft:
    found: bool
    answer: str
    quotes: list[str] = field(default_factory=list)
    chunk_ids: list[int] = field(default_factory=list)
    parse_ok: bool = True


@dataclass
class Verification:
    quote_support: float
    nli_support: float
    relevance: float
    confidence: float
    supported_quotes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def parse_draft(raw: str) -> Draft:
    raw = (raw or "").strip()
    match = _JSON_RE.search(raw)
    if match:
        try:
            obj = json.loads(match.group(0))
            quotes = [str(q) for q in obj.get("quotes") or [] if str(q).strip()]
            chunk_ids = []
            for cid in obj.get("chunk_ids") or []:
                try:
                    chunk_ids.append(int(cid))
                except (TypeError, ValueError):
                    continue
            found = obj.get("found", bool(quotes))
            if isinstance(found, str):
                found = found.strip().lower() == "true"
            return Draft(bool(found), str(obj.get("answer", "")).strip(), quotes, chunk_ids)
        except json.JSONDecodeError:
            pass
    not_found = not raw or "not found" in raw.lower()
    return Draft(found=not not_found, answer=raw, parse_ok=False)


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def quote_support(quotes: list[str], context: str, min_len: int = 12) -> tuple[float, list[str]]:
    """Fraction of quotes that appear (fuzzily) verbatim in the retrieved context."""
    if not quotes:
        return 0.0, []
    ctx = normalize(context)
    scores, supported = [], []
    for q in quotes:
        nq = normalize(q)
        if len(nq) < min_len:
            scores.append(0.0)
            continue
        score = fuzz.partial_ratio(nq, ctx) / 100.0 if nq not in ctx else 1.0
        scores.append(score)
        if score >= 0.9:
            supported.append(q)
    return sum(scores) / len(scores), supported


class NliScorer:
    """Entailment probability that the evidence supports the answer."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        if model_name != "lexical":
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
            self.entail_idx = labels.get("entailment", 1)

    def score(self, evidence: str, claim: str) -> float:
        if not evidence.strip() or not claim.strip():
            return 0.0
        if self.model is None:
            claim_tokens = set(tokenize(claim))
            if not claim_tokens:
                return 0.0
            return len(claim_tokens & set(tokenize(evidence))) / len(claim_tokens)
        probs = self.model.predict([(evidence[:2000], claim[:500])], apply_softmax=True)[0]
        return float(probs[self.entail_idx])


RelevanceFn = Callable[[str, str], float]


class Verifier:
    """Combines three checks into one confidence score.

    quote_support: are the quotes really in the retrieved text (catches fabricated evidence)
    nli_support:   does the evidence entail the answer (catches unsupported claims)
    relevance:     does the evidence address the question (catches off-topic answers)
    """

    def __init__(self, nli: NliScorer, weights: tuple[float, float, float] = (0.4, 0.3, 0.3)):
        self.nli = nli
        self.weights = weights

    def verify(self, draft: Draft, contexts: list[str], question: str, relevance_fn: RelevanceFn | None = None) -> Verification:
        joined = "\n".join(contexts)
        q_support, supported = quote_support(draft.quotes, joined)
        evidence = " ".join(supported) if supported else (draft.answer or joined)
        nli = self.nli.score(evidence, draft.answer) if draft.found else 0.0
        relevance = max(0.0, min(1.0, relevance_fn(question, evidence))) if relevance_fn else 0.0
        w_q, w_n, w_r = self.weights
        confidence = w_q * q_support + w_n * nli + w_r * relevance
        return Verification(q_support, nli, relevance, confidence, supported)
