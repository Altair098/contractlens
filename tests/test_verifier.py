import pandas as pd

from contractlens.evaluation import best_f1, retrieval_hit, threshold_sweep, token_f1
from contractlens.verifier import Draft, NliScorer, Verifier, parse_draft, quote_support

CONTEXT = "This Agreement shall be governed by the laws of the State of New York."


def test_parse_draft_handles_wrapped_json():
    raw = 'Sure!\n```json\n{"found": true, "answer": "New York law", "quotes": ["laws of the State of New York"], "chunk_ids": ["1"]}\n```'
    draft = parse_draft(raw)
    assert draft.found and draft.parse_ok
    assert draft.chunk_ids == [1]


def test_parse_draft_fallback_on_invalid_json():
    draft = parse_draft("Not found in the excerpts.")
    assert not draft.found and not draft.parse_ok


def test_quote_support_rejects_fabricated_quote():
    real, _ = quote_support(["governed by the laws of the State of New York"], CONTEXT)
    fake, supported = quote_support(["governed by the laws of the State of California"], CONTEXT)
    assert real == 1.0
    assert fake < real and not supported


def test_verifier_scores_grounded_answer_higher():
    verifier = Verifier(NliScorer("lexical"))
    good = Draft(True, "The agreement is governed by New York law.", ["governed by the laws of the State of New York"], [1])
    bad = Draft(True, "The agreement is governed by Delaware law.", ["governed by Delaware law"], [1])
    question = "Which state's laws govern the agreement?"
    assert verifier.verify(good, [CONTEXT], question).confidence > verifier.verify(bad, [CONTEXT], question).confidence


def test_relevance_penalizes_off_topic_evidence(index):
    verifier = Verifier(NliScorer("lexical"))
    question = "Which laws govern the agreement? governing law"
    on_topic = Draft(True, "New York law governs.", ["governed by and construed in accordance with the laws of the State of New York"], [1])
    off_topic = Draft(True, "Assignment needs consent.", ["Neither party may assign this Agreement without the prior written consent"], [1])
    contexts = [c.text for c in index.chunks if c.doc_id == "docA"]
    on = verifier.verify(on_topic, contexts, question, index.relevance)
    off = verifier.verify(off_topic, contexts, question, index.relevance)
    assert on.relevance > off.relevance


def test_metrics():
    assert token_f1("laws of New York", "laws of New York") == 1.0
    assert best_f1("New York", ["Delaware", "New York law"]) > 0.5
    assert retrieval_hit([{"start": 0, "end": 100}], [(50, 90)])
    assert not retrieval_hit([{"start": 0, "end": 60}], [(50, 150)])


def test_threshold_sweep_trades_coverage_for_safety():
    df = pd.DataFrame({
        "answerable": [True, True, False, False],
        "model_found": [True, True, True, False],
        "confidence": [0.9, 0.4, 0.5, 0.0],
        "raw_f1": [0.8, 0.6, 0.0, 0.0],
    })
    sweep = threshold_sweep(df, [0.0, 0.55, 0.95]).set_index("threshold")
    assert sweep.loc[0.0, "coverage"] == 1.0 and sweep.loc[0.0, "false_answer_rate"] == 0.5
    assert sweep.loc[0.55, "false_answer_rate"] == 0.0 and sweep.loc[0.55, "accuracy"] == 0.75
    assert sweep.loc[0.95, "coverage"] == 0.0
