import asyncio

import pandas as pd
import pytest
from google.genai import types
from fastapi.testclient import TestClient

from contractlens.evaluation import score_item, summarize
from contractlens.pipeline import ContractLens
from contractlens.verifier import NliScorer, Verifier


@pytest.fixture
def lens(index, settings):
    return ContractLens(index, Verifier(NliScorer("lexical")), settings)


def test_answers_with_grounded_citation(lens):
    result = lens.ask_sync("Which state's laws govern this agreement? governing law", "docA")
    assert result["decision"] == "answered"
    assert "New York" in result["answer"]
    assert result["citations"] and result["citations"][0]["doc_id"] == "docA"
    assert result["confidence"] >= lens.settings.confidence_threshold


def test_low_confidence_answer_is_abstained(index, settings):
    settings.confidence_threshold = 0.99
    lens = ContractLens(index, Verifier(NliScorer("lexical")), settings)
    result = lens.ask_sync("Which state's laws govern this agreement? governing law", "docA")
    assert result["decision"] == "abstained"
    assert result["answer"] == ""


def test_escalation_runs_second_model(index, settings):
    settings.confidence_threshold = 0.99
    settings.escalation_model = "fake"
    lens = ContractLens(index, Verifier(NliScorer("lexical")), settings)
    result = lens.ask_sync("Which state's laws govern this agreement? governing law", "docA")
    assert result["escalated"] and result["llm_calls"] == 2


def test_evaluation_end_to_end(lens, eval_items):
    async def run():
        return [score_item(item, await lens.ask(item.question, item.doc_id)) for item in eval_items]

    df = pd.DataFrame(asyncio.run(run()))
    metrics = summarize(df)
    assert metrics["retrieval_hit_rate"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["accuracy"] == 1.0


def _make_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "LEASE AGREEMENT between Landlord and Tenant.")
    doc.new_page().insert_text((72, 72), "The tenant shall pay monthly rent on the first day of each month.")
    data = doc.tobytes()
    doc.close()
    return data


def test_api(index, settings, monkeypatch):
    monkeypatch.setenv("CL_MODEL", "fake")
    monkeypatch.setenv("CL_EMBEDDER", "tfidf")
    monkeypatch.setenv("CL_NLI_MODEL", "lexical")
    monkeypatch.setenv("CL_INDEX_DIR", str(settings.index_dir))
    monkeypatch.setenv("CL_LOG_DIR", str(settings.log_dir))
    from service.app import app

    with TestClient(app) as client:
        assert client.get("/health").json()["documents"] == 2
        resp = client.post("/ask", json={"question": "Which laws govern this agreement?", "contract_id": "docB"})
        assert resp.status_code == 200
        assert "England" in resp.json()["answer"]
        assert client.post("/ask", json={"question": "governing law", "contract_id": "nope"}).status_code == 404

        upload = client.post("/contracts", files={"file": ("lease.pdf", _make_pdf(), "application/pdf")})
        assert upload.status_code == 201 and upload.json()["pages"] == 2
        resp = client.post("/ask", json={"question": "When must the tenant pay monthly rent?", "contract_id": "lease"})
        assert resp.json()["citations"][0]["page"] == 2
    assert (settings.log_dir / "requests.jsonl").exists()


def test_multi_turn_session_uses_latest_question(lens):
    async def run():
        session = await lens.sessions.create_session(app_name="contractlens", user_id="u")
        replies = []
        for q in ("Which laws govern this agreement?", "Can a party assign this agreement without consent?"):
            msg = types.Content(role="user", parts=[types.Part(text=q)])
            async for _ in lens.runner.run_async(user_id="u", session_id=session.id, new_message=msg):
                pass
            state = (await lens.sessions.get_session(app_name="contractlens", user_id="u", session_id=session.id)).state
            replies.append(state["result"])
        return replies

    first, second = asyncio.run(run())
    assert first["question"] != second["question"]
    assert "assign" in second["answer"].lower()


def test_without_contract_id_answers_from_single_contract(lens):
    result = lens.ask_sync("Which laws govern this agreement?")
    assert result["contract_source"] == "auto"
    assert {c["doc_id"] for c in result["citations"]} == {result["contract_id"]}
    assert {r["doc_id"] for r in result["retrieved"]} == {result["contract_id"]}


def test_contract_named_in_question_is_used(lens):
    result = lens.ask_sync("In docB, which laws govern this agreement?")
    assert result["contract_id"] == "docB" and result["contract_source"] == "mentioned"
    assert "England" in result["answer"]
