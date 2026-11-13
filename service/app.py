from __future__ import annotations

import json
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from contractlens.chunking import chunk_pages
from contractlens.config import get_settings
from contractlens.parsing import get_parser
from contractlens.pipeline import ContractLens

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state["settings"] = settings
    state["app"] = ContractLens.from_settings(settings)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    yield
    state.clear()


app = FastAPI(title="ContractLens", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    contract_id: str | None = None


class AskResponse(BaseModel):
    contract_id: str | None
    decision: str
    answer: str
    quotes: list[str]
    citations: list[dict]
    confidence: float
    escalated: bool
    latency_ms: float


def log_request(payload: dict) -> None:
    path = state["settings"].log_dir / "requests.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


@app.get("/health")
def health() -> dict:
    lens: ContractLens = state["app"]
    return {"status": "ok", "model": lens.settings.model, "documents": len(lens.index.doc_ids())}


@app.get("/contracts")
def contracts() -> list[str]:
    return state["app"].index.doc_ids()


@app.post("/contracts", status_code=201)
async def upload_contract(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are supported")
    lens: ContractLens = state["app"]
    doc_id = Path(file.filename).stem
    if doc_id in lens.index.doc_ids():
        raise HTTPException(409, f"contract {doc_id} already indexed")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        pages = get_parser("local", ocr=False).parse(tmp.name)
    settings = state["settings"]
    _, chunks = chunk_pages(doc_id, pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(422, "no extractable text in PDF")
    lens.index.add(chunks)
    return {"contract_id": doc_id, "pages": len(pages), "chunks": len(chunks)}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    lens: ContractLens = state["app"]
    if req.contract_id and req.contract_id not in lens.index.doc_ids():
        raise HTTPException(404, f"unknown contract {req.contract_id}")
    started = time.perf_counter()
    try:
        result = await lens.ask(req.question, req.contract_id)
    except Exception as exc:
        log_request({"timestamp": datetime.now(timezone.utc).isoformat(), "contract_id": req.contract_id, "decision": "error",
                     "error": str(exc)[:500], "latency_ms": round((time.perf_counter() - started) * 1000, 1)})
        raise HTTPException(502, "model backend failed") from exc
    log_request({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contract_id": req.contract_id,
        "question": req.question,
        "decision": result["decision"],
        "confidence": result["confidence"],
        "escalated": result["escalated"],
        "llm_calls": result["llm_calls"],
        "model": result["model"],
        "latency_ms": result["latency_ms"],
    })
    return AskResponse(**{k: result[k] for k in AskResponse.model_fields})
