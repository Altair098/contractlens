from __future__ import annotations

import re
from typing import Any, AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.genai import types

from .config import Settings
from .index import HybridIndex, SearchHit
from .models import is_native_gemini, resolve_model
from .prompts import format_context, render_prompt
from .tracking import span
from .verifier import Draft, Verifier, parse_draft


def make_answer_agent(name: str, model_name: str, prompt_version: str, output_key: str) -> LlmAgent:
    def instruction(ctx: ReadonlyContext) -> str:
        return render_prompt(prompt_version, ctx.state.get("question", ""), ctx.state.get("context", ""))

    return LlmAgent(
        name=name,
        model=resolve_model(model_name),
        description="Answers contract questions strictly from retrieved excerpts, with verbatim quotes.",
        instruction=instruction,
        include_contents="none",
        output_key=output_key,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512,
            response_mime_type="application/json" if is_native_gemini(model_name) else None,
        ),
    )


def _citations(draft: Draft, hits: list[SearchHit], supported_quotes: list[str]) -> list[dict]:
    chosen: list[SearchHit] = [hits[i - 1] for i in draft.chunk_ids if 1 <= i <= len(hits)]
    for quote in supported_quotes:
        needle = " ".join(quote.lower().split())[:80]
        for hit in hits:
            if needle in " ".join(hit.chunk.text.lower().split()) and hit not in chosen:
                chosen.append(hit)
    return [
        {"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "page": h.chunk.page, "start": h.chunk.start, "end": h.chunk.end}
        for h in chosen
    ]


def _mentioned_contract(question: str, doc_ids: list[str]) -> str | None:
    for doc_id in doc_ids:
        if re.search(rf"(?<![\w-]){re.escape(doc_id)}(?![\w-])", question, re.IGNORECASE):
            return doc_id
    return None


def _contract_line(result: dict) -> str:
    note = {"auto": " (auto-selected as most relevant; name a contract id in your question to choose)"}
    return f"Contract: {result['contract_id']}{note.get(result['contract_source'], '')}"


def _format_reply(result: dict) -> str:
    header = _contract_line(result) if result["contract_id"] else ""
    if result["decision"] != "answered":
        label = "Not found in the contract." if result["decision"] == "not_found" else "Insufficient evidence to answer reliably."
        return "\n".join(x for x in (header, f"{label} (confidence {result['confidence']:.2f})") if x)
    lines = ([header, ""] if header else []) + [result["answer"], ""]
    lines += [f'> "{q}"' for q in result["quotes"]]
    refs = ", ".join(f"{c['chunk_id']}" + (f" p.{c['page']}" if c["page"] else "") for c in result["citations"])
    lines += ["", f"Sources: {refs or 'n/a'} | confidence {result['confidence']:.2f}" + (" | escalated" if result["escalated"] else "")]
    return "\n".join(lines)


class ContractLensAgent(BaseAgent):
    """Retrieve -> answer -> verify -> (escalate | abstain) orchestration."""

    index: Any
    verifier: Any
    settings: Any
    answerer: LlmAgent
    escalator: LlmAgent | None = None

    def __init__(self, index: HybridIndex, verifier: Verifier, settings: Settings):
        answerer = make_answer_agent("answerer", settings.model, settings.prompt_version, "draft_raw")
        escalator = None
        if settings.escalation_model:
            escalator = make_answer_agent("escalator", settings.escalation_model, settings.prompt_version, "escalated_raw")
        super().__init__(
            name="contractlens",
            description="Grounded contract question answering with self-verification.",
            index=index,
            verifier=verifier,
            settings=settings,
            answerer=answerer,
            escalator=escalator,
            sub_agents=[a for a in (answerer, escalator) if a is not None],
        )

    def _event(self, ctx: InvocationContext, state: dict | None = None, text: str | None = None) -> Event:
        content = types.Content(role="model", parts=[types.Part(text=text)]) if text else None
        return Event(author=self.name, invocation_id=ctx.invocation_id, content=content, actions=EventActions(state_delta=state or {}))

    def _verify(self, draft: Draft, hits: list[SearchHit], question: str):
        with span("verify", "PARSER", {"answer": draft.answer, "quotes": draft.quotes}) as s:
            verification = self.verifier.verify(draft, [h.chunk.text for h in hits], question, self.index.relevance)
            if s:
                s.set_outputs(verification.to_dict())
        return verification

    def _resolve_contract(self, question: str, given: str | None) -> tuple[str | None, str]:
        """Answer from exactly one contract: the given one, one named in the question, or the most relevant one."""
        if given:
            return given, "given"
        mentioned = _mentioned_contract(question, self.index.doc_ids())
        if mentioned:
            return mentioned, "mentioned"
        top = self.index.search(question, top_k=1)
        return (top[0].chunk.doc_id, "auto") if top else (None, "none")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        user_text = ctx.user_content.parts[0].text if ctx.user_content and ctx.user_content.parts else ""
        question = (user_text or state.get("question") or "").strip()
        contract_id, contract_source = self._resolve_contract(question, state.get("contract_id"))
        cfg: Settings = self.settings

        with span("retrieve", "RETRIEVER", {"question": question, "contract_id": contract_id}) as s:
            hits = self.index.search(question, top_k=cfg.top_k, doc_id=contract_id)
            if s:
                s.set_outputs([{"chunk_id": h.chunk.chunk_id, "score": h.score} for h in hits])
        context = format_context([h.chunk.text for h in hits]) if hits else "(no excerpts retrieved)"
        yield self._event(ctx, {"question": question, "context": context})

        async for event in self.answerer.run_async(ctx):
            yield event
        llm_calls = 1
        draft = parse_draft(state.get("draft_raw", ""))
        verification = self._verify(draft, hits, question) if draft.found else None
        escalated = False

        if not draft.found:
            decision = "not_found"
        elif not cfg.use_verifier or verification.confidence >= cfg.confidence_threshold:
            decision = "answered"
        elif self.escalator is not None:
            escalated = True
            async for event in self.escalator.run_async(ctx):
                yield event
            llm_calls += 1
            draft = parse_draft(state.get("escalated_raw", ""))
            verification = self._verify(draft, hits, question) if draft.found else None
            if not draft.found:
                decision = "not_found"
            elif verification.confidence >= cfg.confidence_threshold:
                decision = "answered"
            else:
                decision = "abstained"
        else:
            decision = "abstained"

        supported = verification.supported_quotes if verification else []
        result = {
            "question": question,
            "contract_id": contract_id,
            "contract_source": contract_source,
            "decision": decision,
            "answer": draft.answer if decision == "answered" else "",
            "quotes": draft.quotes if decision == "answered" else [],
            "raw_answer": draft.answer,
            "raw_quotes": draft.quotes,
            "parse_ok": draft.parse_ok,
            "citations": _citations(draft, hits, supported) if decision == "answered" else [],
            "confidence": round(verification.confidence, 4) if verification else 0.0,
            "verification": verification.to_dict() if verification else None,
            "escalated": escalated,
            "llm_calls": llm_calls,
            "retrieved": [{"chunk_id": h.chunk.chunk_id, "start": h.chunk.start, "end": h.chunk.end, "doc_id": h.chunk.doc_id} for h in hits],
            "model": cfg.escalation_model if escalated else cfg.model,
        }
        yield self._event(ctx, {"result": result}, _format_reply(result))
