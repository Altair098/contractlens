from __future__ import annotations

import asyncio
import time
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agents import ContractLensAgent
from .config import Settings, get_settings
from .index import HybridIndex, get_embedder
from .tracking import span
from .verifier import NliScorer, Verifier

APP_NAME = "contractlens"


class ContractLens:
    def __init__(self, index: HybridIndex, verifier: Verifier, settings: Settings):
        self.index = index
        self.settings = settings
        self.agent = ContractLensAgent(index, verifier, settings)
        self.sessions = InMemorySessionService()
        self.runner = Runner(app_name=APP_NAME, agent=self.agent, session_service=self.sessions)

    @classmethod
    def from_settings(cls, settings: Settings | None = None, index: HybridIndex | None = None) -> "ContractLens":
        settings = settings or get_settings()
        if index is None:
            index = HybridIndex.load(settings.index_dir, get_embedder(settings.embedder))
        verifier = Verifier(NliScorer(settings.nli_model))
        return cls(index, verifier, settings)

    async def ask(self, question: str, contract_id: str | None = None) -> dict:
        start = time.perf_counter()
        session = await self.sessions.create_session(
            app_name=APP_NAME,
            user_id="api",
            session_id=uuid.uuid4().hex,
            state={"question": question, "contract_id": contract_id},
        )
        message = types.Content(role="user", parts=[types.Part(text=question)])
        with span("contractlens.ask", "AGENT", {"question": question, "contract_id": contract_id}) as s:
            async for _ in self.runner.run_async(user_id="api", session_id=session.id, new_message=message):
                pass
            final = await self.sessions.get_session(app_name=APP_NAME, user_id="api", session_id=session.id)
            result = dict(final.state["result"])
            result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            if s:
                s.set_outputs({k: result[k] for k in ("decision", "answer", "confidence", "escalated")})
        await self.sessions.delete_session(app_name=APP_NAME, user_id="api", session_id=session.id)
        return result

    def ask_sync(self, question: str, contract_id: str | None = None) -> dict:
        return asyncio.run(self.ask(question, contract_id))
