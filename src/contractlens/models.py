from __future__ import annotations

import json
import re
from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from .index import tokenize

_EXCERPT_RE = re.compile(r"^\[(\d+)\]\s(.*?)(?=^\[\d+\]\s|\Z)", re.DOTALL | re.MULTILINE)
_QUESTION_RE = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.;:])\s+|\n+")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "this", "that", "what", "does", "say",
    "about", "contract", "specifically", "is", "are", "be", "any", "with", "by", "if", "as", "at", "it",
}


class ExtractiveFakeLlm(BaseLlm):
    """Deterministic offline model: returns the excerpt sentence that best overlaps the question.

    Used for tests and smoke runs without Ollama or an API key.
    """

    model: str = "fake-extractive"
    min_overlap: int = 2

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        prompt = str(llm_request.config.system_instruction or "")
        question_match = _QUESTION_RE.search(prompt)
        question = question_match.group(1) if question_match else ""
        q_tokens = {t for t in tokenize(question) if t not in _STOPWORDS}

        best = (0, "", 0)
        for match in _EXCERPT_RE.finditer(prompt.split("Contract excerpts:", 1)[-1]):
            excerpt_id, body = int(match.group(1)), match.group(2)
            for sentence in _SENTENCE_RE.split(body):
                sentence = sentence.strip()
                if len(sentence) < 15 or sentence.startswith("Reply with"):
                    continue
                overlap = len(q_tokens & set(tokenize(sentence)))
                if overlap > best[0]:
                    best = (overlap, sentence, excerpt_id)

        if best[0] >= self.min_overlap:
            payload = {"found": True, "answer": best[1], "quotes": [best[1]], "chunk_ids": [best[2]]}
        else:
            payload = {"found": False, "answer": "Not found in the contract.", "quotes": [], "chunk_ids": []}
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=json.dumps(payload))]))


def is_native_gemini(name: str) -> bool:
    return name.startswith("gemini")


def resolve_model(name: str, json_mode: bool = True):
    """Map a model spec to an ADK model.

    gemini-*        -> native ADK Gemini (needs GOOGLE_API_KEY, free tier works)
    fake            -> offline extractive model
    anything else   -> LiteLLM, e.g. ollama_chat/gemma3:4b (JSON mode maps to Ollama's format=json)
    """
    if name == "fake":
        return ExtractiveFakeLlm()
    if is_native_gemini(name):
        return name
    from google.adk.models.lite_llm import LiteLlm

    if json_mode:
        return LiteLlm(model=name, response_format={"type": "json_object"})
    return LiteLlm(model=name)
