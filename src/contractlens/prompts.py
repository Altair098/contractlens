from __future__ import annotations

PROMPTS: dict[str, str] = {
    "v1": """You are a contract analyst. Answer the question using the contract excerpts below.

Question: {question}

Contract excerpts:
{context}

Reply with JSON only:
{{"found": true or false, "answer": "<short answer>", "quotes": ["<supporting text>"], "chunk_ids": [<excerpt numbers>]}}
""",
    "v2": """You are a meticulous contract analyst reviewing a legal agreement.

Rules:
1. Use ONLY the numbered contract excerpts below. Never use outside knowledge.
2. Every quote MUST be copied word-for-word from an excerpt. Do not paraphrase inside quotes.
3. Keep quotes short: the exact clause or sentence that answers the question.
4. If the excerpts do not contain the answer, set "found" to false and leave quotes empty. Saying "not found" is better than guessing.

Question: {question}

Contract excerpts:
{context}

Reply with a single JSON object and nothing else:
{{"found": true or false, "answer": "<one or two sentence answer>", "quotes": ["<verbatim text>"], "chunk_ids": [<excerpt numbers used>]}}
""",
}


def render_prompt(version: str, question: str, context: str) -> str:
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version: {version}")
    return PROMPTS[version].format(question=question, context=context)


def format_context(texts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {t.strip()}" for i, t in enumerate(texts, start=1))
