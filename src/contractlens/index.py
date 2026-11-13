from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder:
    name: str

    def fit(self, texts: list[str]) -> None:
        pass

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model_name}"
        self.model = SentenceTransformer(model_name)
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 200)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0]


class TfidfEmbedder(Embedder):
    """Offline fallback that needs no model download."""

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def _transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]


def get_embedder(spec: str) -> Embedder:
    if spec == "tfidf":
        return TfidfEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    raise ValueError(f"unknown embedder: {spec}")


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    dense_score: float
    rank: int


class HybridIndex:
    """Dense + BM25 retrieval fused with reciprocal rank fusion, filterable by document."""

    def __init__(self, embedder: Embedder, rrf_k: int = 60):
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.embeddings = np.asarray(self.embedder.embed_documents(texts), dtype=np.float32)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def add(self, chunks: list[Chunk]) -> None:
        if isinstance(self.embedder, TfidfEmbedder) or not self.chunks:
            self.build(self.chunks + list(chunks))
            return
        new = np.asarray(self.embedder.embed_documents([c.text for c in chunks]), dtype=np.float32)
        self.chunks.extend(chunks)
        self.embeddings = np.vstack([self.embeddings, new])
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def relevance(self, query: str, text: str) -> float:
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        d = np.asarray(self.embedder.embed_documents([text])[0], dtype=np.float32)
        return float(q @ d)

    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.chunks})

    def search(self, query: str, top_k: int = 4, doc_id: str | None = None) -> list[SearchHit]:
        if not self.chunks:
            return []
        candidates = np.array(
            [i for i, c in enumerate(self.chunks) if doc_id is None or c.doc_id == doc_id], dtype=int
        )
        if candidates.size == 0:
            return []

        dense = self.embeddings[candidates] @ np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        sparse = np.asarray(self.bm25.get_scores(tokenize(query)))[candidates]

        dense_rank = np.argsort(-dense)
        sparse_rank = np.argsort(-sparse)
        fused = np.zeros(candidates.size)
        for ranking in (dense_rank, sparse_rank):
            fused[ranking] += 1.0 / (self.rrf_k + np.arange(1, candidates.size + 1))

        order = np.argsort(-fused)[:top_k]
        return [
            SearchHit(self.chunks[candidates[i]], float(fused[i]), float(dense[i]), rank)
            for rank, i in enumerate(order)
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "rrf_k": self.rrf_k}))

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> "HybridIndex":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        embedder = embedder or get_embedder(meta["embedder"])
        if embedder.name != meta["embedder"]:
            raise ValueError(f"index was built with {meta['embedder']}, got {embedder.name}")
        index = cls(embedder, rrf_k=meta["rrf_k"])
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            index.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        texts = [c.text for c in index.chunks]
        embedder.fit(texts)
        index.embeddings = np.load(directory / "embeddings.npy")
        index.bm25 = BM25Okapi([tokenize(t) for t in texts])
        return index
