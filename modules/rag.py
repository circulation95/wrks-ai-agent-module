from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    return dot / denom if denom else 0.0


@dataclass(frozen=True)
class RAGChunk:
    content: str
    embedding: List[float]


class RAGIndex:
    def __init__(
        self,
        client: OpenAI,
        model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        overlap: int = 150,
    ):
        self.client = client
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[RAGChunk] = []

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def add_text(self, text: str) -> None:
        chunks = _chunk_text(text, self.chunk_size, self.overlap)
        if not chunks:
            return
        embeddings = self._embed_texts(chunks)
        for content, embedding in zip(chunks, embeddings):
            self.chunks.append(RAGChunk(content=content, embedding=embedding))

    def _embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    def search(self, query: str, top_k: int = 4) -> List[dict]:
        if not self.chunks:
            return []
        query_embedding = self._embed_texts([query])[0]
        scored = []
        for chunk in self.chunks:
            score = _cosine_similarity(query_embedding, chunk.embedding)
            scored.append((score, chunk.content))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, content in scored[:top_k]:
            results.append({"score": score, "content": content})
        return results

    def build_context(self, query: str, top_k: int = 4, max_chars: int = 2000) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        parts: List[str] = []
        total = 0
        for item in results:
            chunk = item["content"]
            if total + len(chunk) > max_chars:
                chunk = chunk[: max(0, max_chars - total)]
            if not chunk:
                break
            parts.append(chunk)
            total += len(chunk)
            if total >= max_chars:
                break
        return "\n\n".join(parts)
