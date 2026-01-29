from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class RAGDoc:
    content: str
    score: Optional[float] = None


class RAGIndex:
    def __init__(
        self,
        db_path: str,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        overlap: int = 150,
        fetch_k: int = 30,
        top_n: int = 8,
        use_rerank: bool = True,
    ):
        self.db_path = Path(db_path)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.fetch_k = fetch_k
        self.top_n = top_n
        self.use_rerank = use_rerank

        self._embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self._vectorstore: Optional[FAISS] = None
        self._retriever = None

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        import hashlib

        return hashlib.sha256(data).hexdigest()

    def _hashes_path(self) -> Path:
        return self.db_path / "hashes.json"

    def _index_exists(self) -> bool:
        return (self.db_path / "index.faiss").exists() and (self.db_path / "index.pkl").exists()

    def _load_hashes(self) -> list[str]:
        try:
            return json.loads(self._hashes_path().read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_hashes(self, hashes: list[str]) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._hashes_path().write_text(json.dumps(hashes, ensure_ascii=False), encoding="utf-8")

    def _build_vectorstore(self, docs: Sequence[dict]) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.overlap
        )
        lc_docs: List[Document] = []
        for item in docs:
            text = item.get("text", "")
            source = item.get("source", "uploaded_document")
            if not text:
                continue
            for doc in splitter.create_documents([text]):
                doc.metadata["source"] = source
                lc_docs.append(doc)
        if not lc_docs:
            raise ValueError("No text chunks to index.")

        self._vectorstore = FAISS.from_documents(lc_docs, self._embeddings)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(self.db_path))

    def _load_vectorstore(self) -> None:
        self._vectorstore = FAISS.load_local(
            str(self.db_path), self._embeddings, allow_dangerous_deserialization=True
        )

    def _build_retriever(self) -> None:
        if self._vectorstore is None:
            raise ValueError("Vectorstore is not initialized")

        base_retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self.fetch_k}
        )
        if self.use_rerank and os.environ.get("JINA_API_KEY"):
            compressor = JinaRerank(
                model="jina-reranker-v2-base-multilingual", top_n=self.top_n
            )
            self._retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            )
        else:
            self._retriever = base_retriever

    def ensure_index(self, docs: Sequence[dict], hashes: list[str]) -> None:
        if self._index_exists():
            if hashes and hashes == self._load_hashes():
                self._load_vectorstore()
                self._build_retriever()
                return

        self._build_vectorstore(docs)
        self._save_hashes(hashes)
        self._build_retriever()

    def search(self, query: str, top_k: int = 4) -> List[dict]:
        if self._retriever is None:
            return []
        docs = self._retriever.get_relevant_documents(query)
        results: List[dict] = []
        for doc in docs[:top_k]:
            results.append(
                {
                    "score": None,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "uploaded_document"),
                }
            )
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
        return "".join(parts)
