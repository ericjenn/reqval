"""
RAG Engine for ARP4754A Validator
==================================
Ingests system-specific documents (PDF, TXT, MD, DOCX) into a persistent
FAISS vector store. Each validation agent queries this store to:
  1. Use consistent, project-specific vocabulary in its analysis
  2. Ground recommendations in actual system architecture and constraints
  3. Detect terminology mismatches between requirements and reference docs

Supported input formats : .txt, .md, .pdf, .docx
Vector store            : FAISS (local files, no server, no pydantic)
Embeddings              : OpenAI text-embedding-3-small (via openai SDK directly)
"""

from __future__ import annotations

import os
import pickle
import hashlib
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
import numpy as np

# ── LangChain: document loading and splitting ONLY ────────────────────────
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings are provided by llm_provider (OpenAI or Ollama)
from llm_provider import get_embedder, EMBEDDING_DIM as _PROVIDER_EMBED_DIM

# ── FAISS ─────────────────────────────────────────────────────────────────
import faiss

load_dotenv()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DEFAULT_STORE_DIR  = "./faiss_store"
INDEX_FILE         = "index.faiss"
META_FILE          = "metadata.pkl"
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 150
# EMBEDDING_DIM comes from llm_provider so it matches the active embed model
EMBEDDING_DIM      = _PROVIDER_EMBED_DIM
EMBED_BATCH_SIZE   = 100


# ─────────────────────────────────────────────
# RAGEngine
# ─────────────────────────────────────────────

class RAGEngine:
    """
    FAISS-backed RAG engine. 

    Persistence
    -----------
    Two files are written to `store_dir`:
      index.faiss   – the FAISS flat inner-product index
      metadata.pkl  – list of {text, source_file, page} dicts, one per vector

    The index uses inner-product (IP) similarity on L2-normalised vectors,
    which is equivalent to cosine similarity.

    Typical usage
    -------------
        rag = RAGEngine()
        rag.ingest_folder("./system_docs/")
        context = rag.query("navigation system position accuracy")
    """

    def __init__(self, store_dir: str = DEFAULT_STORE_DIR):
        self.store_dir  = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._embed_fn  = get_embedder()   # callable: list[str] -> np.ndarray
        self._index     : faiss.IndexFlatIP = None   # type: ignore[assignment]
        self._metadata  : List[dict]        = []     # parallel list to index rows
        self._hashes    : set[str]          = set()  # per-file SHA-256

        self._load()

    # ── Persistence ───────────────────────────

    def _index_path(self) -> Path:
        return self.store_dir / INDEX_FILE

    def _meta_path(self) -> Path:
        return self.store_dir / META_FILE

    def _load(self):
        """Load existing FAISS index + metadata from disk if present."""
        if self._index_path().exists() and self._meta_path().exists():
            try:
                self._index = faiss.read_index(str(self._index_path()))
                with open(self._meta_path(), "rb") as fh:
                    saved = pickle.load(fh)
                self._metadata = saved.get("metadata", [])
                self._hashes   = saved.get("hashes", set())
                n = self._index.ntotal
                if n > 0:
                    print(f"  [RAG] Loaded existing store ({n} chunks) from {self.store_dir}")
                return
            except Exception as exc:
                print(f"  [RAG] Could not load existing store ({exc}), starting fresh.")

        # Fresh index
        self._index    = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._metadata = []
        self._hashes   = set()

    def _save(self):
        """Persist the FAISS index and metadata to disk."""
        faiss.write_index(self._index, str(self._index_path()))
        with open(self._meta_path(), "wb") as fh:
            pickle.dump({"metadata": self._metadata, "hashes": self._hashes}, fh)

    # ── Embeddings ────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using the active provider (OpenAI or Ollama).
        Returns L2-normalised float32 array of shape (N, EMBEDDING_DIM).
        """
        return self._embed_fn(texts)

    # ── Document loading ──────────────────────

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def _load_document(self, path: Path) -> List[Document]:
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif suffix in (".txt", ".md"):
                loader = TextLoader(str(path), encoding="utf-8")
            elif suffix in (".docx", ".doc"):
                loader = Docx2txtLoader(str(path))
            else:
                print(f"  [RAG] Unsupported file type, skipping: {path.name}")
                return []
            docs = loader.load()
            for d in docs:
                d.metadata.setdefault("source_file", path.name)
            return docs
        except Exception as exc:
            print(f"  [RAG] Could not load {path.name}: {exc}")
            return []

    # ── Public API ────────────────────────────

    def ingest_file(self, file_path: str | Path) -> int:
        """
        Chunk, embed, and index a single document.
        Returns the number of new chunks added (0 if already indexed).
        """
        path  = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        fhash = self._file_hash(path)
        if fhash in self._hashes:
            print(f"  [RAG] Already indexed (skipping): {path.name}")
            return 0

        raw_docs = self._load_document(path)
        if not raw_docs:
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(raw_docs)
        if not chunks:
            return 0

        texts = [c.page_content for c in chunks]
        vecs  = self._embed(texts)

        self._index.add(vecs)
        for chunk, text in zip(chunks, texts):
            self._metadata.append({
                "text":        text,
                "source_file": chunk.metadata.get("source_file", path.name),
                "page":        chunk.metadata.get("page", ""),
            })

        self._hashes.add(fhash)
        self._save()
        print(f"  [RAG] Ingested {path.name} → {len(chunks)} chunks")
        return len(chunks)

    def ingest_folder(self, folder_path: str | Path) -> int:
        """
        Recursively ingest all supported documents from a folder.
        Returns total number of new chunks added.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        supported = {".txt", ".md", ".pdf", ".docx", ".doc"}
        files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in supported)

        if not files:
            print(f"  [RAG] No supported documents found in {folder_path}")
            return 0

        total = sum(self.ingest_file(f) for f in files)
        print(f"  [RAG] Total chunks in store: {self.chunk_count()}")
        return total

    def query(
        self,
        query_text: str,
        k: int = 5,
        filter_source: Optional[str] = None,
    ) -> str:
        """
        Retrieve the top-k most semantically relevant chunks.

        Args:
            query_text:    Natural language query
            k:             Number of results to return
            filter_source: If set, restrict results to this source filename

        Returns:
            Formatted string of passages ready for LLM injection,
            or empty string if the store is empty.
        """
        if not self.is_ready():
            return ""

        qvec = self._embed([query_text])   # shape (1, DIM)

        # Fetch more candidates when filtering, to ensure k results after filter
        fetch_k = k * 5 if filter_source else k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(qvec, fetch_k)

        passages = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            if filter_source and meta.get("source_file") != filter_source:
                continue
            source = meta.get("source_file", "unknown")
            page   = meta.get("page", "")
            header = f"[Source: {source}" + (f", p.{page}" if page else "") + "]"
            passages.append(f"{header}\n{meta['text'].strip()}")
            if len(passages) >= k:
                break

        return "\n\n---\n\n".join(passages)

    def get_glossary(self, max_chunks: int = 10) -> str:
        """Retrieve terminology / glossary chunks from the store."""
        return self.query(
            "system terminology definitions glossary acronyms units measurements",
            k=max_chunks,
        )

    def is_ready(self) -> bool:
        """True if the store contains at least one chunk."""
        return self._index is not None and self._index.ntotal > 0

    def chunk_count(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    def clear(self):
        """Delete the index and metadata, reset to empty state."""
        self._index    = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._metadata = []
        self._hashes   = set()
        self._save()
        print("  [RAG] Vector store cleared.")

    def list_sources(self) -> List[str]:
        """Return sorted list of unique source file names in the store."""
        sources = {m.get("source_file", "unknown") for m in self._metadata}
        return sorted(sources)


# ─────────────────────────────────────────────
# Module-level singleton (shared across agents)
# ─────────────────────────────────────────────

_rag_instance: Optional[RAGEngine] = None


def get_rag(store_dir: str = DEFAULT_STORE_DIR) -> RAGEngine:
    """Return the module-level RAG singleton, creating it if necessary."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine(store_dir=store_dir)
    return _rag_instance


def reset_rag():
    """Reset the singleton (e.g. between test runs or after --clear-store)."""
    global _rag_instance
    _rag_instance = None
