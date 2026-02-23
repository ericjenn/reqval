"""
llm_provider.py — Unified LLM & Embedding Provider
=====================================================
Single module that constructs the chat model and the embedding function
for the entire pipeline. Switch backends via a single env variable.

──────────────────────────────────────────────────────────────
CONFIGURATION  (.env or environment)
──────────────────────────────────────────────────────────────

  LLM_PROVIDER = openai        ← default
  LLM_PROVIDER = ollama

  ── OpenAI ──────────────────────────────────────────────────
  OPENAI_API_KEY     = sk-...                    (required)
  OPENAI_MODEL       = gpt-4o                    (default)
  OPENAI_EMBED_MODEL = text-embedding-3-small    (default)

  ── Ollama ──────────────────────────────────────────────────
  OLLAMA_BASE_URL    = http://localhost:11434     (default)
  OLLAMA_MODEL       = llama3.1                  (default)
  OLLAMA_EMBED_MODEL = nomic-embed-text          (default)

  Pull models before first use:
    ollama pull llama3.1
    ollama pull nomic-embed-text

──────────────────────────────────────────────────────────────
PUBLIC API
──────────────────────────────────────────────────────────────
  get_llm(temperature)  → LangChain chat model (drop-in for any agent)
  get_embedder()        → fn(list[str]) -> np.ndarray  (L2-normalised)
  EMBEDDING_DIM         → int  — vector size for the active embed model
  provider_info()       → dict — human-readable config summary
  validate_provider()   → None — raises ValueError if misconfigured
"""

from __future__ import annotations

import os
import numpy as np
from typing import List, Callable
from dotenv import load_dotenv

load_dotenv()

# ── Active configuration ──────────────────────────────────────────────────────

PROVIDER          = os.getenv("LLM_PROVIDER",       "openai").strip().lower()
OPENAI_MODEL      = os.getenv("OPENAI_MODEL",        "gpt-4o").strip()
OPENAI_EMBED      = os.getenv("OPENAI_EMBED_MODEL",  "text-embedding-3-small").strip()
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434").strip()
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",        "llama3.1").strip()
OLLAMA_EMBED      = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text").strip()

# Known embedding dimensions — used to size the FAISS index correctly
_EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text":        768,
    "mxbai-embed-large":      1024,
    "all-minilm":              384,
    "bge-m3":                 1024,
}

EMBEDDING_DIM: int = (
    _EMBED_DIMS.get(OPENAI_EMBED, 1536) if PROVIDER == "openai"
    else _EMBED_DIMS.get(OLLAMA_EMBED, 768)
)


# ── Chat model factory ────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.1):
    """
    Return a LangChain chat model for the active provider.
    The returned object is a drop-in replacement anywhere ChatOpenAI was used.
    """
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set.\n"
                "Add it to your .env file:  OPENAI_API_KEY=sk-..."
            )
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=temperature,
            api_key=api_key,
        )

    elif PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{PROVIDER}'. Valid values: 'openai' or 'ollama'"
        )


# ── Embedding factory ─────────────────────────────────────────────────────────

def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def get_embedder() -> Callable[[List[str]], np.ndarray]:
    """
    Return an embedding function:
        fn(texts: list[str]) -> np.ndarray  shape (N, EMBEDDING_DIM), L2-normalised

    OpenAI: calls the OpenAI embeddings API in batches of 100.
    Ollama: calls the local Ollama /api/embed endpoint (one request per batch).
    """
    if PROVIDER == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)

        def _embed_openai(texts: List[str]) -> np.ndarray:
            BATCH = 100
            vectors: List[List[float]] = []
            for i in range(0, len(texts), BATCH):
                resp = client.embeddings.create(
                    model=OPENAI_EMBED,
                    input=texts[i : i + BATCH],
                )
                vectors.extend([item.embedding for item in resp.data])
            return _l2_normalise(np.array(vectors, dtype=np.float32))

        return _embed_openai

    elif PROVIDER == "ollama":
        import requests

        def _embed_ollama(texts: List[str]) -> np.ndarray:
            # Use the /api/embed endpoint (Ollama ≥ 0.1.26) which accepts a list
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": OLLAMA_EMBED, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns {"embeddings": [[...], [...]]}
            vectors = data.get("embeddings") or data.get("embedding")
            if vectors is None:
                raise ValueError(f"Unexpected Ollama embed response: {data}")
            # Older Ollama versions return a single vector for single-text requests
            if isinstance(vectors[0], float):
                vectors = [vectors]
            return _l2_normalise(np.array(vectors, dtype=np.float32))

        return _embed_ollama

    else:
        raise ValueError(f"Unknown LLM_PROVIDER='{PROVIDER}'")


# ── Startup validation ────────────────────────────────────────────────────────

def validate_provider() -> None:
    """
    Verify the provider is reachable and required models are available.
    Call once at startup (main.py). Raises ValueError with a fix hint on failure.
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set.\n"
                "Add it to your .env file:  OPENAI_API_KEY=sk-..."
            )

    elif PROVIDER == "ollama":
        import requests
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        except Exception as exc:
            raise ValueError(
                f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
                f"  Error: {exc}\n"
                f"  Fix:   ollama serve"
            )

        missing = []
        if OLLAMA_MODEL.split(":")[0] not in available:
            missing.append(f"  ollama pull {OLLAMA_MODEL}")
        if OLLAMA_EMBED.split(":")[0] not in available:
            missing.append(f"  ollama pull {OLLAMA_EMBED}")
        if missing:
            raise ValueError(
                "Ollama is running but required models are not pulled:\n"
                + "\n".join(missing)
            )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{PROVIDER}'. Valid values: 'openai' or 'ollama'"
        )


def provider_info() -> dict:
    """Return a dict describing the active configuration (for banners/logs)."""
    if PROVIDER == "openai":
        return {
            "provider":    "OpenAI",
            "llm_model":   OPENAI_MODEL,
            "embed_model": OPENAI_EMBED,
            "embed_dim":   EMBEDDING_DIM,
        }
    return {
        "provider":    "Ollama",
        "llm_model":   OLLAMA_MODEL,
        "embed_model": OLLAMA_EMBED,
        "embed_dim":   EMBEDDING_DIM,
        "base_url":    OLLAMA_BASE_URL,
    }
