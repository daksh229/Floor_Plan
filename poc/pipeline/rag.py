"""RAG over the electrical symbol library.

The library (symbol_library/library.json) is small (~15 entries) but each
entry carries 4-5 aliases and a free-text spec. We embed each entry as a
single document concatenating canonical_name + aliases + spec, build a FAISS
inner-product index over the L2-normalised embeddings, and query with the
same embedding model.

This is real retrieval: query semantics (e.g. "main switchboard") map to the
right entry (distribution_board) even when the literal class name differs.
The embeddings + index are cached to disk so the first run pays the model
download / compute once and subsequent runs are instant.

Used by:
  - Phase 4 (BOM assembly): map every Detection.symbol_class to its library entry
  - Phase 5 (Streamlit UI): show the top-K matches with similarity scores
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .schemas import RAGMatch, SymbolWithMatches, Detection


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 80MB, fast, decent
_SYMBOL_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "symbol_library"
LIBRARY_PATH = _SYMBOL_LIBRARY_DIR / "library.json"
USER_ADDITIONS_PATH = _SYMBOL_LIBRARY_DIR / "user_additions.json"
USER_SYMBOL_DIR = _SYMBOL_LIBRARY_DIR / "user"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".rag_cache"


@dataclass
class _LibraryEntry:
    key: str
    canonical_name: str
    aliases: list[str]
    unit: str
    spec: str
    indicative_cost_aud: float

    def to_document(self) -> str:
        """The text that gets embedded. Concatenating aliases is what makes the
        embedding non-trivial — without aliases, retrieval over 15 distinct
        canonical names would be too easy to be interesting on the demo."""
        alias_str = ", ".join(self.aliases) if self.aliases else ""
        return (
            f"{self.canonical_name}. "
            f"Also known as: {alias_str}. "
            f"Specification: {self.spec}."
        )


def _entries_from(raw: dict) -> dict[str, _LibraryEntry]:
    """Parse a library-shaped dict into {key: _LibraryEntry}."""
    out: dict[str, _LibraryEntry] = {}
    for key, data in (raw.get("symbols") or {}).items():
        out[key] = _LibraryEntry(
            key=key,
            canonical_name=data["canonical_name"],
            aliases=list(data.get("aliases", [])),
            unit=data.get("unit", "ea"),
            spec=data.get("spec", ""),
            indicative_cost_aud=float(data.get("indicative_cost_aud", 0.0)),
        )
    return out


def _load_library(
    library_path: Path = LIBRARY_PATH,
    user_additions_path: Path = USER_ADDITIONS_PATH,
) -> list[_LibraryEntry]:
    """Built-in library merged with user additions (if any).

    User entries with the same key as a built-in entry override the built-in.
    This is the "user wins" semantic from the Phase 6 design discussion.
    """
    builtin = _entries_from(json.loads(library_path.read_text(encoding="utf-8")))
    merged: dict[str, _LibraryEntry] = dict(builtin)
    if user_additions_path.exists():
        try:
            user = _entries_from(json.loads(user_additions_path.read_text(encoding="utf-8")))
            merged.update(user)  # user wins on key collision
        except (json.JSONDecodeError, KeyError):
            # Bad/partial user file: don't poison the built-in library
            pass
    # Stable order: built-ins first then any user-only keys, alphabetically inside each
    builtin_keys = sorted(builtin.keys())
    user_only_keys = sorted(k for k in merged if k not in builtin)
    return [merged[k] for k in builtin_keys + user_only_keys]


def user_library_present() -> bool:
    """True if poc/symbol_library/user_additions.json exists with at least one symbol."""
    if not USER_ADDITIONS_PATH.exists():
        return False
    try:
        raw = json.loads(USER_ADDITIONS_PATH.read_text(encoding="utf-8"))
        return bool(raw.get("symbols"))
    except (json.JSONDecodeError, OSError):
        return False


def get_builtin_entry(library_key: str) -> dict | None:
    """Look up a single entry from the BUILT-IN library only (not user pack).

    Used by the legend extractor to backfill sensible defaults (unit, cost)
    when a card-grid layout — which has no unit/cost columns — happens to
    have a key that matches one of our 15 built-in symbols. The user can
    still override in the Tab 6 preview before ingesting.
    """
    try:
        raw = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (raw.get("symbols") or {}).get(library_key)


def _cache_key(model_name: str, entries: list[_LibraryEntry]) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for e in entries:
        h.update(e.key.encode("utf-8"))
        h.update(e.to_document().encode("utf-8"))
    return h.hexdigest()[:16]


class SymbolRAG:
    """Embeds the library once, then serves k-nearest queries.

    If `sentence-transformers` is unavailable or its model can't be loaded
    (no internet on the demo machine, etc.) the RAG transparently falls
    back to a deterministic string-similarity lookup over the same
    alias-enriched library documents. The fallback returns scores in the
    same [-1, 1] range (mapped from SequenceMatcher.ratio() in [0, 1]) so
    downstream code is mode-agnostic.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        library_path: Path = LIBRARY_PATH,
        user_additions_path: Path = USER_ADDITIONS_PATH,
        cache_dir: Path = CACHE_DIR,
        verbose: bool = False,
        allow_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.library_path = library_path
        self.user_additions_path = user_additions_path
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.allow_fallback = allow_fallback

        self.entries: list[_LibraryEntry] = _load_library(
            library_path, user_additions_path,
        )
        self._index = None        # faiss.IndexFlatIP, built lazily
        self._embeddings: np.ndarray | None = None
        self._model = None         # sentence_transformers.SentenceTransformer
        self._built = False
        self._fallback_active = False

    # ---- public API ----

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def mode(self) -> str:
        if not self._built:
            return "unbuilt"
        return "deterministic-fallback" if self._fallback_active else "embeddings"

    def build(self) -> None:
        """Compute (or load cached) embeddings + FAISS index. Idempotent."""
        if self._built:
            return

        cache_id = _cache_key(self.model_name, self.entries)
        emb_path = self.cache_dir / f"emb_{cache_id}.pkl"

        try:
            if emb_path.exists():
                with emb_path.open("rb") as f:
                    self._embeddings = pickle.load(f)
                if self.verbose:
                    print(f"[rag] loaded cached embeddings from {emb_path.name}")
            else:
                self._embeddings = self._embed([e.to_document() for e in self.entries])
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with emb_path.open("wb") as f:
                    pickle.dump(self._embeddings, f)
                if self.verbose:
                    print(f"[rag] cached embeddings -> {emb_path.name}")

            import faiss  # local import so module imports without faiss
            dim = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            # IndexFlatIP + L2-normalised vectors = cosine similarity
            self._index.add(self._embeddings.astype(np.float32))
        except Exception as exc:  # noqa: BLE001
            if not self.allow_fallback:
                raise
            if self.verbose:
                print(f"[rag] embedding path failed ({type(exc).__name__}: {exc}); "
                      "falling back to deterministic string-similarity lookup.")
            self._embeddings = None
            self._index = None
            self._fallback_active = True

        self._built = True

    def match(self, query: str, k: int = 3) -> list[RAGMatch]:
        if not self._built:
            self.build()
        if self._fallback_active:
            return self._deterministic_match(query, k)
        return self._embedding_match(query, k)

    def match_detections(
        self, detections: Iterable[Detection], k: int = 3
    ) -> list[SymbolWithMatches]:
        """Convenience: take Phase-2 detections and attach top-K library matches."""
        return [
            SymbolWithMatches(detection=d, matches=self.match(d.symbol_class, k=k))
            for d in detections
        ]

    # ---- internals ----

    def _embedding_match(self, query: str, k: int) -> list[RAGMatch]:
        assert self._index is not None and self._embeddings is not None
        q_vec = self._embed([self._normalise_query(query)])
        scores, idxs = self._index.search(q_vec.astype(np.float32), k)
        out: list[RAGMatch] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.entries):
                continue
            e = self.entries[idx]
            out.append(
                RAGMatch(
                    library_key=e.key,
                    canonical_name=e.canonical_name,
                    similarity=float(score),
                    unit=e.unit,
                    spec=e.spec,
                    indicative_cost_aud=e.indicative_cost_aud,
                )
            )
        return out

    def _deterministic_match(self, query: str, k: int) -> list[RAGMatch]:
        """SequenceMatcher.ratio() over normalised query vs normalised entry doc.

        Used when embeddings aren't available. Same RAGMatch contract — the
        similarity field is in [0, 1] (mapped from SequenceMatcher).
        """
        q_norm = _normalise_text(self._normalise_query(query))
        ranked: list[tuple[float, _LibraryEntry]] = []
        for entry in self.entries:
            doc_norm = _normalise_text(entry.to_document())
            ratio = SequenceMatcher(None, q_norm, doc_norm).ratio()
            # also boost direct alias hits — they're a near-certain match
            for alias in entry.aliases:
                if _normalise_text(alias) in q_norm or q_norm in _normalise_text(alias):
                    ratio = max(ratio, 0.85)
                    break
            if _normalise_text(entry.canonical_name).split()[0] in q_norm:
                ratio = max(ratio, 0.75)
            ranked.append((ratio, entry))
        ranked.sort(key=lambda t: t[0], reverse=True)
        return [
            RAGMatch(
                library_key=e.key,
                canonical_name=e.canonical_name,
                similarity=round(score, 4),
                unit=e.unit,
                spec=e.spec,
                indicative_cost_aud=e.indicative_cost_aud,
            )
            for score, e in ranked[:k]
        ]

    def _model_instance(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._model_instance()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @staticmethod
    def _normalise_query(query: str) -> str:
        """Convert detection-class names (snake_case) to natural language for
        better matching, while leaving free-text queries alone."""
        q = query.strip()
        if "_" in q and " " not in q:
            return q.replace("_", " ")
        return q


def _normalise_text(text: str) -> str:
    """Lowercase, strip non-alphanumerics, collapse whitespace. Used by the
    deterministic fallback so 'WP-Switch' and 'wp switch' compare equal."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
