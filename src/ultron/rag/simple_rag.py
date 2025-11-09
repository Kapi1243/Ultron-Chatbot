"""
Simple Retrieval-Augmented Generation (RAG) for local documents.
Indexes .md and .txt files and retrieves relevant chunks for a query.
Falls back gracefully if scikit-learn is unavailable.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class SimpleRAG:
    def __init__(self, files: List[str], chunk_size: int = 700, chunk_overlap: int = 100):
        self.files = [f for f in files if os.path.exists(f)]
        self.chunk_size = max(200, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.chunks: List[str] = []
        self.chunk_meta: List[Tuple[str, int]] = []  # (filepath, chunk_idx)
        self.vectorizer = None
        self.matrix = None
        self._build_index()

    def _read_text(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""

    def _split_chunks(self, text: str) -> List[str]:
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        i = 0
        while i < len(text):
            end = min(len(text), i + self.chunk_size)
            chunks.append(text[i:end])
            if end == len(text):
                break
            i = end - self.chunk_overlap
        return chunks

    def _build_index(self):
        for path in self.files:
            text = self._read_text(path)
            if not text:
                continue
            for idx, chunk in enumerate(self._split_chunks(text)):
                self.chunks.append(chunk)
                self.chunk_meta.append((path, idx))

        if SKLEARN_AVAILABLE and self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
            self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = 3, max_chars: int = 1200) -> str:
        if not self.chunks or not query or len(query.strip()) < 2:
            return ""

        if SKLEARN_AVAILABLE and self.vectorizer is not None and self.matrix is not None:
            try:
                q_vec = self.vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.matrix).flatten()
                idxs = sims.argsort()[::-1][:max(10, top_k)]
                selected: List[str] = []
                total = 0
                for i in idxs:
                    c = self.chunks[i]
                    if total + len(c) > max_chars and selected:
                        break
                    selected.append(c)
                    total += len(c)
                return "\n\n".join(selected[:top_k])
            except Exception:
                pass

        # Fallback: naive ranking by shared words
        q_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        scored = []
        for i, c in enumerate(self.chunks):
            c_words = set(re.findall(r"[a-zA-Z0-9]+", c.lower()))
            score = len(q_words & c_words)
            scored.append((score, i))
        scored.sort(reverse=True)
        selected: List[str] = []
        total = 0
        for score, i in scored[:max(10, top_k)]:
            c = self.chunks[i]
            if total + len(c) > max_chars and selected:
                break
            selected.append(c)
            total += len(c)
        return "\n\n".join(selected[:top_k])
