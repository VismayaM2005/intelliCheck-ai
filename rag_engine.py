"""
rag_engine.py — RAG-Powered Explanation Engine
Uses sentence-transformers for embeddings + FAISS for retrieval.
No external API key required — runs fully locally.
"""

import os
import re
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ──────────────────────────────────────────────
# VECTOR STORE
# ──────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        self.chunks: list[str] = []
        self.index = None
        self.embedder = None
        self._load_embedder()

    def _load_embedder(self):
        if SBERT_AVAILABLE:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.embedder = None

    def build_index(self, documents: list[str]):
        """Chunk docs and build FAISS index."""
        self.chunks = []
        for doc in documents:
            self.chunks.extend(self._chunk(doc))

        if not self.chunks or not self.embedder or not FAISS_AVAILABLE:
            return False

        embeddings = self.embedder.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        return True

    def _chunk(self, text: str, max_words: int = 80) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, buf, count = [], [], 0
        for s in sentences:
            words = s.split()
            if count + len(words) > max_words and buf:
                chunks.append(" ".join(buf))
                buf, count = [], 0
            buf.extend(words)
            count += len(words)
        if buf:
            chunks.append(" ".join(buf))
        return [c for c in chunks if len(c.split()) > 5]

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """Return top-k most relevant chunks for a query."""
        if not self.index or not self.embedder or not self.chunks:
            return []
        q_emb = self.embedder.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        distances, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]


# ──────────────────────────────────────────────
# LLM GENERATOR (local flan-t5)
# ──────────────────────────────────────────────

_generator = None

def _get_generator():
    global _generator
    if _generator is not None:
        return _generator
    if not TRANSFORMERS_AVAILABLE:
        return None
    for model in ["google/flan-t5-base", "google/flan-t5-small"]:
        try:
            _generator = pipeline(
                "text2text-generation",
                model=model,
                max_new_tokens=200,
            )
            return _generator
        except Exception:
            continue
    return None


def _call_llm(prompt: str, max_tokens: int = 200) -> str:
    gen = _get_generator()
    if gen is None:
        return ""
    try:
        out = gen(prompt, max_new_tokens=max_tokens, do_sample=False)
        return out[0].get("generated_text", "").strip()
    except Exception:
        return ""


# ──────────────────────────────────────────────
# RAG EXPLAIN
# ──────────────────────────────────────────────

def explain_similarity(
    flagged_sentence: str,
    matched_source: str,
    rag_engine: RAGEngine,
) -> dict:
    """
    Given a flagged sentence and its best-matching source sentence,
    retrieve relevant context and generate:
      - explanation of WHY they are similar
      - a summary of the source content
      - a safe academic rewrite suggestion
    """
    context_chunks = rag_engine.retrieve(flagged_sentence, top_k=3)
    context = " ".join(context_chunks)[:600] if context_chunks else matched_source[:400]

    # --- WHY similar ---
    overlap_words = _get_overlap_words(flagged_sentence, matched_source)
    why_prompt = (
        f"Context from source documents: {context}\n\n"
        f"Original sentence: \"{flagged_sentence}\"\n"
        f"Similar source sentence: \"{matched_source}\"\n"
        f"Shared key words: {', '.join(overlap_words[:8])}.\n"
        f"Explain briefly why these sentences are considered similar, "
        f"focusing on shared vocabulary and ideas:"
    )
    explanation = _call_llm(why_prompt, max_tokens=120)
    if not explanation:
        explanation = (
            f"These sentences share key terms: {', '.join(overlap_words[:6])}. "
            "The phrasing and core ideas closely mirror the source material."
        )

    # --- SUMMARY ---
    summary_prompt = (
        f"Summarize the following academic passage in 2-3 sentences:\n{context}"
    )
    summary = _call_llm(summary_prompt, max_tokens=100)
    if not summary:
        summary = context[:300] + "..." if len(context) > 300 else context

    # --- REWRITE ---
    rewrite_prompt = (
        f"Rewrite the following sentence in an original academic style, "
        f"keeping the meaning but using different vocabulary and structure:\n"
        f"Sentence: \"{flagged_sentence}\"\n"
        f"Rewritten version:"
    )
    rewrite = _call_llm(rewrite_prompt, max_tokens=100)
    if not rewrite:
        rewrite = _simple_rewrite(flagged_sentence)

    # --- KNOWLEDGE EXPANSION ---
    expand_prompt = (
        f"Given this topic from an academic text: \"{flagged_sentence[:120]}\"\n"
        f"Provide 2-3 related concepts or keywords a student should explore:"
    )
    expansion = _call_llm(expand_prompt, max_tokens=80)
    if not expansion:
        expansion = f"Related topics to explore: {', '.join(overlap_words[:4])}"

    return {
        "explanation": explanation,
        "source_summary": summary,
        "rewrite_suggestion": rewrite,
        "knowledge_expansion": expansion,
        "retrieved_chunks": context_chunks,
    }


def _get_overlap_words(s1: str, s2: str) -> list[str]:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "and", "in", "that", "for", "on", "with", "as",
        "it", "this", "by", "at", "from", "or", "but", "not", "have",
    }
    w1 = set(re.findall(r"\w+", s1.lower())) - stop
    w2 = set(re.findall(r"\w+", s2.lower())) - stop
    return list(w1 & w2)


def _simple_rewrite(sentence: str) -> str:
    starters = [
        "It can be observed that ",
        "Research indicates that ",
        "Scholars have noted that ",
        "Evidence suggests that ",
    ]
    import random
    return random.choice(starters) + sentence[0].lower() + sentence[1:]
