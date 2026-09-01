"""
text_only_rag.py — Ablation baseline: text-only RAG without multimodal or multi-agent components.

Differences from the full system:
  - No LangGraph multi-agent pipeline (SensorStatus / ValidationEngineer /
    Diagnostic / Strategy / Critic nodes all removed)
  - Retrieves ONLY type='text' ManualChunks (no images, no tables, no
    InteractionMemory lookup)
  - Single Qwen inference call (vs 3-5 in the full system)

Used to establish the incremental value of the full multimodal + multi-agent
architecture for the conference paper ablation study.
"""

import sys
import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_rag.ai_client import get_client, MODEL_CHAT
from unified_rag.embeddings.embedder import embedder
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import ManualChunk

TOP_K_TEXT = 3  # same as full system's top_k_text


@dataclass
class TextRAGTiming:
    embed_s: float = 0.0
    retrieve_s: float = 0.0
    llm_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.embed_s + self.retrieve_s + self.llm_s


@dataclass
class TextRAGResult:
    answer: str
    chunks_used: int
    pages: list
    timing: TextRAGTiming = field(default_factory=TextRAGTiming)


class TextOnlyRAGPipeline:
    """
    Minimal single-call RAG pipeline used as the ablation baseline.

    Architecture:
        query → embed (text-embedding-v4)
              → pgvector cosine search (text chunks only)
              → single Qwen call
              → answer
    """

    def __init__(self):
        self._client = get_client()

    def query(
        self,
        anomaly_description: str,
        manual_id: str,
        machine_id: Optional[str] = None,
    ) -> TextRAGResult:
        timing = TextRAGTiming()

        # ── Stage 1: embed query ───────────────────────────────────────────────
        t0 = time.perf_counter()
        query_emb = embedder.embed_text(anomaly_description)
        timing.embed_s = time.perf_counter() - t0

        # ── Stage 2: text-only vector retrieval ───────────────────────────────
        t1 = time.perf_counter()
        db = SessionLocal()
        chunks = []
        pages = []
        try:
            chunks = (
                db.query(ManualChunk)
                .filter(
                    ManualChunk.manual_id == manual_id,
                    ManualChunk.type == "text",          # TEXT ONLY — no images, no tables
                )
                .order_by(ManualChunk.embedding.cosine_distance(query_emb))
                .limit(TOP_K_TEXT)
                .all()
            )
            pages = sorted({c.page for c in chunks if c.page is not None})
        finally:
            db.close()
        timing.retrieve_s = time.perf_counter() - t1

        # ── Stage 3: build context and single LLM call ────────────────────────
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"--- Manual Excerpt {i + 1} (Page {chunk.page}) ---\n{chunk.content}\n\n"

        system_prompt = (
            f"You are an industrial diagnostic assistant for machine manual: {manual_id}.\n\n"
            "Using ONLY the manual excerpts below, provide a concise diagnostic summary "
            "(3-5 sentences) identifying the likely fault, its probable root cause, and "
            "the immediate corrective action. Do not hallucinate information not present "
            "in the excerpts.\n\n"
            f"MANUAL EXCERPTS:\n{context}"
        )

        t2 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Anomaly detected: {anomaly_description}"},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        timing.llm_s = time.perf_counter() - t2

        return TextRAGResult(
            answer=response.choices[0].message.content,
            chunks_used=len(chunks),
            pages=pages,
            timing=timing,
        )
