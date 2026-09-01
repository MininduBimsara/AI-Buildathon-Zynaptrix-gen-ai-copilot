"""
knowledge_agent.py — RAG-based Knowledge Agent for the Industrial AI Copilot.

NOT WIRED INTO app/main.py — confirmed unreachable from the live FastAPI app
(2026-08, OpenAI->Qwen migration audit). Imports vector_pipeline.embedding_generator
and database.neon_vector_store, neither of which exist anywhere in this repo, so
this module raises ModuleNotFoundError on import today. Left as-is (not migrated
to Qwen, not deleted) so the original author can reconcile or revive it later.
Only imported by the also-dead agents/orchestrator_agent.py.

Responsibilities:
  1. Accept a natural-language query (e.g. "motor overload at 7A, vibration high")
  2. Embed the query using EmbeddingGenerator
  3. Retrieve top-k most relevant maintenance doc chunks from Neon pgvector
  4. Format and return a structured maintenance recommendation

Usage:
    from agents.knowledge_agent import KnowledgeAgent
    agent = KnowledgeAgent()
    result = agent.query("Motor current is 7.2A and vibration is high")
    print(result["answer"])
    print(result["sources"])
"""

import logging
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeAgent
# ─────────────────────────────────────────────────────────────────────────────
class KnowledgeAgent:
    """
    Retrieval-Augmented Generation (RAG) agent for industrial maintenance advice.

    Pipeline:
        query text
            → embed (EmbeddingGenerator — OpenAI text-embedding-3-small)
            → similarity search (NeonVectorStore)
            → format context
            → optional LLM synthesis (OpenAI GPT-4o-mini)
            → return structured result

    The agent works in two modes:
      - retrieval_only: Returns raw retrieved chunks (no LLM, always works)
      - llm_synthesis:  Passes retrieved chunks + query to an LLM for a
                        polished natural-language answer (requires API key)
    """

    def __init__(self, top_k: int = 5, use_llm: bool = True):
        """
        Args:
            top_k:    Number of document chunks to retrieve per query.
            use_llm:  Whether to synthesize a natural-language answer via LLM.
                      Falls back to retrieval_only if no API key is found.
        """
        self.top_k   = top_k
        self.use_llm = use_llm and self._has_llm_key()

        # Lazy-loaded to avoid import errors if packages not installed
        self._embedder = None
        self._store    = None

    # ── Setup helpers ────────────────────────────────────────────────────────
    def _has_llm_key(self) -> bool:
        return bool(OPENAI_API_KEY and
                    not OPENAI_API_KEY.startswith("sk-proj-YOUR"))

    def _get_embedder(self):
        if self._embedder is None:
            from vector_pipeline.embedding_generator import EmbeddingGenerator
            embedder = EmbeddingGenerator()
            self._embedder = embedder
        return self._embedder

    def _get_store(self):
        if self._store is None:
            from database.neon_vector_store import NeonVectorStore
            store = NeonVectorStore()
            store.connect()
            self._store = store
        return self._store

    # ── Main query method ─────────────────────────────────────────────────────
    def query(
        self,
        query_text: str,
        fault_type: Optional[str] = None,
        sensor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the knowledge base and return a maintenance recommendation.

        Args:
            query_text:  Natural-language description of the anomaly or question.
            fault_type:  Optional filter — restricts search to this fault type tag.
            sensor:      Optional hint about which sensor triggered the anomaly.
                         Injected into the query for better retrieval.

        Returns:
            Dict with keys:
              - query:       Original query text
              - answer:      Maintenance recommendation (str)
              - sources:     List of retrieved doc chunks [{title, content, score}]
              - fault_type:  Detected or provided fault type
              - mode:        'llm_synthesis' or 'retrieval_only'
        """
        # Enrich query with sensor context if provided
        effective_query = query_text
        if sensor:
            effective_query = f"{sensor} sensor — {query_text}"

        logger.info(f"[KnowledgeAgent] Query: {effective_query!r}  "
                    f"(fault_type={fault_type or 'any'})")

        # Step 1: Embed the query
        embedder = self._get_embedder()
        query_embedding = embedder.embed_query(effective_query)

        # Step 2: Retrieve from Neon
        store = self._get_store()
        results = store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            fault_type_filter=fault_type,
        )

        if not results:
            return {
                "query": query_text,
                "answer": (
                    "⚠️ No relevant maintenance documentation found in the knowledge "
                    "base. Please run the vector uploader to index maintenance docs, "
                    "or consult the machine manual directly."
                ),
                "sources": [],
                "fault_type": fault_type,
                "mode": "retrieval_only",
            }

        # Step 3: Format sources
        sources = [
            {
                "title":      r["title"],
                "content":    r["content"],
                "score":      float(round(r["similarity_score"], 4)),
                "doc_id":     r["doc_id"],
                "fault_type": r["fault_type"],
            }
            for r in results
        ]

        # Step 4: Synthesize answer
        if self.use_llm:
            answer = self._synthesize_with_llm(query_text, sources, sensor)
            mode = "llm_synthesis"
        else:
            answer = self._format_retrieval_only(sources)
            mode = "retrieval_only"

        return {
            "query":      query_text,
            "answer":     answer,
            "sources":    sources,
            "fault_type": fault_type or (results[0]["fault_type"] if results else None),
            "mode":       mode,
        }

    # ── LLM synthesis ─────────────────────────────────────────────────────────
    def _synthesize_with_llm(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        sensor: Optional[str] = None,
    ) -> str:
        """Call OpenAI GPT-4o-mini to synthesize a structured maintenance answer."""

        context = "\n\n---\n\n".join(
            f"[Source {i+1}: {s['title']} (relevance: {s['score']:.2f})]\n{s['content']}"
            for i, s in enumerate(sources)
        )

        prompt = (
            "You are an expert industrial maintenance engineer specialising in "
            "automated tea bag packing machines (BOSCH / Lipton-style packaging lines).\n\n"
            "An AI anomaly detection system has flagged a problem. "
            "Using the retrieved maintenance documentation below, provide a "
            "concise, actionable maintenance recommendation.\n\n"
            "Format your response as:\n"
            "**DIAGNOSIS**: [Likely root cause in 1-2 sentences]\n"
            "**IMMEDIATE ACTIONS**: [Numbered list of urgent steps]\n"
            "**REPAIR STEPS**: [Numbered list of fix procedures]\n"
            "**PREVENTION**: [1-2 preventive measures]\n\n"
            f"Anomaly / Query: {query}\n"
            + (f"Suspect Sensor: {sensor}\n" if sensor else "")
            + f"\nRetrieved Documentation:\n{context}"
        )

        if OPENAI_API_KEY:
            return self._call_openai(prompt)

        return self._format_retrieval_only(sources)


    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[KnowledgeAgent] OpenAI call failed: {e}")
            return self._format_retrieval_only([])

    # ── Retrieval-only fallback ───────────────────────────────────────────────
    @staticmethod
    def _format_retrieval_only(sources: List[Dict[str, Any]]) -> str:
        """Format retrieved sources as a readable text when no LLM is available."""
        if not sources:
            return "No relevant documentation found."
        lines = ["📋 **Retrieved Maintenance Documentation:**\n"]
        for i, s in enumerate(sources, 1):
            lines.append(f"**{i}. {s['title']}** (relevance: {s['score']:.2f})")
            lines.append(s["content"][:500] + ("…" if len(s["content"]) > 500 else ""))
            lines.append("")
        return "\n".join(lines)

    # ── Convenience helpers ───────────────────────────────────────────────────
    def query_from_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method: build a query directly from an alert_service alert dict.

        Args:
            alert: Dict from AlertService with keys:
                   machine_state, suspect_sensor, anomaly_score, message

        Returns:
            Knowledge agent result dict.
        """
        machine_state  = alert.get("machine_state", "unknown")
        suspect_sensor = alert.get("suspect_sensor")
        score          = alert.get("anomaly_score", 0)

        query_text = (
            f"Anomaly detected — machine state: {machine_state}, "
            f"anomaly score: {score:.2f}"
        )
        if suspect_sensor:
            query_text += f", highest anomaly on {suspect_sensor} sensor"

        return self.query(
            query_text=query_text,
            fault_type=machine_state if machine_state != "unknown" else None,
            sensor=suspect_sensor,
        )

    def close(self):
        """Clean up database connection."""
        if self._store is not None:
            self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()