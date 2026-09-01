"""
reembed_manual_chunks.py — Re-embed existing ManualChunk/InteractionMemory rows
after switching embedding providers (OpenAI text-embedding-3-small -> Qwen
text-embedding-v4).

Required after the OpenAI -> Qwen migration: embeddings from different
providers/dimensions are not comparable, and cosine_distance() errors if a
query mixes dimensions with stored rows. This script only touches Postgres —
no PDFs or vision calls needed, since ManualChunk.content already holds the
final extracted/captioned text from the original ingestion.

Usage (from backend/):
    python scripts/reembed_manual_chunks.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from unified_rag.db.database import SessionLocal
from unified_rag.db.models import ManualChunk, InteractionMemory
from unified_rag.embeddings.embedder import embedder

BATCH_SIZE = 20


def reembed(db, rows, text_attr: str, label: str):
    total = len(rows)
    if total == 0:
        print(f"   ∟ No {label} rows found — skipping.")
        return
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        for row in batch:
            text = getattr(row, text_attr)
            if text:
                row.embedding = embedder.embed_text(text)
        db.commit()
        print(f"   ∟ {label}: committed {min(i + BATCH_SIZE, total)}/{total}")


def main():
    db = SessionLocal()
    try:
        print("Re-embedding ManualChunk rows...")
        chunks = db.query(ManualChunk).all()
        reembed(db, chunks, "content", "ManualChunk")

        print("Re-embedding InteractionMemory rows...")
        memories = db.query(InteractionMemory).all()
        reembed(db, memories, "summary", "InteractionMemory")

        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
