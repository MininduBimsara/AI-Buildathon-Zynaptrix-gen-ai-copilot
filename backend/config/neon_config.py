"""
neon_config.py — Neon PostgreSQL + pgvector connection configuration.

NOT WIRED INTO app/main.py — confirmed unreachable from the live FastAPI app
(2026-08, OpenAI->Qwen migration audit). Zero importers anywhere in this repo;
references EMBEDDING_DIM=768 and a `machine_documents` table that don't match
the live schema in unified_rag/db/models.py. Left as-is.
"""

import os
from dotenv import load_dotenv

load_dotenv()

NEON_DB_URL      = os.getenv("NEON_DB_URL", "postgresql://user:password@host/dbname")
EMBEDDING_DIM    = 768          # Embedding vector size (e.g. sentence-transformers)
VECTOR_TABLE     = "machine_documents"
