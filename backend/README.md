# 🏭 Industrial AI Copilot — Backend

> **Predictive Maintenance System for Industrial Machines**
> Real-time anomaly detection, multimodal RAG retrieval, and Qwen-powered multi-agent diagnostic orchestration.

> For the full system architecture, end-to-end flow diagrams, and subsystem breakdown, see the [root README](../README.md). This file covers backend-specific setup only.

---

## 🛠️ Tech Stack

- **Core**: FastAPI (Python 3.10+)
- **Orchestration**: LangGraph (six-node multi-agent pipeline)
- **AI**: Qwen — `qwen-max` (reasoning), `qwen-flash` (lightweight tasks), `qwen-vl-max` (vision/diagram captioning), `text-embedding-v4` (2048-dim embeddings) — all routed through `unified_rag/ai_client.py`
- **Database**: Neon PostgreSQL (with `pgvector` for text/image/memory embeddings)
- **Blob Storage**: Cloudinary (manuals, extracted figures, trained models/scalers, evaluation plots) — no local disk persistence
- **Unified RAG**: YOLOv8-DocLayNet layout detection, Mobile SAM figure decomposition, Qwen-VL diagram captioning
- **Analytics**: TensorFlow/Keras (Dense + LSTM Autoencoders for anomaly detection)
- **Time-Series**: InfluxDB (real-time sensor telemetry)

---

## 📖 Running the Backend

```bash
# 1. Create and activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# fill in AI_API_KEY (Qwen/DashScope), DATABASE_URL (Neon), CLOUDINARY_*, INFLUX_*

# 4. Run the API
uvicorn app.main:app --reload
```
*Access API Docs at: http://localhost:8000/docs*

### Bootstrapping data
Everything below is cloud-native — there is no local `generate_dataset.py` → `.keras` file workflow to run by hand. Instead:
- **Register a machine** (`POST /api/machines` or the frontend `/machines` page) — this triggers synthetic dataset generation, normalization, dual-autoencoder training, and evaluation automatically, uploading every artifact to Cloudinary and registering it in Neon.
- **Ingest a manual** (`POST /ingest-manual` or the frontend `/ingestion` page) — runs the full multimodal ingestion pipeline (layout detection → figure captioning → chunking → embedding) and uploads the source PDF + figures to Cloudinary.
- **Simulate telemetry** — `POST /api/simulator/start?machine_id=...` to exercise anomaly detection and the agent pipeline without real hardware.

---

## 📂 Project Structure

```bash
backend/
├── agents/             # LangGraph six-node pipeline + validation/automation agents
├── app/                # FastAPI routes (Telemetry, Copilot, Assistant, Machines, Evaluation, Simulator)
├── config/             # Sensor schema & machine-state settings
├── models/             # Autoencoder logic, model_registry (cloud asset loading), evaluation
├── preprocessing/      # Data normalization (cloud round-trip via services/pipeline_storage.py)
├── services/           # Cloudinary/pipeline_storage, alerts, incident summarizer, sensor config loader
├── simulator/          # Synthetic sensor data generator
├── scripts/            # Dataset generation & training orchestration, eval/benchmark tooling
└── unified_rag/        # Core Multimodal RAG Engine (parser, embedder, retriever, ai_client)
```

---

## 🚀 Unified RAG Capabilities
The `unified_rag` module supports true multimodal retrieval:
1. **Layout Parsing**: Automatically separates text, tables, and figures from technical PDFs.
2. **Visual Intelligence**: Technical diagrams are interpreted by Qwen-VL to generate searchable captions, with composite drawings decomposed into individually retrievable sub-components via Mobile SAM.
3. **Multimodal Search**: Users can search for specific spare parts or machine settings, and the system retrieves the exact diagram from the manual — text and images share one embedding space, no separate visual index required.

---
*Developed for Zynaptrix Industrial Research*
