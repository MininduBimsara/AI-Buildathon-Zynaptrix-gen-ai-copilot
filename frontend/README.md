# 📊 Industrial AI Copilot — Dashboard

> **Real-time Predictive Maintenance & Multimodal Chat Interface**  
> Built with Next.js 14, Tailwind CSS, and Recharts.

---

## 🚀 Overview
The Industrial AI Dashboard provides a unified interface for monitoring machine health and interacting with technical documentation via a multimodal RAG system.

### Key Features
- **Live Telemetry**: Real-time visualization of temperature, current, vibration, and speed.
- **Anomaly Highlighting**: Instant visual feedback when the backend autoencoder detects deviations.
- **Multimodal RAG Chat**: 
  - Interact with AGENTIC copilot for maintenance strategies.
  - **Visual Retrieval**: The system can retrieve and display exact diagrams/figures from ingested PDF manuals.
- **Manual Management**: Direct PDF upload for automated parsing, captioning, and indexing.

---

## 🛠️ Development

### 1. Install Dependencies
```bash
# From the industrial_copilot/frontend folder
npm install
```

### 2. Configure Environment
Create a `.env.local` file in the root of the frontend folder:
```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### 3. Start Development Server
```bash
npm run dev
```
*Access the Dashboard at: http://localhost:3000*

---

## 📂 Structure
- `src/app/`: Core pages — dashboard (`/`), `/machines`, `/ingestion`.
- `src/components/`: Reusable UI components (Assistant Sidebar, Machine Selector, Procedure Guide, etc.).
- `src/store/`: Redux Toolkit slices for copilot, ingestion, machine, and simulator state.
- `public/`: Static assets and icons.

---
*Powered by Zynaptrix Advanced Agentic Coding*

