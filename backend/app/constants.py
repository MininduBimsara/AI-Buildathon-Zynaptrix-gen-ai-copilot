"""
constants.py — Large text blobs and static configurations for the Assistant.
"""

SYSTEM_ONBOARDING_CONTEXT = """
SYSTEM: Industrial Copilot Hub — Zynaptrix Platform

MODULES:
1. DASHBOARD: Real-time sensor telemetry charts. Select a machine from the top-bar MachineSelector to filter charts.
2. INCIDENT REGISTRY: Lists active anomaly alerts detected by the AI engine. Click an alert to open the Diagnostic Copilot chat.
3. DIAGNOSTIC COPILOT CHAT: AI-guided repair wizard for anomaly incidents. Provides step-by-step repair procedures.
4. CENTRAL ASSISTANT (this bot): A general knowledge assistant. Select a machine manual from the dropdown in the chat header to query RAG manuals.
5. MANUAL INGESTION (Knowledge Ingestion Tab): Upload PDF manuals for machines.
6. SIMULATOR: Start/Stop real-time telemetry simulation per machine.

GEN AI FRAMEWORK ARCHITECTURE:
This system represents a state-of-the-art "Gen AI Framework" for industrial environments.
- Multi-Agent Orchestration: Powered by LangGraph, multiple AI agents (Sensor Analysis, Diagnostic, Strategy, Critic, RAG) collaborate asynchronously to analyze anomalies and recommend procedures.
- Retrieval-Augmented Generation (RAG): Uses pgvector and OpenAI embeddings to semantically search technical manuals isolated per machine.
- Interactive Wizard Engine: Automatically converts dense technical manuals into sequential, conversational step-by-step UI actions.
- Adaptive Learning (Interaction Memory): As users confirm or reject maintenance steps, the framework stores the successful workflows back into the vector DB for future automated resolution.

HOW TO INGEST A MANUAL:
Step 1: Navigate to the Knowledge Ingestion tab from the left navigation.
Step 2: Enter the Manual ID — this must exactly match the manual_id you used when registering the machine in the database.
Step 3: Upload the PDF file using the file picker.
Step 4: Click the "Initial Ingestion" button to start processing. The system will chunk, embed, and store the manual in the vector database.
Step 5: After ingestion completes, go to the machine registration section and link the manual_id to the machine.

HOW TO REGISTER A MACHINE & ADD SENSORS:
Step 1: Go to the machine management section (Machine Registry).
Step 2: Enter the Machine ID (e.g. PUMP-001), Machine Name, Location, and Manual ID.
Step 3: Click "Add Sensor" to configure individual sensors. For each sensor, enter the Sensor Name, Sensor ID, and upload its manufacturer Datasheet (PDF).
Step 4: Save the machine.
What happens next? The system's Gen AI Framework automatically parses the uploaded sensor datasheets to extract operational parameters, generates synthentic telemetry data datasets, and trains a dedicated Anomaly Detection Machine Learning model strictly for that machine in the background.

HOW TO START THE SIMULATOR:
Step 1: Select a machine from the top-bar MachineSelector.
Step 2: Click the "Start" button in the top-right of the dashboard header.
Step 3: The simulator will begin streaming live telemetry data to the charts.

HOW TO USE THE ASSISTANT WITH A MACHINE MANUAL:
Step 1: Open the Central Assistant by clicking the blue chat button (bottom-right).
Step 2: Use the "RAG: [Machine] Manual" dropdown in the chat header to select a machine.
Step 3: Ask any technical question. The assistant will query that machine's specific manual and answer with expert guidance.

SECURITY NOTES:
- All API endpoints require proper session context.
- Machine manuals are isolated per machine_id in the vector database.
- Chat history is stored in the assistant_sessions and assistant_messages tables.
- Anomaly data is stored in anomaly_records table.
"""

SYSTEM_GUIDE_STEPS = {
    "ingest": [
        {"id": "ingest_1", "title": "Navigate to Knowledge Ingestion Tab", "detail": "Go to the Ingestion page from the navigation sidebar."},
        {"id": "ingest_2", "title": "Enter the Manual ID", "detail": "Add the same manual_id you entered when registering the machine. This links the manual to the machine."},
        {"id": "ingest_3", "title": "Upload the PDF file", "detail": "Click the file picker and select the machine manual PDF from your local filesystem."},
        {"id": "ingest_4", "title": "Click the 'Initial Ingestion' button", "detail": "This starts the pipeline: the system will chunk, embed, and store the manual in the vector database (pgvector)."},
        {"id": "ingest_5", "title": "Wait for completion", "detail": "The ingestion log will show progress. Once done, the manual is searchable via RAG."},
    ],
    "register": [
        {"id": "reg_1", "title": "Open Machine Management", "detail": "Navigate to the machine registration section in the admin panel."},
        {"id": "reg_2", "title": "Enter Machine Details", "detail": "Fill in Machine ID (e.g. PUMP-001), Machine Name, Location, and the Manual ID that matches your ingested PDF."},
        {"id": "reg_3", "title": "Add Sensors & Datasheets", "detail": "Click 'Add Sensor'. Enter the Sensor ID and Name, then upload the manufacturer's PDF datasheet for that sensor."},
        {"id": "reg_4", "title": "Save & Train Model", "detail": "Click Save. The Gen AI Framework will securely parse the datasheets, generate a synthetic dataset, and automatically train an anomaly detection ML model for this machine."},
    ],
    "simulator": [
        {"id": "sim_1", "title": "Select a Machine", "detail": "Use the top-bar MachineSelector dropdown to choose which machine to simulate."},
        {"id": "sim_2", "title": "Click the Start Button", "detail": "Click the green 'Start' button in the dashboard header. The simulator begins streaming live telemetry."},
        {"id": "sim_3", "title": "Monitor Telemetry Charts", "detail": "Live sensor data will populate the charts in real-time. An anomaly alert will fire automatically if readings exceed thresholds."},
    ],
    "assistant": [
        {"id": "ast_1", "title": "Open the Central Assistant", "detail": "Click the blue chat bubble button in the bottom-right corner of the dashboard."},
        {"id": "ast_2", "title": "Select a Machine Manual (optional)", "detail": "Use the 'RAG: [Machine] Manual' dropdown in the chat header to target a specific machine's documentation."},
        {"id": "ast_3", "title": "Ask Your Question", "detail": "Type any technical question. If a machine is selected, the answer comes from that machine's manual via RAG. Otherwise it uses general AI knowledge."},
    ],
    "framework": [
        {"id": "fw_1", "title": "Multi-Agent Orchestration Engine", "detail": "Under the hood, LangGraph coordinates multiple AI agents (Sensor Analysis, Strategy, RAG, Critic) to process anomalies entirely autonomously."},
        {"id": "fw_2", "title": "Adaptive Interaction Intelligence", "detail": "As you interact with step-cards (e.g., clicking 'I have done' or 'I am stuck'), the system learns successful resolution paths, saving them back into pgvector via the Interaction Memory."},
        {"id": "fw_3", "title": "Multi-Modal RAG Injection", "detail": "Technical queries dynamically extract relevant text and images from PDF manuals, synthesizing them into Conversational Wizard cards directly in the chat interface."},
    ]
}
