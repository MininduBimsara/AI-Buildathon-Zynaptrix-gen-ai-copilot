import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface TelemetryPoint {
  machineId: string;
  time: string;
  [key: string]: any;
}

// Structured Procedure Types
export interface ProcedureTask {
  id: string;
  text: string;
  critical?: boolean;
}

export interface ProcedureSubPhase {
  title: string;
  tasks: ProcedureTask[];
}

export interface ProcedurePhase {
  id: string;
  title: string;
  type?: string; // 'safety' | 'maintenance'
  subphases: ProcedureSubPhase[];
}

export interface ProcedureData {
  phases: ProcedurePhase[];
}

// === STEP-BY-STEP INTERACTIVE TYPES ===
export interface StepData {
  stepId: string;
  phaseType: string;
  phaseTitle: string;
  subphaseTitle: string;
  stepIndex: number;
  totalSteps: number;
  critical?: boolean;
  stepText?: string;
}

export interface StepResponse {
  status: 'done' | 'undone' | 'cant_do' | 'havent_done' | 'clarification_requested';
  comment?: string;
  advice?: string; // AI provided troubleshooting/branching advice
}

export interface FlatStep {
  id: string;
  text: string;
  phaseId: string;
  phaseTitle: string;
  phaseType: string;
  subphaseTitle: string;
  critical?: boolean;
}

export interface ActiveProcedure {
  flatSteps: FlatStep[];
  currentStepIndex: number;
  responses: Record<string, StepResponse>;
  images: string[];
  completed: boolean;
}

export interface ChatMessage {
  role: 'agent' | 'user';
  content: string;
  images?: string[];
  machineId?: string;
  quickActions?: string[];
  dbId?: number;
  hasSuggestion?: boolean;
  // Step-by-step interactive
  type?: 'text' | 'step' | 'phase_header' | 'procedure_complete' | 'step_clarification' | 'branching_advice' | 'wizard_step';
  stepData?: StepData;
  stepResponse?: StepResponse;
  isDocAlert?: boolean; // New flag for documentation-gap warnings
  metadata?: any;
  timestamp?: string;
  context_source?: string;
}

export interface AnomalyRecord {
  id: number;
  machine_id: string;
  timestamp: string;
  type: string;
  score: number;
  sensor_data: string;
  resolved?: boolean;
}

export interface AssistantSession {
  id: number;
  machine_id?: string;
  title: string;
  timestamp: string;
}

interface CopilotState {
  telemetry: TelemetryPoint[];
  chatHistory: Record<string, ChatMessage[]>;
  loadingHistory: Record<string, boolean>;
  anomalyHistory: AnomalyRecord[];
  activeAnomaly: AnomalyRecord | null;
  systemState: 'NORMAL' | 'ANOMALY';
  anomalyScore: number;
  activeAgents: string[];
  fleetHealth: Record<string, number>;
  activeProcedure: Record<string, ActiveProcedure>;
  assistantSessions: AssistantSession[];
  activeAssistantSessionId: number | null;
  isAssistantOpen: boolean;
  isAssistantSidebarOpen: boolean;
  assistantMachineId: string | null;
  // Feedback validation state
  resolveValidation: {
    isValidating: boolean;
    validationError: string | null;
    suggestions: string[];
  } | null;
  resolveThankYouMessage: string | null;
  // Export progress state
  exportProgress: {
    isExporting: boolean;
    progress: number;
    status: 'generating' | 'success' | 'error';
    errorMessage?: string;
  };
}

const initialState: CopilotState = {
  telemetry: [],
  chatHistory: {
    'general': [
      { role: 'agent', content: '🏭 Industrial Copilot initialized. Monitoring multi-machine factory floor.' }
    ]
  },
  loadingHistory: {},
  anomalyHistory: [],
  activeAnomaly: null,
  systemState: 'NORMAL',
  anomalyScore: 0.001,
  activeAgents: ['Sensor', 'Diagnostic', 'Strategy', 'Critic', 'RAG'],
  fleetHealth: {},
  activeProcedure: {},
  assistantSessions: [],
  activeAssistantSessionId: null,
  isAssistantOpen: false,
  isAssistantSidebarOpen: true,
  assistantMachineId: null,
  resolveValidation: null,
  resolveThankYouMessage: null,
  exportProgress: {
    isExporting: false,
    progress: 0,
    status: 'generating',
  },
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// --- Parse procedure JSON from agent response ---
function parseProcedureFromContent(content: string): ProcedureData | null {
  const startTag = '[PROCEDURE_START]';
  const endTag = '[PROCEDURE_END]';
  const startIdx = content.indexOf(startTag);
  const endIdx = content.indexOf(endTag);
  if (startIdx === -1 || endIdx === -1) return null;
  try {
    const jsonStr = content.substring(startIdx + startTag.length, endIdx).trim();
    return JSON.parse(jsonStr) as ProcedureData;
  } catch {
    return null;
  }
}

// --- Flatten a procedure into linear steps ---
function flattenProcedure(procedure: ProcedureData): FlatStep[] {
  const steps: FlatStep[] = [];
  procedure.phases.forEach(phase => {
    phase.subphases.forEach(sp => {
      sp.tasks.forEach(t => {
        steps.push({
          id: t.id,
          text: t.text,
          phaseId: phase.id,
          phaseTitle: phase.title,
          phaseType: phase.type || 'maintenance',
          subphaseTitle: sp.title,
          critical: t.critical,
        });
      });
    });
  });
  return steps;
}

export const fetchAnomalyHistory = createAsyncThunk(
  'copilot/fetchAnomalies',
  async (machineId: string) => {
    const response = await fetch(`${API_BASE}/api/machines/${machineId}/anomalies`);
    if (!response.ok) throw new Error('Failed to fetch anomaly history');
    return await response.json();
  }
);

export const fetchChatHistory = createAsyncThunk(
  'copilot/fetchChat',
  async (anomalyId: number) => {
    const response = await fetch(`${API_BASE}/api/chat-history/${anomalyId}`);
    if (!response.ok) throw new Error('Failed to fetch chat history');
    return { anomalyId, messages: await response.json() };
  }
);

export const resolveAnomaly = createAsyncThunk(
  'copilot/resolve',
  async (payload: { anomalyId: number; operator_fix: string }, { rejectWithValue }) => {
    const response = await fetch(`${API_BASE}/api/chat-history/${payload.anomalyId}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operator_fix: payload.operator_fix }),
    });
    const data = await response.json();
    
    // Handle validation failure (not a network error, but validation didn't pass)
    if (data.status === 'validation_failed') {
      return rejectWithValue({
        type: 'validation_failed',
        message: data.message,
        suggestions: data.suggestions || []
      });
    }
    
    if (!response.ok) throw new Error('Failed to resolve incident');
    return { anomalyId: payload.anomalyId, data };
  }
);

export const inquireCopilot = createAsyncThunk(
  'copilot/inquire',
  async (payload: { machine_id: string; query: string; machine_state: string; context_anomaly?: AnomalyRecord }, { dispatch }) => {
    const targetId = payload.context_anomaly?.id.toString();
    
    // Add user message first
    dispatch(addChatMessage({ 
        role: 'user', 
        content: payload.query, 
        machineId: payload.machine_id,
        targetId 
    }));
    
    // Add a placeholder agent message
    dispatch(addChatMessage({ 
        role: 'agent', 
        content: '⏳ Analyzing incident context...',
        targetId: targetId
    }));

    const body = {
        machine_id: payload.machine_id,
        user_query: payload.query,
        machine_state: payload.machine_state,
        anomaly_id: payload.context_anomaly?.id,
        recent_readings: payload.context_anomaly ? JSON.parse(payload.context_anomaly.sensor_data) : {},
        suspect_sensor: "Operator-Triggered Context"
    };

    const response = await fetch(`${API_BASE}/api/copilot/invoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) throw new Error('Copilot inquiry failed');
    const data = await response.json();

    // Check for backend-level errors
    if (data.status === 'error' || !data.graph_result) {
        throw new Error(data.message || 'Orchestration failure from AI engine');
    }

    return data;
  }
);

export const clarifyStep = createAsyncThunk(
  'copilot/clarifyStep',
  async (payload: { targetId: string; machineId: string; stepId: string; stepText: string }, { dispatch }) => {
    const { targetId, machineId, stepId, stepText } = payload;
    
    // 1. User message
    dispatch(addChatMessage({ 
        role: 'user', 
        content: `Show me how: "${stepText}"`,
        targetId 
    }));

    // 2. We don't need a placeholder. The fulfilled handler will just push to history instead of overwriting.

    const response = await fetch(`${API_BASE}/api/copilot/invoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
          machine_id: machineId,
          anomaly_id: targetId,
          user_query: `[CLARIFY_STEP] ${stepText}`,
          machine_state: "procedural_support"
      }),
    });

    // Return stepId + stepText so fulfilled handler can reconstruct stepData
    return { targetId, stepId, stepText, data: await response.json() };
  }
);

export const submitAdaptiveStepResponse = createAsyncThunk(
  'copilot/adaptiveResponse',
  async (payload: { targetId: string; machineId: string; stepId: string; status: string; comment?: string; stepText?: string }, { dispatch, getState }) => {
    const { targetId, machineId, stepId, status, comment, stepText } = payload;
    
    const isWizard = stepId.startsWith('wizard_');

    // CASE A: Done with NO comment -> FAST TRACK (Only for fixed procedures)
    if (!isWizard && status === 'done' && (!comment || !comment.trim())) {
        dispatch(respondToStep({ targetId, stepId, status: 'done' }));
        return { targetId, data: null, stepId, stepText: '' };
    }

    // CASE B: Evaluation or Branching Request
    const state = getState() as { copilot: CopilotState };
    const proc = state.copilot.activeProcedure[targetId];
    const step = proc?.flatSteps?.find(s => s.id === stepId);
    
    const labels: Record<string, string> = {
      'done': "✅ I've done this step",
      'undone': '❌ Not completed',
      'cant_do': "⚠️ I'm stuck / Unable to perform",
      'havent_done': "⏳ Not done yet",
    };
    let userMsg = labels[status] || status;
    if (comment && comment.trim()) userMsg += `\n💬 "${comment}"`;
    
    dispatch(addChatMessage({ 
        role: 'user', 
        content: userMsg,
        targetId 
    }));

    const feedbackText = comment?.trim() ? comment : (status === 'done' ? "I've done this step successfully." : 'Operator is stuck and needs alternative ways or deeper troubleshooting for this task.');
    const tag = status === 'done' ? '[EVALUATE_STEP]' : '[STUCK_STEP]';

    const response = await fetch(`${API_BASE}/api/copilot/invoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
          machine_id: machineId,
          anomaly_id: targetId,
          user_query: `${tag} Task: "${stepText || step?.text || 'Current maintenance task'}". User Feedback: "${feedbackText}".`,
          machine_state: isWizard ? "guided_repair_wizard" : "troubleshooting_branch"
      }),
    });

    const data = await response.json();
    const content = data.graph_result.final_execution_plan;

    // AUTO-ADVANCE on [STEP_COMPLETE] for legacy flows
    if (!isWizard && content.includes('[STEP_COMPLETE]')) {
        dispatch(respondToStep({ targetId, stepId, status: 'done', comment }));
    }

    // Return the intelligent context so fulfilled handler can hydrate stepData directly from the single orchestrated response
    return { targetId, stepId, stepText: step?.text || '', data };
  }
);

export const sendStepMessage = createAsyncThunk(
  'copilot/sendStepMessage',
  async (
    payload: { targetId: string; machineId: string; stepId: string; stepText: string; message: string },
    { dispatch, getState }
  ) => {
    const { targetId, machineId, stepId, stepText, message } = payload;

    // 1. Immediately show the user's message in chat
    dispatch(addChatMessage({ role: 'user', content: message, targetId }));

    // 2. Show a thinking indicator
    dispatch(addChatMessage({
      role: 'agent',
      content: '⚙️ Processing your message...',
      targetId,
      type: 'step_clarification',
      stepData: { stepId, stepText, phaseType: '', phaseTitle: '', subphaseTitle: '', stepIndex: 0, totalSteps: 0 }
    }));

    // 3. Classify intent via fast gpt-4o-mini call
    let intent = 'FREE_CHAT';
    try {
      const classRes = await fetch(`${API_BASE}/api/copilot/classify-intent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_message: message, step_text: stepText, machine_id: machineId }),
      });
      const classData = await classRes.json();
      intent = classData.intent || 'FREE_CHAT';
    } catch {
      intent = 'FREE_CHAT';
    }

    // 4. Route to the right action based on intent
    // Unified Conversational Routing
    const wizardQuery = intent === 'FREE_CHAT' 
        ? message 
        : `[CONVERSATIONAL_WIZARD] ${message} (Context: ${intent} for task "${stepText}")`;

    const res = await fetch(`${API_BASE}/api/copilot/invoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        machine_id: machineId,
        anomaly_id: targetId,
        user_query: wizardQuery,
        machine_state: intent === 'FREE_CHAT' ? 'general_inquiry' : 'procedural_support'
      }),
    });
    
    const data = await res.json();
    const finalContent = data.graph_result.final_execution_plan;

    // Auto-advance logic if AI confirms completion in wizard mode
    if (finalContent.includes('[STEP_COMPLETE]')) {
        dispatch(respondToStep({ targetId, stepId, status: 'done', comment: message }));
    }

    return { targetId, stepId, stepText, data, intent };
  }
);

export const fetchAssistantSessions = createAsyncThunk(
  'copilot/fetchAssistantSessions',
  async () => {
    const response = await fetch(`${API_BASE}/api/assistant/sessions`);
    if (!response.ok) throw new Error('Failed to fetch sessions');
    return await response.json();
  }
);

export const deleteAssistantSession = createAsyncThunk(
  'copilot/deleteAssistantSession',
  async (sessionId: number) => {
    const response = await fetch(`${API_BASE}/api/assistant/sessions/${sessionId}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Failed to delete session');
    return sessionId;
  }
);

export const fetchAssistantHistory = createAsyncThunk(
  'copilot/fetchAssistantHistory',
  async (sessionId: number) => {
    const response = await fetch(`${API_BASE}/api/assistant/sessions/${sessionId}/history`);
    if (!response.ok) throw new Error('Failed to fetch session history');
    return { sessionId, messages: await response.json() };
  }
);

export const exportAssistantSession = createAsyncThunk(
  'copilot/exportAssistantSession',
  async (sessionId: number, { rejectWithValue, dispatch }) => {
    try {
      console.log(`[Report Export] Starting professional report generation for session ${sessionId}`);
      
      // Update UI: Start exporting
      dispatch(copilotSlice.actions.setExportProgress({ progress: 5, status: 'generating' }));
      
      // Fetch structured report data from backend
      const response = await fetch(`${API_BASE}/api/assistant/sessions/${sessionId}/report`);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[Report Export] Backend error: ${response.status} - ${errorText}`);
        throw new Error(`Failed to generate report: ${response.status} ${response.statusText}`);
      }
      
      dispatch(copilotSlice.actions.setExportProgress({ progress: 30, status: 'generating' }));
      
      const reportData = await response.json();
      console.log(`[Report Export] Received report data:`, reportData);
      
      // Check if we have solution steps
      if (!reportData.solutionSteps || reportData.solutionSteps.length === 0) {
        throw new Error('No solution steps available. Please ensure the conversation contains diagnostic information.');
      }

      dispatch(copilotSlice.actions.setExportProgress({ progress: 40, status: 'generating' }));

      // Import professional report service
      console.log('[Report Export] Loading professional report service...');
      const { generateDiagnosticReport } = await import('../../professionalReportService');

      // Generate professional diagnostic report PDF
      console.log('[Report Export] Generating professional diagnostic report...');
      await generateDiagnosticReport(reportData, (progress) => {
        console.log(`[Report Export] Progress: ${progress}%`);
        dispatch(copilotSlice.actions.setExportProgress({ progress, status: 'generating' }));
      });

      console.log('[Report Export] Professional report generated successfully!');
      dispatch(copilotSlice.actions.setExportProgress({ progress: 100, status: 'success' }));
      
      // Auto-close modal after 2 seconds
      setTimeout(() => {
        dispatch(copilotSlice.actions.resetExportProgress());
      }, 2000);
      
      return { success: true, sessionId };
    } catch (error: any) {
      console.error('[Report Export] Error:', error);
      dispatch(copilotSlice.actions.setExportProgress({ 
        progress: 0, 
        status: 'error',
        errorMessage: error.message 
      }));
      return rejectWithValue(error.message || 'Export failed');
    }
  }
);

export const inquireAssistant = createAsyncThunk(
  'copilot/inquireAssistant',
  async ({ query, machineId, sessionId }: { query: string; machineId?: string; sessionId?: number }, { dispatch, getState }) => {
    const state = getState() as any;
    const finalMachineId = machineId || state.copilot.assistantMachineId;
    
    // Add user message to UI immediately under the 'assistant' key
    dispatch(addChatMessage({ 
        role: 'user', 
        content: query, 
        targetId: 'assistant' 
    }));

    const response = await fetch(`${API_BASE}/api/assistant`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, machine_id: finalMachineId, session_id: sessionId }),
    });
    if (!response.ok) throw new Error('Assistant failed to respond');
    const data = await response.json();
    if (data.session_id && sessionId !== data.session_id) {
        dispatch(fetchAssistantSessions());
    }
    return data;
  }
);

const copilotSlice = createSlice({
  name: 'copilot',
  initialState,
  reducers: {
    addTelemetry(state, action: PayloadAction<TelemetryPoint>) {
      state.telemetry.push(action.payload);
      if (state.telemetry.length > 20) state.telemetry.shift();
    },
    addChatMessage(state, action: PayloadAction<ChatMessage & { targetId?: string }>) {
      const { targetId, ...msg } = action.payload;
      const key = targetId || 'general';
      if (!state.chatHistory[key]) {
        state.chatHistory[key] = [];
      }
      state.chatHistory[key].push(msg);
    },
    addAnomalyToHistory(state, action: PayloadAction<AnomalyRecord>) {
      state.anomalyHistory.unshift(action.payload);
      if (state.anomalyHistory.length > 50) state.anomalyHistory.pop();
    },
    setActiveAnomaly(state, action: PayloadAction<AnomalyRecord | null>) {
      state.activeAnomaly = action.payload;
    },
    setSystemState(state, action: PayloadAction<'NORMAL' | 'ANOMALY'>) {
      state.systemState = action.payload;
    },
    setAnomalyScore: (state, action: PayloadAction<number>) => {
      state.anomalyScore = action.payload;
    },
    setAssistantOpen: (state, action: PayloadAction<boolean>) => {
      state.isAssistantOpen = action.payload;
      if (action.payload) {
        state.activeAnomaly = null; // Close active anomaly if opening assistant
      }
    },
    setAssistantSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.isAssistantSidebarOpen = action.payload;
    },
    setActiveAssistantSessionId: (state, action: PayloadAction<number | null>) => {
      state.activeAssistantSessionId = action.payload;
      if (action.payload === null) {
          state.chatHistory['assistant'] = [];
      }
    },
    setActiveAgents(state, action: PayloadAction<string[]>) {
      state.activeAgents = action.payload;
    },
    setAssistantMachineId: (state, action: PayloadAction<string | null>) => {
      state.assistantMachineId = action.payload;
    },
    clearResolveState: (state) => {
      state.resolveValidation = null;
      state.resolveThankYouMessage = null;
    },
    
    // Export progress actions
    setExportProgress: (state, action: PayloadAction<{ progress: number; status: 'generating' | 'success' | 'error'; errorMessage?: string }>) => {
      state.exportProgress = {
        isExporting: true,
        ...action.payload,
      };
    },
    resetExportProgress: (state) => {
      state.exportProgress = {
        isExporting: false,
        progress: 0,
        status: 'generating',
      };
    },

    // === NEW: Step-by-step procedure interaction ===
    respondToStep(state, action: PayloadAction<{ targetId: string; stepId: string; status: string; comment?: string }>) {
      const { targetId, stepId, status, comment } = action.payload;
      const proc = state.activeProcedure[targetId];
      if (!proc) return;

      const history = state.chatHistory[targetId];
      if (!history) return;

      // Record response in procedure state
      proc.responses[stepId] = { status: status as StepResponse['status'], comment };

      // Mark related messages as responded
      for (let i = history.length - 1; i >= 0; i--) {
        const msg = history[i];
        if ((msg.type === 'step' || msg.type === 'step_clarification' || msg.type === 'branching_advice') && msg.stepData?.stepId === stepId) {
          msg.stepResponse = { status: status as StepResponse['status'], comment };
        }
      }

      // Add user response message to chat
      const labels: Record<string, string> = {
        'done': '✅ Completed',
        'undone': '❌ Not completed',
        'cant_do': "⚠️ Unable to perform",
        'havent_done': "⏳ Not done yet",
      };
      let userMsg = labels[status] || status;
      if (comment && comment.trim()) userMsg += `\n💬 "${comment}"`;
      history.push({ role: 'user', content: userMsg });

      // Advance to next step ONLY FOR 'done' status
      if (status === 'done') {
        proc.currentStepIndex++;

        // Procedural Completion (Fixed procedures only)
        if (proc.flatSteps.length > 0 && proc.currentStepIndex >= proc.flatSteps.length) {
          proc.completed = true;
          history.push({
            role: 'agent',
            content: '🎉 **All procedure steps completed!**\n\nGreat work! Click **Complete Task** above to finalize and archive this incident with your notes.',
            type: 'procedure_complete',
          });
        }

        // Procedural Advance (Fixed procedures only)
        if (proc.flatSteps.length > 0 && proc.currentStepIndex < proc.flatSteps.length) {
          const nextStep = proc.flatSteps[proc.currentStepIndex];
          const prevStep = proc.flatSteps[proc.currentStepIndex - 1];

          // Phase transition header
          if (nextStep && prevStep && nextStep.phaseId !== prevStep.phaseId) {
            history.push({
              role: 'agent',
              content: `✅ **${prevStep.phaseTitle}** — Complete!\n\nProceeding to **${nextStep.phaseTitle}**...`,
              type: 'phase_header',
            });
          }

          // Add next step message
          history.push({
            role: 'agent',
            content: nextStep.text,
            type: 'step',
            stepData: {
              stepId: nextStep.id,
              phaseType: nextStep.phaseType,
              phaseTitle: nextStep.phaseTitle,
              subphaseTitle: nextStep.subphaseTitle,
              stepIndex: proc.currentStepIndex,
              totalSteps: proc.flatSteps.length,
              critical: nextStep.critical,
            },
            images: proc.images,
          });
        }
      }
    },

    forceAdvanceStep(state, action: PayloadAction<{ targetId: string }>) {
      const { targetId } = action.payload;
      const proc = state.activeProcedure[targetId];
      if (!proc) return;
      const history = state.chatHistory[targetId];
      if (!history) return;

      proc.currentStepIndex++;

      if (proc.flatSteps.length > 0 && proc.currentStepIndex >= proc.flatSteps.length) {
        proc.completed = true;
        history.push({
          role: 'agent',
          content: '🎉 **All procedure steps completed!**\n\nGreat work! Click **Complete Task** above to finalize and archive this incident with your notes.',
          type: 'procedure_complete',
        });
      }

      if (proc.flatSteps.length > 0 && proc.currentStepIndex < proc.flatSteps.length) {
        const nextStep = proc.flatSteps[proc.currentStepIndex];
        const prevStep = proc.flatSteps[proc.currentStepIndex - 1];

        if (nextStep && prevStep && nextStep.phaseId !== prevStep.phaseId) {
          history.push({
            role: 'agent',
            content: `✅ **${prevStep.phaseTitle}** — Complete!\n\nProceeding to **${nextStep.phaseTitle}**...`,
            type: 'phase_header',
          });
        }

        history.push({
          role: 'agent',
          content: nextStep.text,
          type: 'step',
          stepData: {
            stepId: nextStep.id,
            phaseType: nextStep.phaseType,
            phaseTitle: nextStep.phaseTitle,
            subphaseTitle: nextStep.subphaseTitle,
            stepIndex: proc.currentStepIndex,
            totalSteps: proc.flatSteps.length,
            critical: nextStep.critical,
          },
          images: proc.images,
        });
      }
    },
  },
  extraReducers: (builder) => {
    builder.addCase(fetchAnomalyHistory.fulfilled, (state, action) => {
      state.anomalyHistory = action.payload;
    });
    
    builder.addCase(fetchChatHistory.pending, (state, action) => {
        state.loadingHistory[action.meta.arg.toString()] = true;
    });
    builder.addCase(fetchChatHistory.fulfilled, (state, action) => {
        const { anomalyId, messages } = action.payload;
        if (messages.length === 0) {
            // Empty — the auto-diagnosis useEffect in page.tsx will populate
            state.chatHistory[anomalyId.toString()] = [];
        } else {
            // Hydrate messages from DB
            const hydrated = [];
            for (const m of messages) {
                const contentText = m.content || '';
                
                let msgObj: any = {
                    role: m.role,
                    content: contentText,
                    images: m.images || [],
                    dbId: m.db_id,
                    metadata: m.metadata
                };

                // Re-hydrate Wizard/Action statuses if it's an agent message containing markers
                if (m.role === 'agent') {
                    const phaseMatch = contentText.match(/\[PHASE:\s*(.*?)\]/i);
                    const isWizard = contentText.includes('[CONVERSATIONAL_WIZARD]');
                    
                    if (phaseMatch || isWizard) {
                        // Ensure activeProcedure exists for this anomaly to allow button handlers to work
                        if (!state.activeProcedure[anomalyId.toString()]) {
                            state.activeProcedure[anomalyId.toString()] = {
                                flatSteps: [], currentStepIndex: 0, responses: {}, images: [], completed: false
                            };
                        }
                        
                        const finalContent = contentText.replace(/\[STEP_COMPLETE\]/g, '✅').replace(/\[STEP_NEED_HELP\]/g, '⚠️').replace(/\[CONVERSATIONAL_WIZARD\]/gi, '');
                        
                        if (phaseMatch) {
                            hydrated.push({
                                role: 'agent',
                                type: 'phase_header',
                                content: `🛡️ **${phaseMatch[1]}**`
                            });
                            msgObj.content = finalContent.replace(/\[PHASE:.*?\]/gi, '').trim();
                            msgObj.type = 'wizard_step';
                            msgObj.stepData = {
                                stepId: 'wizard_flow_' + m.db_id,
                                stepText: msgObj.content,
                                phaseType: 'diagnostic', phaseTitle: 'Guided Repair', subphaseTitle: phaseMatch[1], stepIndex: 1, totalSteps: 1
                            };
                        } else {
                            msgObj.content = finalContent.trim();
                            msgObj.type = 'wizard_step';
                            msgObj.stepData = {
                                stepId: 'wizard_flow_' + m.db_id,
                                stepText: msgObj.content,
                                phaseType: 'diagnostic', phaseTitle: 'Guided Repair', subphaseTitle: 'AI Navigator', stepIndex: 1, totalSteps: 1
                            };
                        }
                    }
                } else if (m.role === 'user' && m.metadata && m.metadata.action === 'step_response') {
                    // This was a button response, map it to stepResponse mapping natively through the frontend states
                    // (To avoid double printing, we just keep the user message as normal, 
                    // but we look at previous agent messages and mark them as responded)
                    for (let i = hydrated.length - 1; i >= 0; i--) {
                        if (hydrated[i].role === 'agent' && hydrated[i].type === 'wizard_step' && !hydrated[i].stepResponse) {
                            hydrated[i].stepResponse = { status: m.metadata.status, comment: m.metadata.comment };
                            break;
                        }
                    }
                }
                
                hydrated.push(msgObj);
            }
            state.chatHistory[anomalyId.toString()] = hydrated;
        }
        state.loadingHistory[anomalyId.toString()] = false;
    });
    builder.addCase(fetchChatHistory.rejected, (state, action) => {
        const anomalyId = action.meta.arg.toString();
        state.loadingHistory[anomalyId] = false;
        if (!state.chatHistory[anomalyId]) {
            state.chatHistory[anomalyId] = [
                { role: 'agent', content: '⚠️ Could not restore past history. Diagnostic bridge active.' }
            ];
        }
    });
    
    // Resolution Logic
    builder.addCase(resolveAnomaly.pending, (state) => {
        state.resolveValidation = {
            isValidating: true,
            validationError: null,
            suggestions: []
        };
        state.resolveThankYouMessage = null;
    });
    
    builder.addCase(resolveAnomaly.fulfilled, (state, action) => {
        const { anomalyId, data } = action.payload;
        
        // Store thank you message if available
        if (data.thank_you_message) {
            state.resolveThankYouMessage = data.thank_you_message;
        }
        
        // Clear validation state
        state.resolveValidation = null;
        
        if (state.activeAnomaly?.id === anomalyId) {
          state.activeAnomaly = null;
        }

        // Update the item in anomalyHistory to reflect resolved status immediately
        const idx = state.anomalyHistory.findIndex(a => a.id === anomalyId);
        if (idx !== -1) {
            state.anomalyHistory[idx].resolved = true;
        }

        // Cleanup local transient state
        delete state.chatHistory[anomalyId.toString()];
        delete state.activeProcedure[anomalyId.toString()];
    });
    
    builder.addCase(resolveAnomaly.rejected, (state, action) => {
        const payload = action.payload as { type: string; message: string; suggestions: string[] } | undefined;
        
        if (payload?.type === 'validation_failed') {
            // Feedback validation failed - show error and suggestions
            state.resolveValidation = {
                isValidating: false,
                validationError: payload.message,
                suggestions: payload.suggestions || []
            };
        } else {
            // Other error
            state.resolveValidation = {
                isValidating: false,
                validationError: 'Failed to archive incident. Please try again.',
                suggestions: []
            };
        }
    });

    // --- Assistant Session Handlers ---
    builder.addCase(fetchAssistantSessions.fulfilled, (state, action) => {
        state.assistantSessions = action.payload;
    });
    
    builder.addCase(deleteAssistantSession.fulfilled, (state, action) => {
        state.assistantSessions = state.assistantSessions.filter(s => s.id !== action.payload);
        if (state.activeAssistantSessionId === action.payload) {
            state.activeAssistantSessionId = null;
            state.chatHistory['assistant'] = [];
        }
    });

    builder.addCase(fetchAssistantHistory.pending, (state, action) => {
        state.loadingHistory['assistant'] = true;
    });

    builder.addCase(fetchAssistantHistory.fulfilled, (state, action) => {
        const { messages, sessionId } = action.payload;
        state.chatHistory['assistant'] = messages;
        state.activeAssistantSessionId = sessionId;
        state.loadingHistory['assistant'] = false;
    });

    builder.addCase(fetchAssistantHistory.rejected, (state) => {
        state.loadingHistory['assistant'] = false;
    });

    builder.addCase(inquireAssistant.pending, (state) => {
        state.loadingHistory['assistant'] = true;
    });

    builder.addCase(inquireAssistant.fulfilled, (state, action) => {
        const { session_id, content, timestamp, type, images } = action.payload;
        state.loadingHistory['assistant'] = false;

        if (!state.activeAssistantSessionId || state.activeAssistantSessionId !== session_id) {
            state.activeAssistantSessionId = session_id;
        }
        if (!state.chatHistory['assistant']) {
            state.chatHistory['assistant'] = [];
        }

        // Intelligent Type Detection: If it has phases or numbered steps, treat as wizard
        const isProcedural = content.includes('[PHASE:') || /^\d+\./m.test(content) || (images && images.length > 0);
        const finalType = isProcedural ? 'wizard_step' : 'text';

        state.chatHistory['assistant'].push({
            role: 'agent',
            content,
            timestamp,
            type: finalType,
            images: images || [],
            // Ensure stepData is present for wizard types to enable optimized rendering
            stepData: isProcedural ? {
                stepId: 'assistant_step',
                stepText: content.trim(),
                phaseType: 'diagnostic', phaseTitle: 'Expert Assistance', subphaseTitle: 'Assistant Wisdom', stepIndex: 0, totalSteps: 0
            } as any : undefined
        });
    });

    builder.addCase(inquireAssistant.rejected, (state) => {
        state.loadingHistory['assistant'] = false;
    });

    // === MAIN HANDLER: Copilot Response ===
    builder.addCase(inquireCopilot.fulfilled, (state, action) => {
      const targetId = action.meta.arg.context_anomaly?.id.toString() || 'general';
      const history = state.chatHistory[targetId];
      if (!history || history.length === 0) return;

      const lastIdx = history.length - 1;
      const result = action.payload.graph_result;
      
      // DEFENSIBLY GUARD: Prevent TypeError if result is unexpectedly nil
      if (!result || !result.final_execution_plan) {
          history[lastIdx] = {
              role: 'agent',
              content: '⚠️ The AI engine encountered an orchestration failure. Please try clarifying your request.',
          };
          return;
      }

      const content = result.final_execution_plan;
      const procedure = parseProcedureFromContent(content);
      
      const finalContent = content.replace(/\[STEP_COMPLETE\]/g, '✅').replace(/\[STEP_NEED_HELP\]/g, '⚠️').replace(/\[CONVERSATIONAL_WIZARD\]/gi, '');

      // Intelligent Phase Detection: [PHASE: Safety & Lockout]
      const phaseMatch = content.match(/\[PHASE:\s*(.*?)\]/i);
      if (phaseMatch) {
          history[lastIdx] = {
              role: 'agent',
              type: 'phase_header',
              content: `🛡️ **${phaseMatch[1]}**`
          };
          history.push({
              role: 'agent',
              content: finalContent.replace(/\[PHASE:.*?\]/gi, '').trim(),
              images: result.retrieved_images || [],
              machineId: action.meta.arg.machine_id,
              dbId: action.payload.db_id,
              type: 'wizard_step',
              hasSuggestion: /\[SUGGESTION:[\s\S]*?Generate full step-by-step repair procedure[\s\S]*?\]/i.test(content),
              stepData: {
                  stepId: 'wizard_flow',
                  stepText: finalContent.replace(/\[PHASE:.*?\]/gi, '').trim(),
                  phaseType: 'diagnostic', phaseTitle: 'Guided Repair', subphaseTitle: phaseMatch[1], stepIndex: 1, totalSteps: 1
              } as any
          });
          return;
      }

      const isWizard = content.includes('[CONVERSATIONAL_WIZARD]') || action.meta.arg.query.includes('[CONVERSATIONAL_WIZARD]') || /^\d+\./m.test(content);

      history[lastIdx] = {
        role: 'agent',
        content: finalContent.trim(),
        images: result.retrieved_images || [],
        machineId: action.meta.arg.machine_id,
        dbId: action.payload.db_id,
        hasSuggestion: /\[SUGGESTION:[\s\S]*?Generate full step-by-step repair procedure[\s\S]*?\]/i.test(content),
        isDocAlert: content.includes('⚠️ Documentation Alert'),
        // Force wizard_step type if it's a guide or has numbered steps
        type: isWizard ? 'wizard_step' : 'text',
        stepData: isWizard ? {
            stepId: 'wizard_flow',
            stepText: finalContent.trim(),
            phaseType: 'diagnostic', phaseTitle: 'Guided Repair', subphaseTitle: 'Expert Guidance', stepIndex: 1, totalSteps: 1
        } as any : undefined
      };

      // If we've started a wizard session, initialize the step context
      if (content.includes('[CONVERSATIONAL_WIZARD]') || action.meta.arg.query.includes('[CONVERSATIONAL_WIZARD]')) {
          state.activeProcedure[targetId] = {
              flatSteps: [],
              currentStepIndex: 0,
              responses: {},
              images: result.retrieved_images || [],
              completed: false,
          };
      }
    });

     builder.addCase(inquireCopilot.rejected, (state, action) => {
        const targetId = action.meta.arg.context_anomaly?.id.toString() || 'general';
        const history = state.chatHistory[targetId];
        if (history && history.length > 0) {
          const lastIdx = history.length - 1;
          history[lastIdx].content = "⚠️ Error communicating with Copilot backend.";
        }
     });

     // Adaptive Guidance Handlers
     builder.addCase(clarifyStep.fulfilled, (state, action) => {
        const { targetId, stepId, stepText, data } = action.payload;
        const history = state.chatHistory[targetId];
        if (!history) return;
        const lastIdx = history.length - 1;
        const content = data.graph_result.final_execution_plan;
        const finalContent = content.replace(/\[STEP_COMPLETE\]/g, '✅').replace(/\[STEP_NEED_HELP\]/g, '⚠️').replace(/\[CONVERSATIONAL_WIZARD\]/gi, '');
        
        const phaseMatch = content.match(/\[PHASE:\s*(.*?)\]/i);
        if (phaseMatch) {
            history.push({
                role: 'agent',
                type: 'phase_header',
                content: `🚀 **${phaseMatch[1]}**`
            });
        }
        
        history.push({
            role: 'agent',
            content: finalContent.replace(/\[PHASE:.*?\]/gi, '').trim(),
            type: 'wizard_step',
            isDocAlert: content.includes('⚠️ Documentation Alert'),
            images: data.graph_result.retrieved_images || [],
            stepData: {
              stepId,
              stepText: stepText || '',
              phaseType: '', phaseTitle: '', subphaseTitle: 'AI Tutorial Detail', stepIndex: 0, totalSteps: 0
            } as any
        });
     });

     builder.addCase(submitAdaptiveStepResponse.fulfilled, (state, action) => {
        if (!action.payload) return;
        const { targetId, stepId, stepText, data } = action.payload;
        if (!data) return;
        const history = state.chatHistory[targetId];
        if (!history) return;
        const lastIdx = history.length - 1;
        const content = data.graph_result.final_execution_plan;
        const finalContent = content.replace(/\[STEP_COMPLETE\]/g, '✅').replace(/\[STEP_NEED_HELP\]/g, '⚠️');

        // Intelligent Phase Transition
        const phaseMatch = content.match(/\[PHASE:\s*(.*?)\]/i);
        if (phaseMatch) {
            history.push({
                role: 'agent',
                type: 'phase_header',
                content: `🚀 **${phaseMatch[1]}**`
            });
            history.push({
                role: 'agent',
                content: finalContent.replace(/\[PHASE:.*?\]/gi, '').replace(/\[PROCEDURE_FINISH\]/g, '🏁').trim(),
                type: 'wizard_step',
                images: data.graph_result.retrieved_images,
                stepData: {
                    stepId: stepId || 'wizard_step',
                    stepText: finalContent.replace(/\[PHASE:.*?\]/gi, '').trim(),
                    phaseType: 'active_repair', phaseTitle: 'Guided Step', subphaseTitle: phaseMatch[1], stepIndex: 0, totalSteps: 0
                } as any,
            });
            return;
        }

        history.push({
            role: 'agent',
            content: finalContent.replace(/\[PROCEDURE_FINISH\]/g, '🏁'),
            type: finalContent.includes('[PROCEDURE_FINISH]') ? 'procedure_complete' : 'wizard_step',
            isDocAlert: content.includes('⚠️ Documentation Alert'),
            images: data.graph_result.retrieved_images || [],
            stepData: {
              stepId,
              stepText: finalContent.replace(/\[PROCEDURE_FINISH\]/g, '🏁').trim(),
              phaseType: '', phaseTitle: '', subphaseTitle: 'Agent Guidance', stepIndex: 0, totalSteps: 0
            } as any
        });
     });

      // === INTELLIGENT CHAT HANDLER ===
      builder.addCase(sendStepMessage.pending, (state, action) => {
          const { targetId, message } = action.meta.arg;
          const history = state.chatHistory[targetId];
          if (!history) return;

          // Immediately hide buttons on the previous step card
          for (let i = history.length - 1; i >= 0; i--) {
              const msg = history[i];
              if (msg.role === 'agent' && (msg.type === 'wizard_step' || msg.type === 'step')) {
                  // Determine status from the message content
                  let status: any = 'done';
                  if (message.toLowerCase().includes("stuck") || message.toLowerCase().includes("alternative")) {
                      status = 'cant_do';
                  }
                  msg.stepResponse = { status, comment: message };
                  break;
              }
          }
      });

     builder.addCase(sendStepMessage.fulfilled, (state, action) => {
        if (!action.payload) return;
        const { targetId, stepId, stepText, data, intent } = action.payload;
        if (!data) return;
        const history = state.chatHistory[targetId];
        if (!history) return;

        // Find and replace the placeholder "Processing..." message
        const lastIdx = history.length - 1;
        const content = data.graph_result?.final_execution_plan;
        if (!content) return;

        // Route rendering based on classified intent
        const typeMap: Record<string, string> = {
          'NEED_DETAIL': 'step_clarification',
          'NEED_HELP': 'branching_advice',
          'CONFIRM_DONE': 'step_clarification', // Warm confirmation
          'FREE_CHAT': 'text',
        };
        const msgType = typeMap[intent] || 'text';
        const finalContent = content.replace(/\[STEP_COMPLETE\]/g, '✅').replace(/\[STEP_NEED_HELP\]/g, '⚠️');

        // Intelligent Phase Transition
        const phaseMatch = content.match(/\[PHASE:\s*(.*?)\]/i);
        if (phaseMatch) {
            history[lastIdx] = {
                role: 'agent',
                type: 'phase_header',
                content: `🚀 **${phaseMatch[1]}**`
            };
            history.push({
                role: 'agent',
                content: finalContent.replace(/\[PHASE:.*?\]/gi, '').replace(/\[PROCEDURE_FINISH\]/g, '🏁').trim(),
                type: 'wizard_step',
                images: data.graph_result.retrieved_images,
                stepData: {
                    stepId: stepId || 'wizard_step',
                    stepText: finalContent.replace(/\[PHASE:.*?\]/gi, '').trim(),
                    phaseType: 'active_repair', phaseTitle: 'Guided Step', subphaseTitle: phaseMatch[1], stepIndex: 0, totalSteps: 0
                } as any,
            });
            return;
        }

        history[lastIdx] = {
            role: 'agent',
            content: finalContent.replace(/\[PROCEDURE_FINISH\]/g, '🏁'),
            type: finalContent.includes('[PROCEDURE_FINISH]') ? 'procedure_complete' : 'wizard_step',
            images: data.graph_result.retrieved_images,
            // Ensure stepData is always attached in Wizard/Clarification mode
            stepData: {
              stepId: stepId || 'wizard_step',
              stepText: finalContent.replace(/\[PROCEDURE_FINISH\]/g, '🏁').trim(),
              phaseType: 'active_repair', phaseTitle: 'Guided Step', subphaseTitle: 'Conversational Flow', stepIndex: 0, totalSteps: 0
            } as any,
        };
     });
  },
});

export const { 
  addTelemetry, 
  addChatMessage, 
  addAnomalyToHistory,
  setActiveAnomaly,
  setSystemState, 
  setAnomalyScore,
  setAssistantOpen,
  setAssistantSidebarOpen,
  setActiveAssistantSessionId,
  setAssistantMachineId,
  setActiveAgents,
  respondToStep,
  forceAdvanceStep,
  clearResolveState,
  setExportProgress,
  resetExportProgress
} = copilotSlice.actions;

export default copilotSlice.reducer;
