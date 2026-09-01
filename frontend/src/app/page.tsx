"use client"
import React, { useState, useEffect, useRef, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  ChevronRight,
  Database, 
  Gauge, 
  MessageSquare, 
  Play, 
  Send, 
  Server, 
  ShieldAlert,
  ShieldCheck, 
  Square, 
  Thermometer, 
  Vibrate as VibrateIcon,
  ArrowRight,
  Maximize2,
  Minimize2,
  Wrench,
  MessageCircle,
  Zap,
  Droplets,
  Wind,
  Ruler,
  Weight,
  RotateCw,
  Radio,
  Sun,
  FlaskConical,
  Magnet,
  Cpu,
  TrendingUp,
  Hash,
  Move,
  Bot,
  User,
  History,
  X,
  Download
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store/store';
import { SensorMeta } from '../store/slices/machineSlice';
import { 
  addTelemetry, 
  addChatMessage, 
  addAnomalyToHistory,
  setActiveAnomaly,
  setSystemState, 
  setAnomalyScore,
  fetchAnomalyHistory,
  fetchChatHistory,
  resolveAnomaly,
  inquireCopilot,
  setAssistantOpen,
  inquireAssistant,
  respondToStep,
  clarifyStep,
  submitAdaptiveStepResponse,
  forceAdvanceStep,
  sendStepMessage,
  clearResolveState,
  exportAssistantSession,
  resetExportProgress
} from '../store/slices/copilotSlice';
import { fetchMachines, setCurrentMachineId, fetchMachineConfig } from '../store/slices/machineSlice';
import { fetchSimulatorStatus, startSimulator, stopSimulator } from '../store/slices/simulatorSlice';
import MachineSelector from '../components/MachineSelector';
import AssistantSidebar from '../components/AssistantSidebar';
import AssistantMachineSelector from '../components/AssistantMachineSelector';
import ExportProgressModal from '../components/ExportProgressModal';
import { fetchAssistantSessions } from '../store/slices/copilotSlice';

// ── Icon mapping: icon_type (from OpenAI) → Lucide component + accent color ──
const ICON_MAP: Record<string, { icon: React.ElementType; color: string }> = {
  temperature:   { icon: Thermometer,  color: 'rose' },
  current:       { icon: Zap,          color: 'amber' },
  vibration:     { icon: VibrateIcon,  color: 'purple' },
  pressure:      { icon: Gauge,        color: 'blue' },
  speed:         { icon: RotateCw,     color: 'cyan' },
  flow:          { icon: Droplets,     color: 'teal' },
  voltage:       { icon: Zap,          color: 'yellow' },
  humidity:      { icon: Wind,         color: 'sky' },
  distance:      { icon: Ruler,        color: 'indigo' },
  load:          { icon: Weight,       color: 'orange' },
  torque:        { icon: Magnet,       color: 'pink' },
  position:      { icon: Move,         color: 'violet' },
  power:         { icon: TrendingUp,   color: 'emerald' },
  frequency:     { icon: Radio,        color: 'lime' },
  light:         { icon: Sun,          color: 'yellow' },
  gas:           { icon: FlaskConical, color: 'green' },
  force:         { icon: Activity,     color: 'red' },
  conductivity:  { icon: Cpu,          color: 'fuchsia' },
  ph:            { icon: FlaskConical, color: 'teal' },
  weight:        { icon: Weight,       color: 'stone' },
  angle:         { icon: Move,         color: 'cyan' },
  counter:       { icon: Hash,         color: 'slate' },
  generic:       { icon: Server,       color: 'blue' },
};

function getSensorDisplay(iconType: string): { icon: React.ElementType; color: string } {
  return ICON_MAP[iconType] || ICON_MAP['generic'];
}

export default function IndustrialCopilotDashboard() {
  const dispatch = useDispatch<AppDispatch>();
  
  // Redux-driven State
  const { 
      telemetry, 
      chatHistory, 
      anomalyHistory, 
      activeAnomaly, 
      systemState, 
      anomalyScore, 
      loadingHistory, 
      activeProcedure, 
      isAssistantOpen,
      activeAssistantSessionId,
      assistantMachineId,
      resolveValidation,
      resolveThankYouMessage,
      exportProgress
  } = useSelector((state: RootState) => state.copilot);
  const { machines, currentMachineId, machineConfigs } = useSelector((state: RootState) => state.machines);
  const { activeSimulators } = useSelector((state: RootState) => state.simulator);
  
  // Local transient UI states
  const [query, setQuery] = useState('');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isChatMaximized, setIsChatMaximized] = useState(false);
  const [isResolveModalOpen, setIsResolveModalOpen] = useState(false);
  const [operatorFix, setOperatorFix] = useState('');
  const [activeRegistryTab, setActiveRegistryTab] = useState<'active' | 'archive'>('active');
  const [stepComments, setStepComments] = useState<Record<string, string>>({});
  const [stepChatInputs, setStepChatInputs] = useState<Record<string, string>>({});
  const [isMobileHistoryOpen, setIsMobileHistoryOpen] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Filter telemetry points for the current machine - Memoized to prevent unstable references
  const filteredTelemetry = useMemo(() => 
    telemetry.filter(t => t.machineId === currentMachineId),
    [telemetry, currentMachineId]
  );
  
  const latestReading = useMemo(() => 
    filteredTelemetry[filteredTelemetry.length - 1] || {},
    [filteredTelemetry]
  );

  // Build display list: prefer machineConfigs (has icon_type), fall back to live keys - Memoized for stability
  const displayList = useMemo(() => {
    const metaList: SensorMeta[] = machineConfigs[currentMachineId] || [];
    const liveSensorKeys = Object.keys(latestReading).filter(
      k => !['machineId', 'time', 'machine_id', 'state', 'health_score'].includes(k)
    );

    let list: SensorMeta[] = metaList.length > 0
      ? metaList
      : liveSensorKeys.map(k => ({ sensor_id: k, sensor_name: k, icon_type: 'generic', unit: 'units' }));

    if (list.length === 0) {
      list = [
        { sensor_id: 'temperature', sensor_name: 'Temperature', icon_type: 'temperature', unit: '°C' },
        { sensor_id: 'motor_current', sensor_name: 'Motor Current', icon_type: 'current', unit: 'A' },
        { sensor_id: 'vibration', sensor_name: 'Vibration', icon_type: 'vibration', unit: 'mm/s' },
      ];
    }
    return list;
  }, [machineConfigs, currentMachineId, latestReading]);
  
  const isSimulating = activeSimulators.includes(currentMachineId);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    dispatch(fetchMachines());
    dispatch(fetchSimulatorStatus());
    dispatch(fetchAssistantSessions());
  }, [dispatch]);

  useEffect(() => {
    if (currentMachineId) {
      dispatch(fetchAnomalyHistory(currentMachineId));
      dispatch(fetchMachineConfig(currentMachineId));
    }
  }, [currentMachineId, dispatch]);

  // Load chat history and trigger initial diagnosis when an anomaly is selected
  useEffect(() => {
    if (activeAnomaly) {
      const anomalyIdStr = activeAnomaly.id.toString();
      dispatch(fetchChatHistory(activeAnomaly.id)).then((action: any) => {
        // action.payload is { anomalyId, messages }
        const history = action.payload?.messages;
        if (!history || history.length === 0) {
          dispatch(inquireCopilot({
            machine_id: activeAnomaly.machine_id,
            query: "Provide a quick diagnostic summary for this alert.",
            machine_state: activeAnomaly.type,
            context_anomaly: activeAnomaly
          }));
        }
      });
    }
  }, [activeAnomaly?.id, dispatch]);

  // WebSocket Telemetry Stream
  useEffect(() => {
    let ws: WebSocket;
    const connectWS = () => {
      const base = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const wsUrl = base.replace('http', 'ws');
      ws = new WebSocket(`${wsUrl}/ws/telemetry`);
      
      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type === "telemetry") {
            const data = parsed.data;
            dispatch(addTelemetry({
              ...data,
              machineId: data.machine_id || 'PUMP-001',
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            }));
          } else if (parsed.type === "anomaly_alert") {
            const result = parsed.data;
            dispatch(setSystemState('ANOMALY'));
            dispatch(setAnomalyScore(result.anomaly_score || 0.99));
            dispatch(addAnomalyToHistory({
                id: result.id,
                machine_id: result.machine_id,
                timestamp: new Date().toLocaleTimeString(),
                type: result.machine_state,
                score: Math.round(result.anomaly_score * 100),
                sensor_data: JSON.stringify(result.recent_readings || {}),
                resolved: false
            }));
          }
        } catch (err) { console.error(err); }
      };
      ws.onclose = () => setTimeout(connectWS, 3000);
    };
    connectWS();
    return () => ws?.close();
  }, [dispatch]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isChatOpen]);

  const handleManualInquiry = (customQuery?: string) => {
    const finalQuery = customQuery || query;
    if (!finalQuery.trim()) return;
    if (!isAssistantOpen && activeAnomaly?.resolved) return;

    setQuery('');
    
    if (isAssistantOpen) {
      dispatch(inquireAssistant({ 
          query: finalQuery, 
          machineId: assistantMachineId || undefined,
          sessionId: activeAssistantSessionId || undefined
      }));
    } else {
      let processedQuery = finalQuery;
      // Intelligent Switch: If manual repair is requested, use the Conversational Wizard mode
      if (processedQuery.includes("Generate full step-by-step repair procedure")) {
          processedQuery = "[CONVERSATIONAL_WIZARD] Start the guided repair procedure from the beginning.";
      }

      dispatch(inquireCopilot({
        machine_id: currentMachineId,
        query: processedQuery,
        machine_state: activeAnomaly ? activeAnomaly.type : "manual_inquiry_general",
        context_anomaly: activeAnomaly || undefined
      }));
    }
  };

  const handleResolve = async () => {
    if (!activeAnomaly || !operatorFix) return;
    const result = await dispatch(resolveAnomaly({ 
        anomalyId: activeAnomaly.id, 
        operator_fix: operatorFix 
    }));
    
    // Only close modal and reset if successful (not validation failure)
    if (resolveAnomaly.fulfilled.match(result)) {
      // Don't close modal yet - show thank you message
      // The modal will show the thank you state
    }
    // If rejected due to validation, modal stays open with error shown
  };
  
  const handleCloseResolveModal = () => {
    setIsResolveModalOpen(false);
    setOperatorFix('');
    dispatch(clearResolveState());
  };
  
  const handleOpenAssistantFromResolve = () => {
    handleCloseResolveModal();
    dispatch(setAssistantOpen(true));
  };

  const handleStepResponse = (stepId: string, status: string, stepText?: string) => {
    const comment = stepComments[stepId];
    // Assistant mode — send a follow-up message as the user
    if (isAssistantOpen) {
      let followUp = comment?.trim() || '';
      if (!followUp) {
        if (status === 'done') followUp = `I have completed: "${stepText || stepId}". What is next?`;
        else if (status === 'cant_do') followUp = `I am stuck on: "${stepText || stepId}". Can you show me how to do this in more detail?`;
        else followUp = `I need help with: "${stepText || stepId}"."`;
      }
      dispatch(inquireAssistant({
        query: followUp,
        machineId: assistantMachineId || undefined,
        sessionId: activeAssistantSessionId || undefined
      }));
      setStepComments(prev => ({...prev, [stepId]: ''}));
      return;
    }
    if (!activeAnomaly) return;
    const targetId = activeAnomaly.id.toString();
    const isWizard = stepId.startsWith('wizard_') || stepId.startsWith('rag_');
    if (isWizard) {
      let predefinedMessage = comment?.trim() || "";
      if (!predefinedMessage) {
         if (status === 'done') predefinedMessage = "I've done this step successfully.";
         else if (status === 'cant_do') predefinedMessage = "I'm stuck with this step, please show the alternative way.";
      }
      dispatch(sendStepMessage({ targetId, machineId: activeAnomaly.machine_id, stepId, stepText: stepText || 'Current maintenance task', message: predefinedMessage }));
    } else {
      if (status === 'done' && (!comment || !comment.trim())) {
        dispatch(respondToStep({ targetId, stepId, status: status as any }));
      } else {
        dispatch(submitAdaptiveStepResponse({ targetId, machineId: activeAnomaly.machine_id, stepId, status, comment, stepText }));
      }
    }
    setStepComments(prev => ({...prev, [stepId]: ''}));
  };

  const handleSendStepMessage = (stepId: string, stepText: string) => {
    const msg = stepChatInputs[stepId]?.trim();
    if (!msg) return;
    if (isAssistantOpen) {
      dispatch(inquireAssistant({ query: msg, machineId: assistantMachineId || undefined, sessionId: activeAssistantSessionId || undefined }));
      setStepChatInputs(prev => ({...prev, [stepId]: ''}));
      return;
    }
    if (!activeAnomaly) return;
    dispatch(sendStepMessage({ targetId: activeAnomaly.id.toString(), machineId: activeAnomaly.machine_id, stepId, stepText, message: msg }));
    setStepChatInputs(prev => ({...prev, [stepId]: ''}));
  };

  const handleClarifyStep = (stepId: string, stepText: string) => {
    if (isAssistantOpen) {
      dispatch(inquireAssistant({ query: `Explain in more detail: "${stepText}"`, machineId: assistantMachineId || undefined, sessionId: activeAssistantSessionId || undefined }));
      return;
    }
    if (!activeAnomaly) return;
    const isWizard = stepId.startsWith('wizard_') || stepId.startsWith('rag_');
    if (isWizard) {
      dispatch(sendStepMessage({ targetId: activeAnomaly.id.toString(), machineId: activeAnomaly.machine_id, stepId, stepText, message: "Please tell me exactly how to do this step in simple terms." }));
    } else {
      dispatch(clarifyStep({ targetId: activeAnomaly.id.toString(), machineId: activeAnomaly.machine_id, stepText, stepId }));
    }
  };

  // Helper: render step text with inline images + Markdown
  const renderStepContent = (text: string, images?: string[]) => {
    let processedText = text;
    if (images && images.length > 0) {
      images.forEach((url, idx) => {
        const tag = new RegExp(`\\[IMAGE[_\s-]?${idx}\\]`, 'gi');
        processedText = processedText.replace(tag, `![Technical Diagram](${url})`);
      });
    }

    return (
      <div className="markdown-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ img: ({node, ...props}) => (<img {...props} className="max-w-full md:max-w-2xl max-h-80 object-contain rounded-xl border border-slate-600 my-3 shadow-lg" />) }}>
          {processedText}
        </ReactMarkdown>
      </div>
    );
  };

  const toggleSimulation = async () => {
    if (isSimulating) await dispatch(stopSimulator(currentMachineId));
    else await dispatch(startSimulator(currentMachineId));
    dispatch(fetchSimulatorStatus());
  };


  return (
    <div className="min-h-screen bg-transparent text-slate-200 p-4 md:p-8 font-sans selection:bg-blue-500/30 overflow-x-hidden flex flex-col">
      
      {/* Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8 border-b border-slate-800 pb-6 bg-[#020617] shadow-2xl shadow-black/80 rounded-2xl p-6">
        <div>
          <h1 className="text-2xl md:text-3xl font-black bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent flex items-center gap-3">
            <Server className="text-blue-500" size={24} />
            Industrial Copilot Hub
          </h1>
          <div className="flex items-center gap-2 mt-1">
             <div className={`h-2 w-2 rounded-full ${isSimulating ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'}`}></div>
             <p className="text-slate-400 text-[10px] font-bold tracking-widest uppercase opacity-70">
               {isSimulating ? 'Live Telemetry Active' : 'Simulator Standby'}
             </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4 md:gap-6 w-full md:w-auto">
          <MachineSelector />
          <button
              onClick={toggleSimulation}
              className={`flex-1 md:flex-initial px-5 py-2.5 rounded-xl font-black text-[10px] uppercase tracking-widest transition-all flex items-center justify-center gap-2 shadow-2xl ${isSimulating ? 'bg-red-500/10 text-red-500 border border-red-500/30' : 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/30'}`}
            >
              {isSimulating ? <Square size={12} fill="currentColor" /> : <Play size={12} fill="currentColor" />}
              {isSimulating ? "Stop" : "Start"}
          </button>
        </div>
      </header>

      {/* Main Grid: Left (Sensor Lines) | Right (Detected Anomalies) */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 flex-1 overflow-hidden">
        
        {/* Left Aspect: Sensor Telemetry (2/3 width) */}
        <div className="xl:col-span-2 flex flex-col gap-6 overflow-hidden min-h-0">
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
             {(() => {
                const chartColors = ['blue', 'emerald', 'amber', 'purple', 'rose', 'cyan', 'teal', 'orange'];

                return displayList.map((sensor, i) => {
                  const { icon: Icon, color } = getSensorDisplay(sensor.icon_type);
                  const rawVal = latestReading[sensor.sensor_id];
                  const val = rawVal !== undefined ? Number(rawVal).toFixed(2) : '—';
                  const accentColor = chartColors[i % chartColors.length];
                  return (
                   <div key={sensor.sensor_id} className={`bg-[#0a0f1e] border-slate-700/50 rounded-3xl p-6 shadow-2xl relative overflow-hidden group transition-all hover:border-slate-700`}>
                      <div className={`absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity text-${accentColor}-500`}>
                         <Icon size={60} />
                      </div>
                      <div className="flex items-center gap-2 mb-2">
                        <div className={`p-1.5 rounded-lg bg-${accentColor}-500/10`}>
                          <Icon size={14} className={`text-${accentColor}-400`} />
                        </div>
                        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
                          {sensor.sensor_name.replace(/_/g, ' ')}
                        </h3>
                      </div>
                      <p className="text-3xl font-black text-white">
                        {val} <span className="text-xs text-slate-500 font-light">{sensor.unit}</span>
                      </p>
                   </div>
                  );
                });
             })()}
          </div>

          <div className="bg-[#0a0f1e] border-slate-700/50 rounded-3xl p-4 md:p-8 shadow-2xl relative overflow-hidden h-[400px] md:h-[600px] flex flex-col">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
               <h2 className="text-lg md:text-xl font-black flex items-center gap-3 text-white">
                 <Activity className="text-blue-500" /> Real-Time Sensor Stream
               </h2>
                <div className="flex flex-wrap items-center gap-3 md:gap-4 text-[9px] md:text-[10px] font-bold text-slate-500">
                   {(() => {
                     const colors = ['bg-blue-500', 'bg-emerald-500', 'bg-amber-500', 'bg-purple-500', 'bg-rose-500', 'bg-cyan-500', 'bg-teal-500', 'bg-orange-500'];
                     return displayList.map((sensor, i) => (
                       <span key={sensor.sensor_id} className="flex items-center gap-2">
                         <div className={`h-1.5 w-4 ${colors[i % colors.length]} rounded`}></div>
                         {sensor.sensor_name.substring(0, 5).toUpperCase()}
                       </span>
                     ));
                   })()}
                </div>
            </div>
            <div className="flex-1 w-full min-h-[300px]">
              {isMounted && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={filteredTelemetry}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="time" stroke="#475569" tick={{fill: '#475569', fontSize: 10, fontWeight: 700}} hide />
                    <YAxis stroke="#475569" tick={{fill: '#475569', fontSize: 10, fontWeight: 700}} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                    />
                    {(() => {
                       const colors = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7', '#f43f5e', '#06b6d4', '#14b8a6', '#f97316'];
                       return displayList.map((sensor, i) => (
                         <Line 
                            key={sensor.sensor_id} 
                            type="monotone" 
                            dataKey={sensor.sensor_id} 
                            stroke={colors[i % colors.length]} 
                            strokeWidth={3} 
                            dot={false} 
                            isAnimationActive={false} 
                         />
                       ));
                    })()}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        {/* Right Aspect: Anomaly Archive (1/3 width) */}
        <div className="flex flex-col gap-6 overflow-hidden min-h-0">
          <div className="bg-[#0a0f1e] border-slate-700/50 rounded-3xl p-6 shadow-2xl flex flex-col h-full overflow-hidden">
            <h2 className="text-lg font-black text-white mb-6 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Database className="text-blue-400" size={20} /> Incident Registry
              </span>
              <div className={`px-3 py-1 rounded-full text-[10px] font-black border ${systemState === 'NORMAL' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-red-500/10 border-red-500/30 text-red-500'}`}>
                {systemState}
              </div>
            </h2>

            <div className="mb-6">
              <div className="flex p-1 bg-slate-800/50 rounded-xl border border-slate-700/50">
                <button 
                  onClick={() => setActiveRegistryTab('active')}
                  className={`flex-1 py-2 px-4 rounded-lg text-xs font-bold transition-all ${
                    activeRegistryTab === 'active' 
                    ? 'bg-blue-600 text-white shadow-lg' 
                    : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Active Alerts
                </button>
                <button 
                  onClick={() => setActiveRegistryTab('archive')}
                  className={`flex-1 py-2 px-4 rounded-lg text-xs font-bold transition-all ${
                    activeRegistryTab === 'archive' 
                    ? 'bg-emerald-600 text-white shadow-lg' 
                    : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Archive
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-slate-800">
               {anomalyHistory
                .filter(a => a.machine_id === currentMachineId)
                .filter(a => activeRegistryTab === 'active' ? !a.resolved : a.resolved).length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center opacity-20 py-20 text-center">
                    <CheckCircle size={40} className="mb-4" />
                    <p className="text-xs font-bold uppercase tracking-widest">No Anomalies Logged</p>
                 </div>
               ) : (
                 anomalyHistory
                  .filter(a => a.machine_id === currentMachineId)
                  .filter(a => activeRegistryTab === 'active' ? !a.resolved : a.resolved)
                  .map((item) => (
                   <button
                    key={item.id}
                    onClick={() => {
                        dispatch(setActiveAnomaly(item));
                        dispatch(setAssistantOpen(false));
                        setIsChatOpen(true);
                    }}
                    className={`w-full p-4 rounded-2xl border transition-all text-left relative group ${activeAnomaly?.id === item.id ? 'bg-blue-600/10 border-blue-500' : 'bg-slate-950/50 border-slate-800/50 hover:bg-slate-900'}`}
                   >
                     <div className="flex justify-between items-start mb-2">
                        <span className={`text-[9px] font-black px-2 py-0.5 rounded uppercase tracking-[0.1em] ${item.type?.includes('fault') ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'}`}>
                           {item.type?.replace('machine_', '') ?? 'unknown'}
                        </span>
                        <div className="flex items-center gap-2">
                            {item.resolved && <CheckCircle size={10} className="text-emerald-500" />}
                            <span className="text-[9px] text-slate-500 font-mono italic">{item.timestamp}</span>
                        </div>
                     </div>
                     <p className="text-sm font-bold text-slate-200">Industrial Incident #{item.id}</p>
                     <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-tighter">Variance Score: {item.score ?? '—'}%</p>
                     <div className="absolute right-4 bottom-4 opacity-0 group-hover:opacity-100 transition-all text-blue-400">
                        <ArrowRight size={16} />
                     </div>
                   </button>
                 ))
               )}
            </div>
          </div>
        </div>
      </div>

      {/* Floating Chat Trigger */}
      <button 
        onClick={() => {
            dispatch(setAssistantOpen(true));
            setIsChatOpen(true);
        }}
        className="fixed bottom-8 right-8 h-16 w-16 bg-blue-600 hover:bg-blue-500 text-white rounded-full shadow-2xl shadow-blue-500/40 flex items-center justify-center transition-all hover:scale-110 active:scale-95 group z-40"
      >
        <MessageSquare size={24} className="group-hover:rotate-12 transition-transform" />
        {!isAssistantOpen && activeAnomaly && (
            <div className="absolute -top-1 -right-1 h-5 w-5 bg-red-500 rounded-full border-2 border-slate-950 flex items-center justify-center text-[10px] font-black">1</div>
        )}
      </button>

      {/* 🚀 PREMIUM GALAXY CHAT MODAL */}
      {isChatOpen && (
        <div className={`fixed inset-0 bg-transparent backdrop-blur-[6px] z-50 flex items-center justify-center animate-in fade-in duration-500 ${isChatMaximized ? 'p-0' : 'p-4'}`}>
          <div className={`relative bg-[#020617] border border-blue-500/30 shadow-[0_0_120px_rgba(30,58,138,0.3)] flex flex-col overflow-hidden animate-in zoom-in-95 slide-in-from-bottom-10 duration-700 transition-all ${isChatMaximized ? 'w-full h-full rounded-none' : 'w-full md:max-w-4xl h-full md:h-[85vh] md:rounded-[3rem] rounded-none p-0'}`}>
             <div className="absolute inset-0 pointer-events-none opacity-20">
                <div className="absolute inset-0" style={{ background: 'radial-gradient(circle at 10% 90%, #0a0f1e 0%, transparent 40%), radial-gradient(circle at 90% 10%, #03041a 0%, transparent 40%)' }} />
             </div>
             
             <div className="p-4 md:p-10 border-b border-blue-500/20 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-transparent backdrop-blur-md relative overflow-hidden">
                <div className="relative z-10">
                   <h2 className="text-xl md:text-3xl font-black flex items-center gap-4 text-white uppercase tracking-tighter drop-shadow-[0_0_15px_rgba(59,130,246,0.6)]">
                      {isAssistantOpen ? <Bot className="text-emerald-400" size={24} /> : <ShieldCheck className="text-blue-400" size={24} />}
                      {isAssistantOpen ? "Central Assistant" : "Diagnostic Copilot"}
                   </h2>
                   <div className="flex items-center gap-3 mt-2">
                     <div className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse shadow-[0_0_8px_#3b82f6]"></div>
                     {isAssistantOpen ? (
                           <p className="text-emerald-400 text-[10px] font-black uppercase tracking-[0.2em]">RAG Orchestration Eng • Level 4 Access</p>
                     ) : activeAnomaly ? (
                       <p className="text-blue-400 text-[10px] font-black uppercase tracking-[0.2em]">Resolving High-Intensity Incident #{activeAnomaly.id}</p>
                     ) : (
                       <p className="text-slate-500 text-[10px] font-black uppercase tracking-[0.2em]">System State: Surveillance Standby</p>
                     )}
                   </div>
                </div>
                <div className="flex flex-wrap items-center gap-3 md:gap-6 relative z-10 w-full md:w-auto">
                    {isAssistantOpen && <AssistantMachineSelector />}
                    {isAssistantOpen && activeAssistantSessionId && (
                      <button 
                        onClick={() => dispatch(exportAssistantSession(activeAssistantSessionId))}
                        className="flex-1 md:flex-initial flex items-center justify-center gap-2 px-4 md:px-6 py-2 md:py-2.5 bg-blue-600/90 hover:bg-blue-500 text-white text-[9px] md:text-[10px] font-black uppercase tracking-[0.1em] rounded-xl transition-all shadow-lg active:scale-95 border border-blue-400/30"
                      >
                         <Download size={14} /> Report
                      </button>
                    )}
                    {!isAssistantOpen && activeAnomaly && !activeAnomaly.resolved && (
                      <button 
                        onClick={() => setIsResolveModalOpen(true)}
                        className="flex-1 md:flex-initial flex items-center justify-center gap-3 px-4 md:px-8 py-2.5 md:py-3 bg-emerald-600/90 hover:bg-emerald-500 text-white text-[9px] md:text-[11px] font-black uppercase tracking-[0.1em] rounded-2xl transition-all shadow-[0_0_40px_rgba(16,185,129,0.3)] active:scale-95 animate-pulse border border-emerald-400/30 backdrop-blur-sm"
                      >
                         <CheckCircle size={16} /> Complete
                      </button>
                    )}
                     <div className="flex items-center gap-4 ml-auto md:ml-0">
                       {isAssistantOpen && (
                         <button 
                           onClick={() => setIsMobileHistoryOpen(true)}
                           className="md:hidden p-2 text-slate-400 hover:text-emerald-400 transition-all transform hover:scale-110"
                           title="Assistant History"
                         >
                           <History size={22} />
                         </button>
                       )}
                       <button onClick={() => setIsChatMaximized(!isChatMaximized)} className="hidden md:block text-slate-400 hover:text-white transition-all transform hover:scale-110">
                          {isChatMaximized ? <Minimize2 size={22} /> : <Maximize2 size={22} />}
                      </button>
                      <button onClick={() => { setIsChatOpen(false); dispatch(setAssistantOpen(false)); }} className="text-slate-400 hover:text-red-500 transition-all transform hover:scale-110">
                          <X size={26} />
                      </button>
                    </div>
                </div>
             </div>

             <div className="flex flex-1 overflow-hidden min-h-0">
                <div className="hidden md:block">
                   {isAssistantOpen && <AssistantSidebar />}
                </div>

                   <div className="flex-1 flex flex-col overflow-hidden bg-[#0a0f1e] relative">
                      {/* Mobile History Drawer Overlay */}
                      {isAssistantOpen && isMobileHistoryOpen && (
                        <div className="fixed inset-0 z-[70] bg-[#020617] md:hidden animate-in slide-in-from-left duration-500 flex flex-col">
                           <div className="p-6 border-b border-blue-500/20 flex justify-between items-center bg-[#0a0f1e]/80 backdrop-blur-md">
                              <div className="flex items-center gap-3">
                                 <History className="text-emerald-400" size={20} />
                                 <h3 className="text-white font-black uppercase text-xs tracking-[0.2em]">Interaction Memory</h3>
                              </div>
                              <button 
                                onClick={() => setIsMobileHistoryOpen(false)}
                                className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl transition-all"
                              >
                                <X size={24} />
                              </button>
                           </div>
                           <div className="flex-1 overflow-hidden">
                              <AssistantSidebar onItemClick={() => setIsMobileHistoryOpen(false)} />
                           </div>
                        </div>
                      )}

                      <div className="relative flex-1 overflow-y-auto p-4 md:p-8 space-y-6 md:space-y-8 scrollbar-thin scrollbar-thumb-slate-800">
                         {loadingHistory[isAssistantOpen ? 'assistant' : (activeAnomaly?.id.toString() || 'general')] ? (
                             <div className="h-full flex flex-col items-center justify-center opacity-40">
                                 <RotateCw size={48} className="animate-spin text-blue-500 mb-6" />
                                 <p className="text-xs font-bold uppercase tracking-widest">Restoring Context...</p>
                             </div>
                         ) : (
                             (chatHistory[isAssistantOpen ? 'assistant' : (activeAnomaly?.id.toString() || 'general')] || []).map((msg, index) => {
                         const hasSuggestion = msg.role === 'agent' && msg.content.includes('[SUGGESTION:');
                         const displayContent = msg.content
                             .replace(/\[PROCEDURE_START\][\s\S]*?\[PROCEDURE_END\]/gi, '')
                             .replace(/\[ACTION:\s*(.*?)\]/gi, '')
                             .replace(/\[SUGGESTION:.*?\]/gi, '')
                             .trim();
                         const isFinalSummary = msg.metadata?.type === 'final_summary';

                         if (msg.type === 'phase_header') {
                           return (
                             <div key={index} className="flex justify-center my-6">
                               <div className="bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 text-white px-4 md:px-8 py-3 md:py-4 rounded-2xl border border-slate-600 shadow-xl text-center max-w-lg mx-auto">
                                 <div className="text-xs md:text-sm font-black leading-relaxed">
                                   <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                 </div>
                               </div>
                             </div>
                           );
                         }

                         if (msg.type === 'step' && msg.stepData) {
                           const sd = msg.stepData;
                           const isSafety = sd.phaseType === 'safety';
                           const isResponded = !!msg.stepResponse;

                           return (
                             <div key={index} className="flex justify-start">
                               <div className={`max-w-[100%] md:max-w-[90%] w-full rounded-2xl md:rounded-3xl shadow-2xl border overflow-hidden transition-all duration-500 ${
                                 isResponded ? 'bg-slate-900 border-slate-700/50 opacity-70' : isSafety ? 'bg-[#1a0f02] border-amber-500/40' : 'bg-[#0a0f1e] border-blue-500/40'
                               }`}>
                                 <div className={`px-4 md:px-6 py-2 md:py-3 flex items-center justify-between ${isSafety ? 'bg-amber-500/10 border-b border-amber-500/20' : 'bg-blue-500/10 border-b border-blue-500/20'}`}>
                                   <div className="flex items-center gap-2">
                                     {isSafety ? <ShieldAlert size={14} className="text-amber-400" /> : <Wrench size={14} className="text-blue-400" />}
                                     <span className={`text-[9px] md:text-[10px] font-black uppercase tracking-widest ${isSafety ? 'text-amber-400' : 'text-blue-400'}`}>{sd.subphaseTitle}</span>
                                   </div>
                                   <span className="text-[9px] md:text-[10px] font-bold text-slate-500 bg-slate-800 px-2 md:px-3 py-1 rounded-full">{sd.stepIndex + 1} / {sd.totalSteps}</span>
                                 </div>

                                 <div className="p-4 md:p-6">
                                   <div className="text-white text-sm md:text-base leading-relaxed">{renderStepContent(msg.content, msg.images)}</div>
                                   {sd.critical && !isResponded && <div className="inline-flex items-center gap-1.5 bg-red-500/15 text-red-400 text-[9px] md:text-[10px] font-black uppercase tracking-widest px-2.5 py-1.5 rounded-lg border border-red-500/30 mt-3"><AlertTriangle size={12} /> Safety Critical</div>}
                                   {isResponded && msg.stepResponse && (
                                     <div className={`mt-4 md:mt-5 flex items-center gap-2 text-xs md:text-sm font-bold rounded-xl px-4 py-3 ${msg.stepResponse.status === 'done' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
                                       <CheckCircle size={16} /><span>{msg.stepResponse.status === 'done' ? 'Completed' : 'Stuck'}</span>
                                       {msg.stepResponse.comment && <span className="text-slate-400 font-normal ml-2 truncate">&mdash; &quot;{msg.stepResponse.comment}&quot;</span>}
                                     </div>
                                   )}
                                   {!isResponded && (
                                     <div className="mt-4 md:mt-6 space-y-3">
                                       <div className="flex flex-wrap gap-2">
                                         <button onClick={() => handleClarifyStep(sd.stepId, msg.content || sd.stepText || '')} className="flex-1 py-2 md:py-2.5 px-3 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 border border-blue-500/20 rounded-xl text-[9px] md:text-[10px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2"><MessageCircle size={13} /> Detail</button>
                                         <button onClick={() => handleStepResponse(sd.stepId, 'cant_do', msg.content || sd.stepText)} className="flex-1 py-2 md:py-2.5 px-3 bg-red-600/10 hover:bg-red-600/20 text-red-400 border border-red-500/20 rounded-xl text-[9px] md:text-[10px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2"><AlertTriangle size={13} /> Stuck</button>
                                         <button onClick={() => handleStepResponse(sd.stepId, 'done', msg.content || sd.stepText)} className="flex-1 py-2 md:py-2.5 px-3 bg-emerald-600/10 hover:bg-emerald-600 hover:text-white text-emerald-400 border border-emerald-500/20 rounded-xl text-[9px] md:text-[10px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2"><CheckCircle size={13} /> Done</button>
                                       </div>
                                       <div className="flex gap-2 items-center bg-slate-900/80 border border-slate-700/60 rounded-2xl px-3 md:px-4 py-2.5 md:py-3 focus-within:border-blue-500/50 transition-colors">
                                         <input className="flex-1 bg-transparent text-xs md:text-sm text-slate-200 placeholder-slate-500 focus:outline-none" placeholder="Type a message..." value={stepChatInputs[sd.stepId] || ''} onChange={(e) => setStepChatInputs(prev => ({...prev, [sd.stepId]: e.target.value}))} onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSendStepMessage(sd.stepId, msg.content))} />
                                         <button onClick={() => handleSendStepMessage(sd.stepId, msg.content)} disabled={!stepChatInputs[sd.stepId]?.trim()} className="text-blue-400 hover:text-blue-300 disabled:opacity-30 p-1"><ArrowRight size={18} /></button>
                                       </div>
                                     </div>
                                   )}
                                 </div>
                               </div>
                             </div>
                           );
                         }

                         if (msg.type === 'procedure_complete') {
                           return (
                             <div key={index} className="flex justify-center my-6">
                               <div className="bg-gradient-to-r from-emerald-900/30 to-emerald-800/20 border border-emerald-500/30 text-white px-6 md:px-10 py-6 md:py-8 rounded-2xl md:rounded-3xl shadow-2xl text-center max-w-lg mx-auto">
                                 <div className="text-3xl md:text-5xl mb-4">🎉</div>
                                 <div className="text-base md:text-lg font-black mb-2">Procedure Complete!</div>
                                 <p className="text-xs md:text-sm text-slate-400">Click <span className="text-emerald-400 font-bold">Complete</span> above to finalize.</p>
                               </div>
                             </div>
                           );
                         }

                         if (msg.role === 'user') {
                           return (
                             <div key={index} className="flex justify-end">
                               <div className="max-w-[85%] md:max-w-[75%] bg-[#1e3a8a] text-white rounded-2xl md:rounded-3xl rounded-br-none p-4 md:p-7 shadow-2xl border border-blue-500/30">
                                 <p className="text-xs md:text-sm leading-relaxed whitespace-pre-line">{msg.content}</p>
                               </div>
                             </div>
                           );
                         }

                         return (
                           <div key={index} className="flex justify-start group">
                             <div className={`max-w-[95%] md:max-w-[85%] rounded-2xl p-4 md:p-6 shadow-sm ${isFinalSummary ? 'bg-[#064e3b] border-emerald-500/50 text-emerald-50 w-full' : 'bg-[#111827] border-slate-700/50 text-slate-100 rounded-tl-none'}`}>
                               <div className="flex items-center gap-2 mb-2 opacity-50 text-[9px] md:text-[10px] uppercase tracking-widest font-bold">
                                 {isFinalSummary ? <CheckCircle className="w-3 h-3 text-emerald-400" /> : <Bot className="w-3 h-3" />}
                                 {isFinalSummary ? 'Completion Report' : isAssistantOpen && msg.role === 'agent' ? 'System Mentor' : 'AI Mentor'}
                               </div>
                               {msg.context_source && <div className="mb-2 inline-flex items-center gap-1.5 px-2 py-0.5 bg-slate-900 border border-slate-700/50 rounded-full text-[7px] md:text-[8px] font-black uppercase text-slate-400"><Database size={10} /> Source: {msg.context_source}</div>}
                               {displayContent && (
                                 <div className="text-xs md:text-sm leading-relaxed space-y-3 markdown-content">
                                   {(msg.images && msg.images.length > 0) ? renderStepContent(displayContent, msg.images) : <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayContent}</ReactMarkdown>}
                                 </div>
                               )}
                               {msg.stepResponse && (
                                 <div className={`mt-4 flex items-center gap-2 text-xs md:text-sm font-bold ${msg.stepResponse.status === 'done' ? 'text-emerald-400' : 'text-red-400'}`}>
                                   <CheckCircle size={16} /><span className="capitalize">{msg.stepResponse.status.replace('_', ' ')}</span>
                                   {msg.stepResponse.comment && <span className="text-slate-400 font-normal ml-2">&mdash; &quot;{msg.stepResponse.comment}&quot;</span>}
                                 </div>
                               )}
                               {msg.hasSuggestion && !activeAnomaly?.resolved && (
                                 <div className="mt-6 pt-5 border-t border-slate-700/50">
                                   <button onClick={() => handleManualInquiry("Generate full step-by-step repair procedure")} className="w-full py-3 md:py-4 px-4 md:px-6 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-black text-xs md:text-sm uppercase tracking-widest rounded-2xl flex items-center justify-center gap-3 transition-all shadow-xl border border-blue-400/30"><ShieldCheck size={18} /> START REPAIR <ArrowRight size={16} /></button>
                                 </div>
                               )}
                               {!msg.stepResponse && (msg.type === 'step_clarification' || msg.type === 'branching_advice' || msg.type === 'wizard_step') && (
                                 <div className="mt-6 pt-5 border-t border-slate-700/50 space-y-3">
                                   <div className="flex flex-wrap gap-2">
                                     <button onClick={() => msg.stepData?.stepId && handleClarifyStep(msg.stepData.stepId, msg.stepData.stepText || msg.content)} className="flex-1 py-2 md:py-2.5 px-3 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 border border-blue-500/20 rounded-xl text-[9px] md:text-[10px] font-black uppercase tracking-widest flex items-center justify-center gap-2"><MessageCircle size={13} /> Detail</button>
                                     <button onClick={() => msg.stepData?.stepId && handleStepResponse(msg.stepData.stepId, 'cant_do')} className="flex-1 py-2 md:py-2.5 px-3 bg-red-600/10 hover:bg-red-600/20 text-red-400 border border-red-500/20 rounded-xl text-[9px] md:text-[10px] font-black uppercase tracking-widest flex items-center justify-center gap-2"><AlertTriangle size={13} /> Stuck</button>
                                     <button onClick={() => msg.stepData?.stepId && handleStepResponse(msg.stepData.stepId, 'done')} className="flex-1 py-2 md:py-2.5 px-3 bg-emerald-600/10 hover:bg-emerald-600 text-white text-emerald-400 border border-emerald-500/20 rounded-xl text-[9px] md:text-[10px] font-black uppercase tracking-widest flex items-center justify-center gap-2"><CheckCircle size={13} /> Done</button>
                                   </div>
                                   {msg.stepData?.stepId && (
                                     <div className="flex gap-2 items-center bg-[#0a0f1e]/95 border border-slate-700/60 rounded-2xl px-3 md:px-4 py-2.5 md:py-3 focus-within:border-blue-500/50 transition-colors">
                                       <input className="flex-1 bg-transparent text-xs md:text-sm text-slate-200 placeholder-slate-500 focus:outline-none" placeholder="Talk to AI mentor..." value={stepChatInputs[msg.stepData.stepId] || ''} onChange={(e) => setStepChatInputs(prev => ({...prev, [msg.stepData!.stepId]: e.target.value}))} onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && msg.stepData?.stepId && (e.preventDefault(), handleSendStepMessage(msg.stepData.stepId, msg.stepData.stepText || msg.content))} />
                                       <button onClick={() => msg.stepData?.stepId && handleSendStepMessage(msg.stepData.stepId, msg.stepData.stepText || msg.content)} disabled={!msg.stepData?.stepId || !stepChatInputs[msg.stepData.stepId]?.trim()} className="text-blue-400 hover:text-blue-300 disabled:opacity-30 p-1"><ArrowRight size={18} /></button>
                                     </div>
                                   )}
                                 </div>
                               )}
                             </div>
                           </div>
                         );
                       })
                         )}
                         <div ref={chatEndRef} />
                      </div>

                  <div className="relative p-4 md:p-8 bg-[#000000] border-t border-slate-800 space-y-6">
                      <div className="flex items-center gap-3 md:gap-4">
                         <input 
                             className={`flex-1 bg-slate-900 border border-slate-800 rounded-2xl px-4 md:px-6 py-3 md:py-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-200 ${(!isAssistantOpen && activeAnomaly?.resolved) ? 'opacity-50 cursor-not-allowed' : ''}`}
                             placeholder={isAssistantOpen ? "Ask anything..." : activeAnomaly?.resolved ? "Archived" : "Type query..."}
                             value={query}
                             readOnly={!isAssistantOpen && activeAnomaly?.resolved}
                             onChange={(e) => setQuery(e.target.value)}
                             onKeyDown={(e) => e.key === 'Enter' && handleManualInquiry()}
                         />
                         <button 
                            onClick={() => handleManualInquiry()}
                            className="p-3 md:p-4 bg-blue-600 hover:bg-blue-500 text-white rounded-2xl transition-all shadow-xl shadow-blue-500/20 active:scale-95"
                         >
                            <Send size={20} />
                         </button>
                    </div>
                  </div>
               </div>
             </div>
           </div>
        </div>
      )}

      {isResolveModalOpen && (
        <div className="fixed inset-0 bg-transparent backdrop-blur-xl z-[60] flex items-center justify-center p-4">
           <div className="bg-slate-900/40 backdrop-blur-2xl border border-slate-800 rounded-[3rem] p-10 max-w-lg w-full shadow-4xl animate-in zoom-in-95 duration-300">
              {resolveThankYouMessage ? (
                <>
                  <div className="h-16 w-16 bg-emerald-600/20 text-emerald-500 rounded-3xl flex items-center justify-center mb-6 mx-auto">
                     <CheckCircle size={32} />
                  </div>
                  <div className="prose prose-invert prose-sm max-w-none mb-8">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {resolveThankYouMessage}
                    </ReactMarkdown>
                  </div>
                  <div className="flex gap-4">
                     <button onClick={handleCloseResolveModal} className="flex-1 px-6 py-4 rounded-2xl bg-emerald-600 text-white font-black hover:bg-emerald-500 transition-all text-xs uppercase tracking-widest shadow-xl shadow-emerald-500/20">Done</button>
                     <button onClick={handleOpenAssistantFromResolve} className="flex-1 px-6 py-4 rounded-2xl bg-blue-600 text-white font-bold hover:bg-blue-500 transition-all text-xs uppercase tracking-widest flex items-center justify-center gap-2"><MessageCircle size={16} /> Open Assistant</button>
                  </div>
                </>
              ) : (
                <>
                  <div className="h-16 w-16 bg-emerald-600/20 text-emerald-500 rounded-3xl flex items-center justify-center mb-6"><CheckCircle size={32} /></div>
                  <h2 className="text-2xl font-black text-white mb-2">Finalize Diagnostic</h2>
                  <p className="text-slate-400 text-sm leading-relaxed mb-6">Briefly describe the **actual manual actions** taken on the machine. This will be vectorized into the AI knowledge base for future troubleshooting.</p>
                  {resolveValidation?.validationError && (
                    <div className="mb-6 p-4 bg-amber-500/10 border border-amber-500/30 rounded-2xl">
                      <div className="flex items-start gap-3">
                        <AlertTriangle className="text-amber-500 flex-shrink-0 mt-0.5" size={18} />
                        <div>
                          <p className="text-amber-400 text-sm font-medium mb-2">{resolveValidation.validationError}</p>
                          {resolveValidation.suggestions?.length > 0 && (
                            <ul className="text-amber-300/70 text-xs space-y-1">{resolveValidation.suggestions.map((s, i) => (<li key={i}>• {s}</li>))}</ul>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                  <textarea 
                    className={`w-full bg-[#0a0f1e] border rounded-3xl p-6 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 min-h-[150px] mb-6 ${resolveValidation?.validationError ? 'border-amber-500/50' : 'border-slate-800'}`}
                    placeholder="e.g. Manually bled the hydraulic lines and replaced the leaking filter at ACV-102. Thermal integrity restored."
                    value={operatorFix}
                    onChange={(e) => setOperatorFix(e.target.value)}
                    disabled={resolveValidation?.isValidating}
                  />
                  <div className="flex gap-4">
                     <button onClick={handleCloseResolveModal} disabled={resolveValidation?.isValidating} className="flex-1 px-6 py-4 rounded-2xl bg-slate-800 text-slate-400 font-bold hover:text-white transition-all text-xs uppercase tracking-widest disabled:opacity-50">Cancel</button>
                     <button onClick={handleResolve} disabled={!operatorFix.trim() || resolveValidation?.isValidating} className="flex-[2] px-6 py-4 rounded-2xl bg-emerald-600 text-white font-black hover:bg-emerald-500 transition-all text-xs uppercase tracking-widest shadow-xl shadow-emerald-500/20 disabled:opacity-30 flex items-center justify-center gap-2">
                        {resolveValidation?.isValidating ? (<><div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>Validating...</>) : ('Archive Incident')}
                     </button>
                  </div>
                </>
              )}
           </div>
        </div>
      )}
      <ExportProgressModal isOpen={exportProgress.isExporting} progress={exportProgress.progress} status={exportProgress.status} errorMessage={exportProgress.errorMessage} onClose={() => dispatch(resetExportProgress())} />
    </div>
  );
}
