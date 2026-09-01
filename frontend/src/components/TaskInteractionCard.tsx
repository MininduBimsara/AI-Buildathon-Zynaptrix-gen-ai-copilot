'use client';
import React from 'react';
import { useDispatch } from 'react-redux';
import { AppDispatch } from '@/store/store';
import { inquireCopilot, AnomalyRecord } from '@/store/slices/copilotSlice';
import { CheckCircle2, AlertCircle, ArrowRightCircle, HelpCircle } from 'lucide-react';

interface TaskInteractionCardProps {
  actions: string[];
  machineId: string;
  activeAnomaly: AnomalyRecord | null;
  isLatestMessage: boolean;
  disabled?: boolean;
}

export default function TaskInteractionCard({ 
  actions, 
  machineId, 
  activeAnomaly, 
  isLatestMessage,
  disabled 
}: TaskInteractionCardProps) {
  const dispatch = useDispatch<AppDispatch>();

  if (!actions || actions.length === 0 || !isLatestMessage) return null;

  const handleAction = (actionLabel: string) => {
    if (disabled) return;

    // Dispatch the inquiry as if the user typed the action label
    // We prefix it with a nice status indicator for the chat history
    const isSuccess = actionLabel.toLowerCase().includes('complete') || 
                      actionLabel.toLowerCase().includes('yes') || 
                      actionLabel.toLowerCase().includes('✅');
    
    const prefix = isSuccess ? '✅ ' : '⚠️ ';
    const userMessage = `${prefix}${actionLabel}`;

    dispatch(inquireCopilot({
      machine_id: machineId,
      query: userMessage,
      machine_state: activeAnomaly ? 'ANOMALY_DIAGNOSTIC' : 'NORMAL',
      context_anomaly: activeAnomaly || undefined
    }));
  };

  const getIcon = (label: string) => {
    const l = label.toLowerCase();
    if (l.includes('complete') || l.includes('yes') || l.includes('✅')) return <CheckCircle2 size={16} />;
    if (l.includes('fail') || l.includes('no') || l.includes('❌')) return <AlertCircle size={16} />;
    if (l.includes('help') || l.includes('explain')) return <HelpCircle size={16} />;
    return <ArrowRightCircle size={16} />;
  };

  const getColorClass = (label: string) => {
    const l = label.toLowerCase();
    if (l.includes('complete') || l.includes('yes') || l.includes('✅')) 
      return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500 hover:text-white';
    if (l.includes('fail') || l.includes('no') || l.includes('❌')) 
      return 'bg-rose-500/10 text-rose-400 border-rose-500/30 hover:bg-rose-500 hover:text-white';
    return 'bg-blue-500/10 text-blue-400 border-blue-500/30 hover:bg-blue-500 hover:text-white';
  };

  return (
    <div className="mt-6 pt-4 border-t border-slate-700/50 flex flex-wrap gap-3 animate-in fade-in slide-in-from-bottom-2 duration-500">
      {actions.map((action, idx) => (
        <button
          key={idx}
          onClick={() => handleAction(action)}
          disabled={disabled}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-xl border text-xs font-black uppercase tracking-wider
            transition-all duration-300 transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed
            ${getColorClass(action)}
          `}
        >
          {getIcon(action)}
          {action.replace(/^[✅⚠️❌🔎]\s*/, '')}
        </button>
      ))}
    </div>
  );
}
