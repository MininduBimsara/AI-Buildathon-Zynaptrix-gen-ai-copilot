"use client"
import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store/store';
import { setAssistantMachineId } from '../store/slices/copilotSlice';
import { Database, ChevronDown } from 'lucide-react';

export default function AssistantMachineSelector() {
  const dispatch = useDispatch<AppDispatch>();
  const { machines } = useSelector((state: RootState) => state.machines);
  const { assistantMachineId } = useSelector((state: RootState) => state.copilot);

  const selectedMachine = machines.find((m: any) => m.id === assistantMachineId);

  return (
    <div className="relative group">
      <div className="flex items-center gap-3 px-4 py-2 bg-slate-900 border border-slate-700/50 rounded-xl hover:border-blue-500/50 transition-all cursor-pointer">
        <Database size={14} className="text-blue-400" />
        <select
          value={assistantMachineId || ''}
          onChange={(e) => dispatch(setAssistantMachineId(e.target.value || null))}
          className="bg-transparent text-[11px] font-black uppercase tracking-widest text-slate-200 outline-none cursor-pointer pr-4 appearance-none"
        >
          <option key="default-none" value="" className="bg-[#0a0f1e]">No Specific Manual (General AI)</option>
          {machines.map((m: any, index: number) => (
            <option key={m.machine_id || `machine-${index}`} value={m.machine_id} className="bg-[#0a0f1e]">
              RAG: {m.name} Manual
            </option>
          ))}
        </select>
        <ChevronDown size={14} className="text-slate-500 group-hover:text-blue-400 transition-colors" />
      </div>
      <div className="absolute -top-6 left-2 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
        <span className="text-[9px] font-black text-blue-500 uppercase tracking-widest bg-blue-500/10 px-2 py-0.5 rounded">
          Set RAG Context
        </span>
      </div>
    </div>
  );
}
