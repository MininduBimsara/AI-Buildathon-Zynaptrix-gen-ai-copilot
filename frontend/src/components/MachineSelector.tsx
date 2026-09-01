"use client"
import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Cpu, Zap, Droplets, RotateCw, Settings, Search } from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store/store';
import { setCurrentMachineId } from '../store/slices/machineSlice';

export default function MachineSelector() {
  const dispatch = useDispatch<AppDispatch>();
  const { machines, currentMachineId } = useSelector((state: RootState) => state.machines);
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const selectedMachine = machines.find(m => m.machine_id === currentMachineId);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const getMachineIcon = (machineId: string) => {
    if (machineId.includes('PUMP')) return <Droplets size={16} className="text-blue-400" />;
    if (machineId.includes('TURBINE')) return <RotateCw size={16} className="text-emerald-400" />;
    if (machineId.includes('LATHE')) return <Cpu size={16} className="text-purple-400" />;
    if (machineId.includes('GEN')) return <Zap size={16} className="text-amber-400" />;
    return <Settings size={16} className="text-slate-400" />;
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-3 px-4 py-2 bg-slate-900/50 hover:bg-slate-800 border border-slate-700/50 hover:border-blue-500/50 rounded-xl transition-all group"
      >
        <div className="p-1.5 bg-slate-800 rounded-lg group-hover:bg-blue-500/10 transition-colors">
          {selectedMachine ? getMachineIcon(selectedMachine.machine_id) : <Search size={16} className="text-slate-500" />}
        </div>
        <div className="text-left hidden sm:block">
          <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest leading-none mb-1">Active Context</p>
          <p className="text-xs font-bold text-slate-200">{selectedMachine?.name || "Global Search"}</p>
        </div>
        <ChevronDown size={14} className={`text-slate-500 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-72 md:w-64 bg-[#0a0f1e] border border-slate-800 rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] overflow-hidden z-[100] animate-in fade-in zoom-in-95 duration-200">
          <div className="p-3 border-b border-slate-800 bg-slate-900/30">
            <p className="text-[9px] md:text-[10px] font-black text-slate-500 uppercase tracking-widest px-2">Select Machine Context</p>
          </div>
          <div className="max-h-80 overflow-y-auto p-2">
            <button
              onClick={() => {
                dispatch(setCurrentMachineId(null as any));
                setIsOpen(false);
              }}
              className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all ${
                !currentMachineId ? 'bg-blue-600/10 border border-blue-500/30 text-white' : 'hover:bg-slate-800/50 text-slate-400'
              }`}
            >
              <div className="p-1.5 bg-slate-800 rounded-lg"><Search size={14} /></div>
              <div className="text-left">
                <p className="text-xs font-bold">General Assistant</p>
                <p className="text-[10px] opacity-60">Search all platform docs</p>
              </div>
            </button>
            
            {machines.map((m, index) => (
              <button
                key={m.machine_id || `main-machine-${index}`}
                onClick={() => {
                  dispatch(setCurrentMachineId(m.machine_id));
                  setIsOpen(false);
                }}
                className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all mt-1 ${
                  currentMachineId === m.machine_id ? 'bg-blue-600/10 border border-blue-500/30 text-white' : 'hover:bg-slate-800/50 text-slate-400'
                }`}
              >
                <div className="p-1.5 bg-slate-800 rounded-lg">{getMachineIcon(m.machine_id)}</div>
                <div className="text-left">
                  <p className="text-xs font-bold">{m.name}</p>
                  <p className="text-[10px] opacity-60">Code: {m.machine_id}</p>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
