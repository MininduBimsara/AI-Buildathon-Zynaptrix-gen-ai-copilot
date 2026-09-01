"use client"
import React, { useState, useMemo } from 'react';
import { 
  Plus, 
  Search, 
  Trash2, 
  MessageSquare, 
  ChevronLeft, 
  ChevronRight,
  Clock,
  Settings,
  X,
  Download
} from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store/store';
import { 
  setAssistantSidebarOpen, 
  setActiveAssistantSessionId, 
  deleteAssistantSession,
  fetchAssistantHistory,
  exportAssistantSession
} from '../store/slices/copilotSlice';

export default function AssistantSidebar({ onItemClick }: { onItemClick?: () => void }) {
  const dispatch = useDispatch<AppDispatch>();
  const { 
    assistantSessions, 
    activeAssistantSessionId, 
    isAssistantSidebarOpen 
  } = useSelector((state: RootState) => state.copilot);
  
  const [searchQuery, setSearchQuery] = useState('');

  const filteredSessions = useMemo(() => {
    return assistantSessions.filter(s => 
      s.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.machine_id?.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [assistantSessions, searchQuery]);

  const groupSessionsByDate = (sessions: any[]) => {
    const now = new Date();
    const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    
    const groups: Record<string, any[]> = {
      'Today': [],
      'Previous': []
    };

    sessions.forEach(s => {
      if (!s.timestamp) return;
      const date = s.timestamp.split(' ')[0];
      if (date === today) groups['Today'].push(s);
      else groups['Previous'].push(s);
    });

    return groups;
  };

  const grouped = groupSessionsByDate(filteredSessions);

  if (!isAssistantSidebarOpen) {
    return (
      <div className="w-16 h-full bg-[#030712] border-r border-slate-800 flex flex-col items-center py-6 transition-all duration-300">
        <button 
          onClick={() => dispatch(setAssistantSidebarOpen(true))}
          className="p-2 hover:bg-slate-800 rounded-lg text-slate-500 hover:text-white transition-colors"
        >
          <ChevronRight size={20} />
        </button>
        <div className="mt-8 flex flex-col gap-6">
          <button 
            onClick={() => dispatch(setActiveAssistantSessionId(null))}
            className="p-3 bg-blue-600 hover:bg-blue-500 rounded-xl shadow-lg transition-transform active:scale-95"
          >
            <Plus size={20} className="text-white" />
          </button>
          <HistoryPreview sessions={assistantSessions.slice(0, 5)} />
        </div>
      </div>
    );
  }

  return (
    <div className="w-full md:w-80 h-full bg-[#030712] border-r border-slate-800 flex flex-col transition-all duration-300 relative group/sidebar shadow-[20px_0_50px_rgba(0,0,0,0.4)]">
      
      {/* 🛠️ Sidebar Controls */}
      <div className="p-4 md:p-6 pb-2">
        <div className="flex items-center justify-between mb-4 md:mb-6">
          <h2 className="text-xs md:text-sm font-black text-white uppercase tracking-widest flex items-center gap-2">
            <Clock size={16} className="text-blue-500" /> Interaction Memory
          </h2>
          <button 
            onClick={() => dispatch(setAssistantSidebarOpen(false))}
            className="text-slate-600 hover:text-white p-1 hover:bg-slate-800 rounded"
          >
            <ChevronLeft size={18} />
          </button>
        </div>

        <button
          onClick={() => {
            dispatch(setActiveAssistantSessionId(null));
            onItemClick?.();
          }}
          className="w-full flex items-center justify-center gap-3 px-4 py-2.5 md:py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-xl md:rounded-2xl font-black text-[10px] md:text-[11px] uppercase tracking-widest transition-all shadow-xl active:scale-[0.98] mb-4 md:mb-6"
        >
          <Plus size={16} /> New Inquiry
        </button>

        <div className="relative group/search">
          <Search size={14} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within/search:text-blue-500 transition-colors" />
          <input
            type="text"
            placeholder="Search..."
            className="w-full bg-[#0a0f1e] border border-slate-800 hover:border-slate-700 focus:border-blue-500/50 rounded-xl py-2.5 md:py-3 pl-10 pr-4 text-xs text-slate-300 placeholder-slate-600 outline-none transition-all"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* 🚀 Session Feed */}
      <div className="flex-1 overflow-y-auto px-3 md:px-4 py-4 space-y-6 md:space-y-8 scrollbar-thin scrollbar-thumb-slate-800/50">
        {Object.entries(grouped).map(([label, sessions]) => sessions.length > 0 && (
          <div key={label} className="space-y-2">
            <h3 className="px-3 text-[9px] md:text-[10px] font-black text-slate-600 uppercase tracking-widest">{label}</h3>
            {sessions.map(s => (
              <div 
                key={s.id} 
                className="group/item relative"
              >
                <button
                  onClick={() => {
                    dispatch(fetchAssistantHistory(s.id));
                    onItemClick?.();
                  }}
                  className={`w-full flex flex-col items-start gap-1 p-3 md:p-4 rounded-xl md:rounded-2xl transition-all border text-left ${
                    activeAssistantSessionId === s.id 
                    ? 'bg-blue-600/10 border-blue-500/50 text-white shadow-lg' 
                    : 'bg-transparent border-transparent hover:bg-slate-900 text-slate-400 opacity-60 hover:opacity-100 hover:scale-[1.01]'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <MessageSquare size={12} className={activeAssistantSessionId === s.id ? 'text-blue-400' : 'text-slate-600'} />
                    <span className="text-[9px] md:text-[10px] font-bold opacity-40 italic">{s.timestamp.split(' ')[1]}</span>
                  </div>
                  <p className="text-xs font-bold truncate w-full">{s.title}</p>
                  {s.machine_id && (
                    <span className="text-[9px] font-black bg-slate-800/80 px-2 py-0.5 rounded-lg text-slate-500 border border-slate-700/50">{s.machine_id}</span>
                  )}
                </button>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 flex gap-2 opacity-0 group-hover/item:opacity-100 transition-all">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      dispatch(exportAssistantSession(s.id));
                    }}
                    className="p-2 bg-blue-600 text-white rounded-xl shadow-xl transition-all hover:scale-110 active:scale-90"
                    title="Export to PDF"
                  >
                    <Download size={12} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      dispatch(deleteAssistantSession(s.id));
                    }}
                    className="p-2 bg-red-500 text-white rounded-xl shadow-xl transition-all hover:scale-110 active:scale-90"
                    title="Delete Session"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ))}
        {filteredSessions.length === 0 && (
          <div className="flex flex-col items-center py-10 text-slate-700 select-none">
            <X size={32} className="opacity-20 translate-y-4" />
            <p className="text-[10px] font-black uppercase tracking-widest">No history</p>
          </div>
        )}
      </div>

    </div>
  );
}

function HistoryPreview({ sessions }: { sessions: any[] }) {
  return (
    <div className="flex flex-col gap-3">
      {sessions.map(s => (
         <div key={s.id} className="w-8 h-8 rounded-lg bg-slate-800/50 border border-slate-700/50 flex items-center justify-center text-slate-600 hover:text-blue-500 hover:border-blue-500 transition-all cursor-pointer">
            <MessageSquare size={14} />
         </div>
      ))}
    </div>
  );
}
