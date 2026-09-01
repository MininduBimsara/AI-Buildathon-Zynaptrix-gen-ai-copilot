"use client"
import React from 'react';
import { RotateCw, FileText, Check, X } from 'lucide-react';

interface ExportProgressModalProps {
  isOpen: boolean;
  progress: number;
  status: 'generating' | 'success' | 'error';
  errorMessage?: string;
  onClose: () => void;
}

export default function ExportProgressModal({
  isOpen,
  progress,
  status,
  errorMessage,
  onClose
}: ExportProgressModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#0a0f1e] border border-slate-700 rounded-2xl shadow-2xl p-8 min-w-[400px] max-w-md animate-in zoom-in-95 duration-300">
        
        {/* Status Icon */}
        <div className="flex justify-center mb-6">
          {status === 'generating' && (
            <div className="w-16 h-16 rounded-full bg-blue-600/20 border-2 border-blue-500 flex items-center justify-center">
              <RotateCw size={32} className="text-blue-500 animate-spin" />
            </div>
          )}
          {status === 'success' && (
            <div className="w-16 h-16 rounded-full bg-emerald-600/20 border-2 border-emerald-500 flex items-center justify-center">
              <Check size={32} className="text-emerald-500" />
            </div>
          )}
          {status === 'error' && (
            <div className="w-16 h-16 rounded-full bg-red-600/20 border-2 border-red-500 flex items-center justify-center">
              <X size={32} className="text-red-500" />
            </div>
          )}
        </div>

        {/* Title */}
        <h3 className="text-xl font-black text-white text-center mb-2 uppercase tracking-wider">
          {status === 'generating' && 'Generating Report'}
          {status === 'success' && 'Report Ready!'}
          {status === 'error' && 'Export Failed'}
        </h3>

        {/* Message */}
        <p className="text-slate-400 text-sm text-center mb-6">
          {status === 'generating' && 'Creating professional diagnostic document...'}
          {status === 'success' && 'Your PDF has been downloaded successfully'}
          {status === 'error' && (errorMessage || 'An error occurred during export')}
        </p>

        {/* Progress Bar */}
        {status === 'generating' && (
          <div className="mb-6">
            <div className="flex justify-between text-xs text-slate-500 mb-2 font-bold">
              <span>Progress</span>
              <span>{progress}%</span>
            </div>
            <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Action Buttons */}
        {status !== 'generating' && (
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-bold text-sm uppercase tracking-wider transition-all active:scale-95"
            >
              Close
            </button>
            {status === 'success' && (
              <button
                onClick={() => {
                  // Optional: Trigger another export
                  onClose();
                }}
                className="flex-1 px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-bold text-sm uppercase tracking-wider transition-all active:scale-95 flex items-center justify-center gap-2"
              >
                <FileText size={16} />
                Done
              </button>
            )}
          </div>
        )}

        {/* Loading Spinner Details */}
        {status === 'generating' && (
          <div className="mt-6 text-center">
            <div className="flex items-center justify-center gap-2 text-xs text-slate-600">
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></div>
              <span>This may take a few moments</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
