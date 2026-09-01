"use client"
import React, { useState } from 'react';
import { UploadCloud, FileText, CheckCircle, AlertTriangle, Loader2, Database, ShieldCheck, ArrowRight } from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../../store/store';
import { uploadManual, clearUploadStatus } from '../../store/slices/ingestionSlice';

export default function IngestionPage() {
  const dispatch = useDispatch<AppDispatch>();
  const { isUploading, uploadStatus } = useSelector((state: RootState) => state.ingestion);
  
  const [manualId, setManualId] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);

  const handleUpload = () => {
    if (!uploadFile || !manualId) return;
    dispatch(uploadManual({ manualId, file: uploadFile })).then((res) => {
      if (res.meta.requestStatus === 'fulfilled') {
        setTimeout(() => {
          dispatch(clearUploadStatus());
          setManualId("");
          setUploadFile(null);
        }, 5000);
      }
    });
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <header className="mb-12">
        <h1 className="text-4xl font-black text-white mb-2 flex items-center gap-3">
          <Database className="text-blue-500" size={36} />
          Knowledge Management
        </h1>
        <p className="text-slate-400 text-lg">Ingest technical documentation into the RAG engine for expert diagnostic support.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Upload Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-blue-600/10 rounded-full blur-3xl -z-10 group-hover:bg-blue-600/20 transition-colors"></div>
          
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <UploadCloud size={20} className="text-blue-400" />
            Upload Technical Manual
          </h2>

          <div className="space-y-6">
            <div>
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Manual Identifier</label>
              <input 
                type="text" 
                value={manualId}
                onChange={(e) => setManualId(e.target.value.replace(/[^a-zA-Z0-9_-]/g, ''))}
                placeholder="e.g. LATHE_PRECISION_V2" 
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all text-white"
                disabled={isUploading}
              />
            </div>
            
            <div>
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">PDF Document</label>
              <label className={`flex flex-col items-center justify-center gap-4 bg-slate-950 border-2 border-dashed rounded-2xl p-8 transition-all cursor-pointer ${uploadFile ? 'border-blue-500/50 bg-blue-500/5' : 'border-slate-800 hover:border-slate-700'}`}>
                <div className={`p-4 rounded-full ${uploadFile ? 'bg-blue-500/20 text-blue-400' : 'bg-slate-800 text-slate-500'}`}>
                  <FileText size={32} />
                </div>
                <div className="text-center">
                  <span className="block text-sm font-bold text-slate-200">{uploadFile ? uploadFile.name : "Choose a file"}</span>
                  <span className="text-xs text-slate-500 mt-1 block">Maximum size: 50MB</span>
                </div>
                <input 
                  type="file" 
                  accept="application/pdf" 
                  className="hidden" 
                  onChange={(e) => setUploadFile(e.target.files ? e.target.files[0] : null)}
                  disabled={isUploading}
                />
              </label>
            </div>

            <button 
              onClick={handleUpload}
              disabled={!uploadFile || !manualId || isUploading}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-bold py-4 px-6 rounded-2xl transition-all flex items-center justify-center gap-3 shadow-xl shadow-blue-500/20 group"
            >
              {isUploading ? <Loader2 size={20} className="animate-spin" /> : <UploadCloud size={20} />}
              {isUploading ? "Vectorizing Document..." : "Initialize Ingestion"}
              {!isUploading && <ArrowRight size={18} className="translate-x-0 group-hover:translate-x-1 transition-transform" />}
            </button>

            {uploadStatus && (
              <div className={`p-4 rounded-xl border flex items-start gap-3 ${
                uploadStatus.includes('Successful') ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 
                uploadStatus.includes('Error') ? 'bg-red-500/10 border-red-500/20 text-red-400' : 
                'bg-blue-500/10 border-blue-500/20 text-blue-400'
              }`}>
                {uploadStatus.includes('Successful') ? <CheckCircle size={18} className="mt-0.5 shrink-0" /> : 
                 uploadStatus.includes('Error') ? <AlertTriangle size={18} className="mt-0.5 shrink-0" /> : 
                 <Loader2 size={18} className="animate-spin mt-0.5 shrink-0" />}
                <p className="text-sm font-medium leading-tight">{uploadStatus}</p>
              </div>
            )}
          </div>
        </div>

        {/* Guidance Card */}
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <ShieldCheck size={20} className="text-emerald-400" />
              Ingestion Protocol
            </h3>
            <ul className="space-y-4">
              <li className="flex gap-3 text-sm text-slate-400">
                <div className="h-5 w-5 rounded-full bg-slate-800 flex items-center justify-center text-[10px] font-bold text-white shrink-0">1</div>
                Use unique identifiers for machine manuals to ensure RAG routing precision.
              </li>
              <li className="flex gap-3 text-sm text-slate-400">
                <div className="h-5 w-5 rounded-full bg-slate-800 flex items-center justify-center text-[10px] font-bold text-white shrink-0">2</div>
                PDFs should contain clear technical diagrams for the Vision model to interpret.
              </li>
              <li className="flex gap-3 text-sm text-slate-400">
                <div className="h-5 w-5 rounded-full bg-slate-800 flex items-center justify-center text-[10px] font-bold text-white shrink-0">3</div>
                The system will automatically perform layout detection and multimodal vectorization.
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-indigo-600/20 to-blue-600/20 border border-blue-500/20 rounded-3xl p-8">
            <h3 className="text-white font-bold mb-2">Experimental Feature</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              We are currently testing Multi-Manual Cross-Referencing. Ensure your table of contents is well-structured for optimal retrieval accuracy.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
