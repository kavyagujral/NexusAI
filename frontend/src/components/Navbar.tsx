import React from 'react';
import { Layers, Cpu, Database, Video, Zap } from 'lucide-react';

interface NavbarProps {
  onOpenUpload: () => void;
  activeVideoCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({ onOpenUpload, activeVideoCount }) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/[0.06] bg-surface-950/70 backdrop-blur-2xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-[60px] flex items-center justify-between">

        {/* Logo */}
        <div className="flex items-center space-x-3">
          <div className="relative h-9 w-9">
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-nexus-500 to-indigo-500 opacity-20 blur-sm" />
            <div className="relative h-9 w-9 rounded-xl bg-gradient-to-br from-nexus-600 via-indigo-500 to-cyan-400 p-px">
              <div className="h-full w-full bg-surface-950 rounded-[11px] flex items-center justify-center">
                <Layers className="h-4.5 w-4.5 text-nexus-300" style={{ width: 18, height: 18 }} />
              </div>
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-[17px] tracking-tight text-white">
                Nexus
              </span>
              <span className="px-2 py-0.5 text-[9px] font-bold tracking-widest uppercase rounded-full bg-nexus-500/15 text-nexus-300 border border-nexus-500/25 leading-none pt-[3px]">
                RAG
              </span>
            </div>
            <p className="text-[11px] text-slate-500 hidden sm:block leading-none mt-0.5">
              Spatiotemporal Multimodal Engine
            </p>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3">
          {/* System status pills */}
          <div className="hidden md:flex items-center gap-2">
            <div className="badge-pill bg-emerald-500/8 text-emerald-400 border-emerald-500/20">
              <Cpu className="h-3 w-3" />
              <span>NumPy Attention</span>
            </div>
            <div className="badge-pill bg-amber-500/8 text-amber-400 border-amber-500/20">
              <Database className="h-3 w-3" />
              <span>ChromaDB</span>
            </div>
            {activeVideoCount > 0 && (
              <div className="badge-pill bg-nexus-500/10 text-nexus-300 border-nexus-500/25">
                <Zap className="h-3 w-3" />
                <span>{activeVideoCount} Indexed</span>
              </div>
            )}
          </div>

          <button
            onClick={onOpenUpload}
            className="relative flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold text-white transition-all duration-200 active:scale-95 cursor-pointer overflow-hidden group"
            style={{
              background: 'linear-gradient(135deg, #2547eb 0%, #3b67f6 50%, #4f46e5 100%)',
              boxShadow: '0 0 20px rgba(59, 103, 246, 0.35), 0 2px 8px rgba(0,0,0,0.3)',
            }}
          >
            <span className="absolute inset-0 bg-white/0 group-hover:bg-white/10 transition-all duration-200" />
            <Video className="h-4 w-4 relative z-10" />
            <span className="relative z-10">Upload Video</span>
          </button>
        </div>
      </div>
    </header>
  );
};
