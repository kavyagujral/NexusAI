import React from 'react';
import { Play, Clock, Eye, FileText, Waves } from 'lucide-react';
import type { RetrievalResult } from '../types';

interface RetrievalResultCardProps {
  result: RetrievalResult;
  onJumpToTimestamp?: (timestampSeconds: number) => void;
}

function confColor(score: number) {
  if (score >= 0.5) return { text: 'text-emerald-400', bg: 'rgba(52,211,153,0.08)', border: 'rgba(52,211,153,0.2)' };
  if (score >= 0.3) return { text: 'text-amber-400', bg: 'rgba(251,191,36,0.07)', border: 'rgba(251,191,36,0.18)' };
  return { text: 'text-slate-400', bg: 'rgba(100,116,139,0.08)', border: 'rgba(100,116,139,0.15)' };
}

export const RetrievalResultCard: React.FC<RetrievalResultCardProps> = ({
  result,
  onJumpToTimestamp,
}) => {
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
  const conf = confColor(result.similarity_score);

  let frameUrl = '';
  if (result.relevant_frame) {
    const normalized = result.relevant_frame.replace(/\\/g, '/');
    if (normalized.includes('data/')) {
      const sub = normalized.split('data/')[1];
      frameUrl = `${API_BASE_URL}/static/data/${sub}`;
    } else {
      frameUrl = `${API_BASE_URL}/static/data/frames/${result.video_id}/${normalized.split('/').pop()}`;
    }
  }

  const transcriptText = result.transcript || result.answer;

  return (
    <div
      className="result-card group rounded-2xl overflow-hidden transition-all duration-200"
      style={{
        background: 'rgba(12, 17, 30, 0.8)',
        border: '1px solid rgba(255,255,255,0.06)',
        backdropFilter: 'blur(16px)',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.border = '1px solid rgba(59,103,246,0.25)';
        (e.currentTarget as HTMLElement).style.boxShadow = '0 0 20px rgba(59,103,246,0.08)';
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.border = '1px solid rgba(255,255,255,0.06)';
        (e.currentTarget as HTMLElement).style.boxShadow = 'none';
      }}
    >
      {/* Top accent bar */}
      <div className="h-[2px]"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(59,103,246,0.5), rgba(129,140,248,0.3), transparent)' }} />

      <div className="p-5 space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-2 flex-wrap">
            {/* Timestamp jump button */}
            <button
              onClick={() => onJumpToTimestamp?.(result.timestamp_start)}
              className="group/btn flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all duration-200 active:scale-95 cursor-pointer"
              style={{
                background: 'rgba(59,103,246,0.12)',
                border: '1px solid rgba(59,103,246,0.25)',
                color: '#93b2fd',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = 'rgba(59,103,246,0.2)';
                (e.currentTarget as HTMLElement).style.boxShadow = '0 0 12px rgba(59,103,246,0.3)';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = 'rgba(59,103,246,0.12)';
                (e.currentTarget as HTMLElement).style.boxShadow = 'none';
              }}
            >
              <Clock className="h-3 w-3" />
              <span>{result.formatted_timestamp}</span>
              <Play className="h-2.5 w-2.5 fill-current opacity-70 group-hover/btn:opacity-100 transition-opacity" />
            </button>

            {/* Visual labels */}
            {result.visual_context.slice(0, 2).map((label, idx) => (
              <span key={idx}
                className="flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-medium"
                style={{
                  background: 'rgba(6,182,212,0.07)',
                  border: '1px solid rgba(6,182,212,0.15)',
                  color: '#67e8f9',
                }}
              >
                <Eye className="h-2.5 w-2.5" />
                {label}
              </span>
            ))}
          </div>

          {/* Confidence */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold"
            style={{ background: conf.bg, border: `1px solid ${conf.border}` }}>
            <span className="text-slate-500 font-normal text-[10px]">conf.</span>
            <span className={conf.text}>{(result.similarity_score * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* Content grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Transcript */}
          <div className="rounded-xl p-4 space-y-2"
            style={{ background: 'rgba(6, 9, 14, 0.6)', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-600">
              <Waves className="h-3 w-3 text-cyan-600" />
              Audio · CLO05 GRU
            </div>
            <blockquote className="text-xs text-slate-300 italic border-l-2 border-cyan-700/40 pl-3 leading-relaxed line-clamp-4">
              "{transcriptText}"
            </blockquote>
          </div>

          {/* Frame */}
          <div className="rounded-xl overflow-hidden"
            style={{ background: 'rgba(6, 9, 14, 0.8)', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div className="flex items-center justify-between px-3 py-2">
              <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-600">
                <FileText className="h-3 w-3 text-nexus-600" />
                Frame · CLO04 CNN
              </div>
              <span className="text-[9px] text-slate-700 font-mono">@{result.timestamp_start}s</span>
            </div>
            <div className="relative aspect-video bg-slate-950 flex items-center justify-center">
              {frameUrl ? (
                <img
                  src={frameUrl}
                  alt={`Frame at ${result.timestamp_start}s`}
                  className="w-full h-full object-cover"
                  onError={(e) => { (e.target as HTMLElement).style.display = 'none'; }}
                />
              ) : (
                <div className="flex flex-col items-center gap-2 text-slate-700">
                  <Eye className="h-5 w-5" />
                  <span className="text-[10px]">No frame</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
