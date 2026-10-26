import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Mic,
  Eye,
  Cpu,
  Layers,
  Database,
  Activity,
} from 'lucide-react';
import type { ProcessingJobStatus } from '../types';

interface ProcessingStatusCardProps {
  status: ProcessingJobStatus;
}

const PIPELINE_STAGES = [
  { key: 'EXTRACTING_AUDIO', label: 'FFmpeg Audio', icon: Mic, color: '#67e8f9' },
  { key: 'WHISPER_TRANSCRIPTION', label: 'Whisper STT', icon: Mic, color: '#a78bfa' },
  { key: 'FRAME_SAMPLING', label: '2s Frame Sampling', icon: Eye, color: '#93b2fd' },
  { key: 'CNN_ANALYSIS', label: 'ResNet-50 CNN', icon: Layers, color: '#60a5fa' },
  { key: 'GRU_ENCODING', label: 'PyTorch GRU', icon: Cpu, color: '#f472b6' },
  { key: 'TEMPORAL_ALIGNMENT', label: 'Stream Alignment', icon: Activity, color: '#34d399' },
  { key: 'NUMPY_ATTENTION_FUSION', label: 'NumPy Attention', icon: Cpu, color: '#fbbf24' },
  { key: 'CHROMADB_INDEXING', label: 'ChromaDB Index', icon: Database, color: '#fb923c' },
];

function getStageIndex(statusKey: string) {
  return PIPELINE_STAGES.findIndex(s => s.key === statusKey);
}

export const ProcessingStatusCard: React.FC<ProcessingStatusCardProps> = ({ status }) => {
  const isFailed = status.status === 'FAILED';
  const isCompleted = status.status === 'COMPLETED';
  const currentStageIdx = getStageIndex(status.status);

  return (
    <div
      className="rounded-2xl overflow-hidden animate-slide-up"
      style={{
        background: 'rgba(10, 14, 24, 0.95)',
        border: isFailed
          ? '1px solid rgba(239,68,68,0.25)'
          : isCompleted
            ? '1px solid rgba(52,211,153,0.25)'
            : '1px solid rgba(59,103,246,0.25)',
        boxShadow: isCompleted
          ? '0 0 30px rgba(52,211,153,0.1)'
          : isFailed
            ? '0 0 30px rgba(239,68,68,0.1)'
            : '0 0 30px rgba(59,103,246,0.12)',
      }}
    >
      {/* Animated top bar */}
      <div className="h-[2px] w-full relative overflow-hidden">
        <div
          className="h-full transition-all duration-700"
          style={{
            width: `${Math.max(5, status.progress_percent)}%`,
            background: isCompleted
              ? 'linear-gradient(90deg, #34d399, #10b981)'
              : isFailed
                ? '#ef4444'
                : 'linear-gradient(90deg, #3b67f6, #818cf8, #67e8f9)',
          }}
        />
        {!isCompleted && !isFailed && (
          <div
            className="absolute inset-0 h-full"
            style={{
              background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent)',
              backgroundSize: '200% 100%',
              animation: 'skeleton-shimmer 2s linear infinite',
            }}
          />
        )}
      </div>

      <div className="p-6 space-y-5">
        {/* Header */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div
              className="h-10 w-10 rounded-xl flex items-center justify-center shrink-0"
              style={{
                background: isCompleted
                  ? 'rgba(52,211,153,0.12)'
                  : isFailed
                    ? 'rgba(239,68,68,0.12)'
                    : 'rgba(59,103,246,0.12)',
                border: `1px solid ${isCompleted ? 'rgba(52,211,153,0.25)' : isFailed ? 'rgba(239,68,68,0.25)' : 'rgba(59,103,246,0.25)'}`,
              }}
            >
              {isCompleted ? (
                <CheckCircle2 className="h-5 w-5 text-emerald-400" />
              ) : isFailed ? (
                <AlertTriangle className="h-5 w-5 text-red-400" />
              ) : (
                <Loader2 className="h-5 w-5 text-nexus-400 animate-spin" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-bold text-white">
                  {isCompleted ? 'Spatiotemporal Index Ready' : isFailed ? 'Pipeline Failed' : 'ML Pipeline Running'}
                </h4>
                <span
                  className="px-2 py-0.5 text-[9px] font-bold tracking-wider uppercase rounded-full"
                  style={{
                    background: isCompleted ? 'rgba(52,211,153,0.15)' : isFailed ? 'rgba(239,68,68,0.15)' : 'rgba(59,103,246,0.15)',
                    color: isCompleted ? '#34d399' : isFailed ? '#f87171' : '#93b2fd',
                    border: `1px solid ${isCompleted ? 'rgba(52,211,153,0.2)' : isFailed ? 'rgba(239,68,68,0.2)' : 'rgba(59,103,246,0.2)'}`,
                  }}
                >
                  {status.status.replace(/_/g, ' ')}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5 truncate max-w-sm">{status.current_stage}</p>
            </div>
          </div>

          <div className="text-right shrink-0">
            <span
              className="text-2xl font-extrabold font-mono"
              style={{ color: isCompleted ? '#34d399' : isFailed ? '#f87171' : '#93b2fd' }}
            >
              {Math.round(status.progress_percent)}%
            </span>
          </div>
        </div>

        {/* Stage pipeline indicators */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {PIPELINE_STAGES.map((stage, idx) => {
            const Icon = stage.icon;
            const isActive = status.status === stage.key;
            const isDone = isCompleted || (currentStageIdx > idx && currentStageIdx !== -1);

            return (
              <div
                key={stage.key}
                className="pipeline-step flex items-center gap-2 px-3 py-2 rounded-xl text-[11px] transition-all duration-300"
                style={{
                  animationDelay: `${idx * 50}ms`,
                  background: isActive
                    ? `rgba(59,103,246,0.15)`
                    : isDone
                      ? 'rgba(52,211,153,0.06)'
                      : 'rgba(12,17,30,0.6)',
                  border: isActive
                    ? '1px solid rgba(59,103,246,0.4)'
                    : isDone
                      ? '1px solid rgba(52,211,153,0.15)'
                      : '1px solid rgba(255,255,255,0.05)',
                  color: isActive ? '#fff' : isDone ? '#34d399' : '#64748b',
                }}
              >
                <Icon
                  className="h-3 w-3 shrink-0"
                  style={{ color: isActive ? stage.color : isDone ? '#34d399' : '#475569' }}
                />
                <span className="truncate font-medium">{stage.label}</span>
                {isActive && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-nexus-400 pulse-ring shrink-0" />}
                {isDone && !isActive && <CheckCircle2 className="ml-auto h-3 w-3 text-emerald-500 shrink-0" />}
              </div>
            );
          })}
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-3 gap-3 pt-2 border-t border-white/[0.05]">
          {[
            { label: 'Frames Sampled', value: `${status.frames_extracted}`, unit: 'frames', color: '#67e8f9' },
            { label: 'Audio Segments', value: `${status.audio_segments_count}`, unit: 'segs', color: '#a78bfa' },
            { label: 'ChromaDB Vectors', value: `${status.chunks_indexed}`, unit: 'vecs', color: '#fbbf24' },
          ].map(({ label, value, unit, color }) => (
            <div key={label}
              className="rounded-xl px-4 py-3"
              style={{ background: 'rgba(6,9,14,0.7)', border: '1px solid rgba(255,255,255,0.05)' }}>
              <p className="text-[10px] text-slate-600 uppercase tracking-wider font-medium">{label}</p>
              <p className="text-lg font-extrabold font-mono mt-0.5" style={{ color }}>
                {value}
                <span className="text-[10px] text-slate-600 font-normal ml-1">{unit}</span>
              </p>
            </div>
          ))}
        </div>

        {/* Error */}
        {status.error_message && (
          <div className="rounded-xl p-3 text-xs text-red-300"
            style={{ background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.2)' }}>
            <strong className="text-red-400">Error: </strong>{status.error_message}
          </div>
        )}
      </div>
    </div>
  );
};
