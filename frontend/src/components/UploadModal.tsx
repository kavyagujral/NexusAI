import React, { useState, useRef } from 'react';
import { Upload, X, Film, AlertCircle, Loader2, HardDrive } from 'lucide-react';
import { uploadVideo } from '../services/api';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (videoId: string, jobId: string) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onUploadSuccess,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = () => setIsDragging(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
    }
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) { setError('Please select a video file first.'); return; }
    setIsUploading(true);
    setError(null);
    try {
      const res = await uploadVideo(file);
      onUploadSuccess(res.video_id, res.job_id);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to upload video');
    } finally {
      setIsUploading(false);
    }
  };

  const fileSizeMB = file ? (file.size / (1024 * 1024)).toFixed(1) : null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
      style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)' }}>
      <div
        className="relative w-full max-w-lg animate-slide-up overflow-hidden rounded-3xl"
        style={{
          background: 'rgba(10, 14, 24, 0.97)',
          border: '1px solid rgba(255,255,255,0.08)',
          boxShadow: '0 24px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04)',
        }}
      >
        {/* Top glow strip */}
        <div className="absolute top-0 left-0 right-0 h-px"
          style={{ background: 'linear-gradient(90deg, transparent, rgba(59,103,246,0.6), transparent)' }} />

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl flex items-center justify-center"
              style={{ background: 'rgba(59,103,246,0.12)', border: '1px solid rgba(59,103,246,0.2)' }}>
              <Film className="h-5 w-5 text-nexus-400" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">Upload Educational Video</h3>
              <p className="text-[11px] text-slate-500 mt-0.5">Zoom recordings, lectures, screen shares</p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isUploading}
            className="h-8 w-8 rounded-lg flex items-center justify-center text-slate-500 hover:text-white hover:bg-white/5 transition-all cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <form onSubmit={handleSubmit} className="px-6 py-5 space-y-4">
          {/* Dropzone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="cursor-pointer rounded-2xl p-8 text-center transition-all duration-200"
            style={{
              border: isDragging
                ? '2px dashed rgba(59,103,246,0.7)'
                : file
                  ? '2px dashed rgba(52,211,153,0.4)'
                  : '2px dashed rgba(255,255,255,0.1)',
              background: isDragging
                ? 'rgba(59,103,246,0.06)'
                : file
                  ? 'rgba(52,211,153,0.04)'
                  : 'rgba(12,17,30,0.6)',
              boxShadow: isDragging ? '0 0 20px rgba(59,103,246,0.1) inset' : 'none',
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="video/mp4,video/x-matroska,video/quicktime,video/webm"
              className="hidden"
            />
            <div className="flex flex-col items-center gap-3">
              <div className="h-12 w-12 rounded-2xl flex items-center justify-center transition-all"
                style={{
                  background: file ? 'rgba(52,211,153,0.12)' : 'rgba(255,255,255,0.04)',
                  border: file ? '1px solid rgba(52,211,153,0.25)' : '1px solid rgba(255,255,255,0.08)',
                }}>
                {file
                  ? <Film className="h-6 w-6 text-emerald-400" />
                  : <Upload className="h-6 w-6 text-slate-500" />
                }
              </div>
              {file ? (
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-white truncate max-w-xs">{file.name}</p>
                  <div className="flex items-center justify-center gap-2 text-xs text-slate-400">
                    <HardDrive className="h-3 w-3" />
                    <span>{fileSizeMB} MB</span>
                    <span className="text-slate-700">·</span>
                    <span className="text-emerald-400 font-medium">Ready to upload</span>
                  </div>
                </div>
              ) : (
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-slate-300">
                    {isDragging ? 'Drop it here!' : 'Click to browse or drag & drop'}
                  </p>
                  <p className="text-xs text-slate-500">MP4, MKV, MOV, WebM · Up to 500 MB</p>
                </div>
              )}
            </div>
          </div>

          {/* Pipeline note */}
          <div className="rounded-xl px-4 py-3 flex items-start gap-3"
            style={{ background: 'rgba(59,103,246,0.06)', border: '1px solid rgba(59,103,246,0.12)' }}>
            <div className="mt-0.5 h-4 w-4 shrink-0 text-nexus-400">
              <svg viewBox="0 0 16 16" fill="none" className="h-4 w-4">
                <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
                <path d="M8 5v3.5M8 11v.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              After upload, Nexus runs an 8-stage ML pipeline: FFmpeg audio extraction → Whisper transcription → ResNet-50 frame analysis → PyTorch GRU encoding → Spatiotemporal alignment → NumPy attention fusion → ChromaDB indexing.
            </p>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 p-3 rounded-xl text-xs text-red-400"
              style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-1">
            <button
              type="button"
              onClick={onClose}
              disabled={isUploading}
              className="px-4 py-2 rounded-xl text-sm text-slate-400 hover:text-white hover:bg-white/5 transition-all cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!file || isUploading}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed transition-all active:scale-95 cursor-pointer"
              style={{
                background: 'linear-gradient(135deg, #2547eb, #3b67f6)',
                boxShadow: file && !isUploading ? '0 0 20px rgba(59,103,246,0.35)' : 'none',
              }}
            >
              {isUploading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Uploading…</span>
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4" />
                  <span>Start Pipeline</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
