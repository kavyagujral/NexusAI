import React, { useRef, useEffect, useState } from 'react';
import type { VideoMetadata } from '../types';
import { Film, Clock, ExternalLink } from 'lucide-react';

interface VideoPlayerProps {
  video: VideoMetadata;
  seekToTimestamp?: number | null;
}

function formatDuration(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export const VideoPlayerWithTimestamp: React.FC<VideoPlayerProps> = ({
  video,
  seekToTimestamp,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
  const videoSrc = `${API_BASE_URL}/static/data/videos/${video.video_id}.mp4`;
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (seekToTimestamp !== null && seekToTimestamp !== undefined && videoRef.current) {
      videoRef.current.currentTime = seekToTimestamp;
      videoRef.current.play().catch(() => { });
    }
  }, [seekToTimestamp]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const progress = video.duration_seconds > 0
    ? (currentTime / video.duration_seconds) * 100
    : 0;

  return (
    <div
      className="rounded-2xl overflow-hidden border border-white/[0.07] animate-slide-up"
      style={{
        background: 'rgba(10, 14, 24, 0.95)',
        boxShadow: '0 8px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05)',
      }}
    >
      {/* Header */}
      <div className="px-4 pt-4 pb-3 flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="shrink-0 h-8 w-8 rounded-lg flex items-center justify-center"
            style={{ background: 'rgba(59, 103, 246, 0.12)', border: '1px solid rgba(59, 103, 246, 0.2)' }}>
            <Film className="h-4 w-4 text-nexus-400" />
          </div>
          <div className="min-w-0">
            <h4 className="text-sm font-semibold text-white truncate leading-tight">
              {video.filename}
            </h4>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-[10px] text-slate-500 font-mono">{video.video_id}</span>
              {video.duration_seconds > 0 && (
                <>
                  <span className="text-slate-700">·</span>
                  <span className="text-[10px] text-slate-500 flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatDuration(video.duration_seconds)}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
        <a
          href={videoSrc}
          target="_blank"
          rel="noreferrer"
          className="shrink-0 p-1.5 rounded-lg text-slate-600 hover:text-slate-300 hover:bg-white/5 transition-all"
          title="Open in new tab"
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      </div>

      {/* Video */}
      <div className="relative bg-black" style={{ aspectRatio: '16/9' }}>
        <video
          ref={videoRef}
          controls
          onTimeUpdate={handleTimeUpdate}
          className="w-full h-full object-contain"
          src={videoSrc}
        >
          Your browser does not support HTML5 video.
        </video>

        {/* Seek indicator overlay — shows briefly when seeked */}
        {seekToTimestamp !== null && seekToTimestamp !== undefined && seekToTimestamp > 0 && (
          <div
            key={seekToTimestamp}
            className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-mono font-semibold text-white pointer-events-none animate-fade-in"
            style={{
              background: 'rgba(59, 103, 246, 0.85)',
              backdropFilter: 'blur(8px)',
              boxShadow: '0 0 12px rgba(59, 103, 246, 0.5)',
              animation: 'fade-in 0.2s ease both, fade-out 0.3s ease 2s both',
            }}
          >
            <Clock className="h-3 w-3" />
            ↳ {formatDuration(seekToTimestamp)}
          </div>
        )}
      </div>

      {/* Progress bar */}
      <div className="h-0.5 bg-slate-900 relative">
        <div
          className="h-full transition-all duration-300"
          style={{
            width: `${progress}%`,
            background: 'linear-gradient(90deg, #3b67f6, #818cf8)',
          }}
        />
      </div>

      {/* Footer */}
      <div className="px-4 py-3 flex items-center justify-between">
        <span className="text-[10px] text-slate-600 font-medium uppercase tracking-wider">Live Preview</span>
        <span className="text-[10px] text-slate-600 font-mono">
          {formatDuration(currentTime)} / {video.duration_seconds > 0 ? formatDuration(video.duration_seconds) : '--:--'}
        </span>
      </div>
    </div>
  );
};
