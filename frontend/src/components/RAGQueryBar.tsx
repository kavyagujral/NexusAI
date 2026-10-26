import React, { useState, useRef } from 'react';
import { Search, Sparkles, Loader2, ArrowRight, ChevronDown } from 'lucide-react';
import type { VideoMetadata } from '../types';

interface RAGQueryBarProps {
  onSearch: (query: string, videoId?: string) => void;
  isLoading: boolean;
  videos: VideoMetadata[];
  selectedVideoId?: string;
  onSelectVideo: (videoId?: string) => void;
}

const DEFAULT_PRESET_QUERIES = [
  'When did the professor explain the microservice architecture?',
  'Where is the database consistency model discussed?',
  'Show me when the API gateway diagram was shown.',
];

const BINARY_SEARCH_PRESETS = [
  'What is binary search and how does it divide the search area?',
  'Show me when the code snippet is displayed.',
  'Explain how the middle element is selected.',
];

export const RAGQueryBar: React.FC<RAGQueryBarProps> = ({
  onSearch,
  isLoading,
  videos,
  selectedVideoId,
  onSelectVideo,
}) => {
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const currentVideo = videos.find((v) => v.video_id === selectedVideoId);
  const isBinarySearch = currentVideo?.filename.toLowerCase().includes('binary search');
  const activePresets = isBinarySearch ? BINARY_SEARCH_PRESETS : DEFAULT_PRESET_QUERIES;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim(), selectedVideoId);
    }
  };

  const handleChipClick = (preset: string) => {
    setQuery(preset);
    onSearch(preset, selectedVideoId);
  };

  return (
    <div className="w-full space-y-4">
      <form onSubmit={handleSubmit}>
        {/* Main search container */}
        <div
          className="relative rounded-2xl transition-all duration-300"
          style={{
            background: isFocused
              ? 'rgba(12, 17, 30, 0.95)'
              : 'rgba(12, 17, 30, 0.8)',
            border: isFocused
              ? '1px solid rgba(59, 103, 246, 0.5)'
              : '1px solid rgba(255, 255, 255, 0.08)',
            boxShadow: isFocused
              ? '0 0 0 4px rgba(59, 103, 246, 0.08), 0 0 30px -5px rgba(59, 103, 246, 0.25), 0 8px 32px rgba(0,0,0,0.3)'
              : '0 4px 24px rgba(0,0,0,0.2)',
            backdropFilter: 'blur(20px)',
          }}
        >
          <div className="flex items-center gap-2 p-2">
            {/* Video selector */}
            {videos.length > 0 && (
              <div className="relative shrink-0">
                <select
                  value={selectedVideoId || ''}
                  onChange={(e) => onSelectVideo(e.target.value || undefined)}
                  className="appearance-none pl-3 pr-8 py-2.5 rounded-xl text-xs font-medium text-slate-300 border border-white/[0.07] focus:outline-none focus:border-nexus-500/50 cursor-pointer truncate max-w-[180px]"
                  style={{ background: 'rgba(6, 9, 14, 0.8)' }}
                >
                  <option value="">All Videos</option>
                  {videos.map((v) => (
                    <option key={v.video_id} value={v.video_id}>
                      {v.filename.length > 28 ? v.filename.slice(0, 28) + '…' : v.filename}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-3 w-3 text-slate-500 pointer-events-none" />
              </div>
            )}

            {/* Divider */}
            {videos.length > 0 && <div className="h-6 w-px bg-white/10 shrink-0" />}

            {/* Search icon */}
            <Search className={`h-4 w-4 shrink-0 ml-1 transition-colors duration-200 ${isFocused ? 'text-nexus-400' : 'text-slate-500'}`} />

            {/* Input */}
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Ask anything about the video: 'When did the lecturer explain binary search?'"
              className="flex-1 bg-transparent py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none min-w-0"
            />

            {/* Submit button */}
            <button
              type="submit"
              disabled={!query.trim() || isLoading}
              className="shrink-0 flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 active:scale-95 cursor-pointer"
              style={{
                background: isLoading || !query.trim()
                  ? 'rgba(37, 71, 235, 0.5)'
                  : 'linear-gradient(135deg, #2547eb, #3b67f6)',
                boxShadow: query.trim() && !isLoading
                  ? '0 0 20px rgba(59, 103, 246, 0.4)'
                  : 'none',
              }}
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Searching…</span>
                </>
              ) : (
                <>
                  <span>Search</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </>
              )}
            </button>
          </div>

          {/* Loading bar */}
          {isLoading && (
            <div className="absolute bottom-0 left-0 right-0 h-[2px] rounded-b-2xl overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  background: 'linear-gradient(90deg, transparent, #3b67f6, #818cf8, #3b67f6, transparent)',
                  backgroundSize: '200% 100%',
                  animation: 'skeleton-shimmer 1.5s linear infinite',
                }}
              />
            </div>
          )}
        </div>
      </form>

      {/* Preset chips */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-slate-500 mr-1">
          <Sparkles className="h-3 w-3 text-nexus-400" />
          <span>Try:</span>
        </div>
        {activePresets.map((preset, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => handleChipClick(preset)}
            disabled={isLoading}
            className="preset-chip text-[11px] px-3 py-1.5 rounded-xl border border-white/[0.07] text-slate-400 hover:text-white hover:border-nexus-500/40 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            style={{ background: 'rgba(12, 17, 30, 0.8)' }}
          >
            {preset.length > 55 ? preset.slice(0, 55) + '…' : preset}
          </button>
        ))}
      </div>
    </div>
  );
};
