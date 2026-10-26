import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { UploadModal } from './components/UploadModal';
import { ProcessingStatusCard } from './components/ProcessingStatusCard';
import { RAGQueryBar } from './components/RAGQueryBar';
import { RetrievalResultCard } from './components/RetrievalResultCard';
import { VideoPlayerWithTimestamp } from './components/VideoPlayerWithTimestamp';
import type { VideoMetadata, ProcessingJobStatus, RetrievalResult } from './types';
import { fetchVideos, fetchJobStatus, querySpatiotemporalRAG } from './services/api';
import {
  Sparkles,
  Cpu,
  Search,
  CheckCircle2,
  Video,
  BookOpen,
  FileQuestion,
  Play,
  Brain,
  Eye,
  Mic,
  Layers,
  Zap,
} from 'lucide-react';


export function App() {
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [activeJob, setActiveJob] = useState<ProcessingJobStatus | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string | undefined>(undefined);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [noResultsMessage, setNoResultsMessage] = useState<string | null>(null);
  const [synthesizedExplanation, setSynthesizedExplanation] = useState<string | null>(null);
  const [topicStartTime, setTopicStartTime] = useState<string | null>(null);
  const [topicStartSeconds, setTopicStartSeconds] = useState<number | null>(null);
  const [searchResults, setSearchResults] = useState<RetrievalResult[]>([]);
  const [seekTime, setSeekTime] = useState<number | null>(null);
  const [lastQuery, setLastQuery] = useState<string>('');

  const loadVideos = async () => {
    try {
      const list = await fetchVideos();
      setVideos(list);
      if (list.length > 0 && !selectedVideoId) {
        setSelectedVideoId(list[0].video_id);
      }
    } catch (err) {
      console.warn('Could not fetch video list:', err);
    }
  };

  useEffect(() => { loadVideos(); }, []);

  useEffect(() => {
    if (!activeJob || activeJob.status === 'COMPLETED' || activeJob.status === 'FAILED') return;
    const interval = setInterval(async () => {
      try {
        const updated = await fetchJobStatus(activeJob.job_id);
        setActiveJob(updated);
        if (updated.status === 'COMPLETED') loadVideos();
      } catch (err) {
        console.error('Failed to poll job status:', err);
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [activeJob]);

  const handleUploadSuccess = (videoId: string, jobId: string) => {
    setSelectedVideoId(videoId);
    setActiveJob({
      job_id: jobId,
      video_id: videoId,
      status: 'QUEUED',
      progress_percent: 5,
      current_stage: 'Video received. Pipeline starting...',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      frames_extracted: 0,
      audio_segments_count: 0,
      chunks_indexed: 0,
    });
  };

  const handleSearch = async (query: string, videoId?: string) => {
    setIsSearching(true);
    setLastQuery(query);
    setHasSearched(true);
    setSynthesizedExplanation(null);
    setTopicStartTime(null);
    setTopicStartSeconds(null);
    try {
      const res = await querySpatiotemporalRAG(query, videoId);
      setSearchResults(res.results);
      setNoResultsMessage(res.message || null);
      setSynthesizedExplanation(res.synthesized_explanation || null);
      setTopicStartTime(res.topic_start_time || null);
      setTopicStartSeconds(res.topic_start_seconds ?? null);
    } catch (err: any) {
      alert(err.message || 'Search failed');
    } finally {
      setIsSearching(false);
    }
  };

  const selectedVideo = videos.find((v) => v.video_id === selectedVideoId);

  // Strip markdown markers from the synthesized explanation for plain display
  const cleanExplanation = synthesizedExplanation
    ? synthesizedExplanation
      .replace(/^###\s*/gm, '')
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
    : null;

  return (
    <div className="min-h-screen bg-surface-950 text-slate-100 flex flex-col">
      <Navbar
        onOpenUpload={() => setIsUploadOpen(true)}
        activeVideoCount={videos.length}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

        {/* ── Hero ─────────────────────────────────────────────── */}
        <section className="relative overflow-hidden rounded-3xl p-8 sm:p-12"
          style={{
            background: 'linear-gradient(135deg, rgba(10,14,26,0.97) 0%, rgba(14,20,40,0.97) 100%)',
            border: '1px solid rgba(255,255,255,0.06)',
            boxShadow: '0 8px 60px rgba(0,0,0,0.4)',
          }}
        >
          {/* Background orbs */}
          <div className="orb orb-1" />
          <div className="orb orb-2" />
          <div className="orb orb-3" />

          {/* Grid pattern overlay */}
          <div className="absolute inset-0 opacity-[0.02]"
            style={{
              backgroundImage: `linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px),
                                linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)`,
              backgroundSize: '40px 40px',
            }}
          />

          <div className="relative z-10 max-w-3xl space-y-6">
            {/* Tag */}
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold"
              style={{
                background: 'rgba(59, 103, 246, 0.12)',
                border: '1px solid rgba(59, 103, 246, 0.25)',
                color: '#93b2fd',
              }}
            >
              <Sparkles className="h-3 w-3" />
              Spatiotemporal Multimodal AI Engine
            </div>

            {/* Headline */}
            <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-[1.1]">
              Nexus connects{' '}
              <span className="gradient-text-nexus">what was said</span>,{' '}
              <span style={{
                background: 'linear-gradient(135deg, #818cf8, #67e8f9)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>what was shown</span>,
              <br />
              and{' '}
              <span style={{
                background: 'linear-gradient(135deg, #67e8f9, #34d399)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>when it happened</span>.
            </h1>

            <p className="text-sm sm:text-base text-slate-400 leading-relaxed max-w-2xl">
              Upload educational lecture recordings or Zoom screen shares. Nexus samples frames,
              runs ResNet-50 CNN vision classification, recurrent GRU temporal audio modeling,
              and NumPy cross-modal attention to index every moment for instant Q&A.
            </p>

            {/* Feature pills */}
            <div className="flex flex-wrap gap-2 pt-1">
              {[
                { icon: Brain, label: 'NumPy Cross-Modal Attention', color: '#93b2fd', bg: 'rgba(59,103,246,0.1)', border: 'rgba(59,103,246,0.2)' },
                { icon: Eye, label: 'ResNet-50 Vision (CLO04)', color: '#67e8f9', bg: 'rgba(6,182,212,0.08)', border: 'rgba(6,182,212,0.18)' },
                { icon: Mic, label: 'Whisper + GRU Audio (CLO05)', color: '#a78bfa', bg: 'rgba(139,92,246,0.08)', border: 'rgba(139,92,246,0.18)' },
                { icon: Layers, label: 'ChromaDB 512-dim', color: '#fbbf24', bg: 'rgba(251,191,36,0.07)', border: 'rgba(251,191,36,0.15)' },
              ].map(({ icon: Icon, label, color, bg, border }) => (
                <div key={label}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium"
                  style={{ background: bg, border: `1px solid ${border}`, color }}
                >
                  <Icon className="h-3 w-3" />
                  {label}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Processing pipeline status ────────────────────────── */}
        {activeJob && (
          <section>
            <ProcessingStatusCard status={activeJob} />
          </section>
        )}

        {/* ── Query bar ─────────────────────────────────────────── */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <Search className="h-4 w-4 text-nexus-400" />
            <h2 className="text-base font-bold text-white">Query Video Grounding</h2>
            {hasSearched && lastQuery && (
              <span className="ml-auto text-xs text-slate-500 truncate max-w-xs hidden sm:block">
                "{lastQuery}"
              </span>
            )}
          </div>
          <RAGQueryBar
            onSearch={handleSearch}
            isLoading={isSearching}
            videos={videos}
            selectedVideoId={selectedVideoId}
            onSelectVideo={setSelectedVideoId}
          />
        </section>

        {/* ── Main content grid ─────────────────────────────────── */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left: Results */}
          <div className="lg:col-span-7 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-400 flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-nexus-400" />
                Grounding Results
                {searchResults.length > 0 && (
                  <span className="px-1.5 py-0.5 rounded-md text-[11px] font-bold text-nexus-300"
                    style={{ background: 'rgba(59, 103, 246, 0.15)' }}>
                    {searchResults.length}
                  </span>
                )}
              </h3>
            </div>

            {/* ── Synthesized AI Answer ── */}
            {cleanExplanation && (
              <div className="answer-panel rounded-2xl p-5 space-y-4">
                {/* Header row */}
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <div className="h-7 w-7 rounded-lg flex items-center justify-center shrink-0"
                      style={{ background: 'rgba(59, 103, 246, 0.15)', border: '1px solid rgba(59, 103, 246, 0.25)' }}>
                      <Sparkles className="h-3.5 w-3.5 text-nexus-400" />
                    </div>
                    <div>
                      <p className="text-xs font-bold text-nexus-300 uppercase tracking-wider leading-none">AI Answer</p>
                      <p className="text-[10px] text-slate-500 leading-none mt-0.5">Grounded in video content only</p>
                    </div>
                  </div>
                  {topicStartSeconds !== null && (
                    <button
                      onClick={() => setSeekTime(topicStartSeconds)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white transition-all active:scale-95 cursor-pointer shrink-0"
                      style={{
                        background: 'linear-gradient(135deg, #2547eb, #3b67f6)',
                        boxShadow: '0 0 15px rgba(59, 103, 246, 0.35)',
                      }}
                    >
                      <Play className="h-3 w-3 fill-current" />
                      Jump to {topicStartTime}
                    </button>
                  )}
                </div>

                {/* Divider */}
                <div className="h-px" style={{ background: 'rgba(59, 103, 246, 0.15)' }} />

                {/* Answer text — parse sections */}
                <div className="space-y-3">
                  {cleanExplanation.split('\n').filter(Boolean).map((line, i) => {
                    if (line.startsWith('Answer:')) {
                      return null; // skip header line
                    }
                    if (line.startsWith('Explanation:')) {
                      return (
                        <p key={i} className="text-sm text-slate-200 leading-relaxed">
                          <span className="text-nexus-400 font-semibold">Explanation: </span>
                          {line.replace('Explanation:', '').trim()}
                        </p>
                      );
                    }
                    if (line.startsWith('Further Details:')) {
                      return (
                        <p key={i} className="text-sm text-slate-300 leading-relaxed">
                          <span className="text-indigo-400 font-semibold">Further Details: </span>
                          {line.replace('Further Details:', '').trim()}
                        </p>
                      );
                    }
                    if (line.startsWith('Visual Context:')) {
                      return (
                        <p key={i} className="text-xs text-slate-500 leading-relaxed">
                          <Eye className="inline h-3 w-3 mr-1 text-cyan-500" />
                          {line.replace('Visual Context:', '').trim()}
                        </p>
                      );
                    }
                    if (line.startsWith('📍')) {
                      return (
                        <div key={i}
                          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold"
                          style={{
                            background: 'rgba(59, 103, 246, 0.12)',
                            border: '1px solid rgba(59, 103, 246, 0.2)',
                            color: '#93b2fd',
                          }}
                        >
                          <Zap className="h-3 w-3" />
                          {line.replace('📍', '').trim()}
                        </div>
                      );
                    }
                    return null;
                  })}
                </div>
              </div>
            )}

            {/* ── Evidence segments ── */}
            {searchResults.length > 0 ? (
              <div className="space-y-3">
                <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest">
                  Video Evidence Segments
                </p>
                {searchResults.map((result, idx) => (
                  <div key={idx} style={{ animationDelay: `${idx * 80}ms` }}>
                    <RetrievalResultCard
                      result={result}
                      onJumpToTimestamp={(ts) => setSeekTime(ts)}
                    />
                  </div>
                ))}
              </div>
            ) : hasSearched ? (
              /* No results */
              <div className="rounded-2xl p-8 text-center space-y-4"
                style={{
                  background: 'rgba(120, 53, 15, 0.08)',
                  border: '1px solid rgba(217, 119, 6, 0.2)',
                }}>
                <div className="h-12 w-12 rounded-2xl mx-auto flex items-center justify-center"
                  style={{ background: 'rgba(217, 119, 6, 0.1)', border: '1px solid rgba(217, 119, 6, 0.2)' }}>
                  <FileQuestion className="h-6 w-6 text-amber-400" />
                </div>
                <div className="space-y-2">
                  <h4 className="text-base font-semibold text-white">No Linked Video Moments Found</h4>
                  <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
                    {noResultsMessage || `No moments in this video match "${lastQuery}". This question doesn't appear to be covered in the video content.`}
                  </p>
                </div>
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-[11px] text-slate-400"
                  style={{ background: 'rgba(6, 9, 14, 0.8)', border: '1px solid rgba(255,255,255,0.07)' }}>
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  Nexus only answers questions grounded in video content
                </div>
              </div>
            ) : (
              /* Empty state */
              <div className="rounded-2xl p-12 text-center space-y-4"
                style={{
                  background: 'rgba(12, 17, 30, 0.5)',
                  border: '1px solid rgba(255,255,255,0.05)',
                }}>
                <div className="h-16 w-16 rounded-2xl mx-auto flex items-center justify-center"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <Search className="h-7 w-7 text-slate-600" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-slate-400">No query yet</p>
                  <p className="text-xs text-slate-600 max-w-sm mx-auto">
                    Type a question or pick a preset chip above to ground your question against the video timeline.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Right: Video player + Architecture */}
          <div className="lg:col-span-5 space-y-5">
            {selectedVideo ? (
              <VideoPlayerWithTimestamp
                video={selectedVideo}
                seekToTimestamp={seekTime}
              />
            ) : (
              <div className="rounded-2xl p-10 text-center space-y-3"
                style={{
                  background: 'rgba(12, 17, 30, 0.6)',
                  border: '1px solid rgba(255,255,255,0.05)',
                }}>
                <Video className="h-8 w-8 text-slate-700 mx-auto" />
                <p className="text-xs text-slate-500">
                  Upload a video to preview synchronized playback with spatiotemporal timestamps.
                </p>
              </div>
            )}

            {/* Architecture card */}
            <div className="rounded-2xl p-5 space-y-4"
              style={{
                background: 'rgba(10, 14, 24, 0.8)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-emerald-400" />
                <h4 className="text-xs font-bold text-white uppercase tracking-wider">Phase 2 ML Architecture</h4>
              </div>
              <div className="space-y-2.5">
                {[
                  {
                    icon: Brain,
                    color: 'text-nexus-400',
                    bg: 'rgba(59,103,246,0.08)',
                    border: 'rgba(59,103,246,0.15)',
                    label: 'CLO01/CLO02',
                    desc: 'Pure NumPy Single-Head Cross-Modal Attention (Q=AW_Q, K=VW_K, V=VW_V) + LayerNorm',
                  },
                  {
                    icon: Eye,
                    color: 'text-cyan-400',
                    bg: 'rgba(6,182,212,0.07)',
                    border: 'rgba(6,182,212,0.15)',
                    label: 'CLO04 Vision',
                    desc: 'ResNet-50 CNN on 2.0s frames — slide, architecture, code, whiteboard, talking-head',
                  },
                  {
                    icon: Mic,
                    color: 'text-violet-400',
                    bg: 'rgba(139,92,246,0.07)',
                    border: 'rgba(139,92,246,0.15)',
                    label: 'CLO05 Audio',
                    desc: 'OpenAI Whisper + PyTorch GRU contextual sequence modeling',
                  },
                  {
                    icon: Layers,
                    color: 'text-amber-400',
                    bg: 'rgba(251,191,36,0.06)',
                    border: 'rgba(251,191,36,0.12)',
                    label: 'Vector Store',
                    desc: 'ChromaDB persistent 512-dim cosine spatiotemporal indexing',
                  },
                ].map(({ icon: Icon, color, bg, border, label, desc }) => (
                  <div key={label} className="flex items-start gap-3 p-2.5 rounded-xl transition-all duration-200 hover:bg-white/[0.02]"
                    style={{ border: '1px solid transparent' }}
                    onMouseEnter={e => (e.currentTarget.style.border = `1px solid ${border}`)}
                    onMouseLeave={e => (e.currentTarget.style.border = '1px solid transparent')}
                  >
                    <div className={`shrink-0 h-6 w-6 rounded-md flex items-center justify-center ${color}`}
                      style={{ background: bg }}>
                      <Icon className="h-3.5 w-3.5" />
                    </div>
                    <div>
                      <span className={`text-[11px] font-bold ${color}`}>{label}: </span>
                      <span className="text-[11px] text-slate-400">{desc}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>

      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadSuccess={handleUploadSuccess}
      />
    </div>
  );
}

export default App;
