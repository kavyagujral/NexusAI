export type ProcessingStageStatus =
  | 'QUEUED'
  | 'EXTRACTING'
  | 'ANALYZING'
  | 'ALIGNING'
  | 'COMPLETED'
  | 'FAILED';

export interface VideoMetadata {
  video_id: string;
  filename: string;
  file_size_bytes: number;
  duration_seconds: number;
  created_at: string;
  status: ProcessingStageStatus;
}

export interface ProcessingJobStatus {
  job_id: string;
  video_id: string;
  status: ProcessingStageStatus;
  progress_percent: number;
  current_stage: string;
  created_at: string;
  updated_at: string;
  error_message?: string | null;
  frames_extracted: number;
  audio_segments_count: number;
  chunks_indexed: number;
}

export interface RetrievalResult {
  video_id: string;
  answer: string;
  timestamp_start: number;
  timestamp_end: number;
  topic_start_time?: string;
  formatted_timestamp: string;
  transcript: string;
  visual_context: string[];
  relevant_frame: string;
  similarity_score: number;
}

export interface QueryResponse {
  query: string;
  total_results: number;
  results: RetrievalResult[];
  synthesized_explanation?: string | null;
  topic_start_time?: string | null;
  topic_start_seconds?: number | null;
  message?: string;
}


