import type { VideoMetadata, ProcessingJobStatus, QueryResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function uploadVideo(
  file: File,
  _onProgress?: (percent: number) => void
): Promise<{ video_id: string; job_id: string; filename: string }> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/videos/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(errData.detail || `Upload failed with status ${response.status}`);
  }

  return response.json();
}

export async function fetchJobStatus(jobId: string): Promise<ProcessingJobStatus> {
  const response = await fetch(`${API_BASE_URL}/api/processing/${jobId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch job status: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchVideoStatus(videoId: string): Promise<ProcessingJobStatus> {
  const response = await fetch(`${API_BASE_URL}/api/processing/video/${videoId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch video status: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchVideos(): Promise<VideoMetadata[]> {
  const response = await fetch(`${API_BASE_URL}/api/videos/`);
  if (!response.ok) {
    throw new Error(`Failed to list videos: ${response.statusText}`);
  }
  return response.json();
}

export async function querySpatiotemporalRAG(
  query: string,
  videoId?: string
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/query/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, video_id: videoId || null, top_k: 3 }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ detail: 'Query failed' }));
    throw new Error(errData.detail || `Query failed: ${response.statusText}`);
  }

  return response.json();
}
