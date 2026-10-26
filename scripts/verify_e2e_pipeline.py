"""
Nexus Spatiotemporal RAG Engine - End-to-End Verification Harness
=================================================================
Executes the real production pipeline on data/videos/educational_lecture_5min.mp4:
1. Video Ingestion & Metadata Inspection
2. FFmpeg Audio Extraction
3. Whisper Speech-to-Text Transcription
4. 2-Second Temporal Frame Sampling (OpenCV)
5. Pretrained ResNet-50 CNN Feature Extraction & Categorization
6. Sequential PyTorch GRU Recurrent Audio Encoding
7. Spatiotemporal Stream Alignment (2.0s Windows & Overlap Weighting)
8. Pure NumPy Single-Head Cross-Modal Attention (Audio Query, Visual Key/Value)
9. 512-D Fused Vector Normalization
10. ChromaDB Persistent Collection Indexing
11. Natural Language Query Execution & Top-k Retrieval Verification
"""

import sys
import os
import time
import asyncio
from pathlib import Path
import numpy as np

# Ensure backend package in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.utils.logger import logger
from backend.audio.audio_processor import AudioProcessor
from backend.vision.frame_processor import FrameProcessor
from backend.vision.visual_encoder import VisualEncoder
from backend.models.gru_encoder import TemporalGRUEncoder
from backend.models.temporal_alignment import SpatiotemporalAligner2s
from backend.models.multimodal_attention import SpatiotemporalAligner, SpatiotemporalChunk
from backend.rag.vector_store import ChromaVectorStore
from backend.rag.retriever import SpatiotemporalRetriever


async def run_end_to_end_verification():
    print("=" * 80)
    print(" NEXUS SPATIOTEMPORAL RAG ENGINE — REAL END-TO-END PIPELINE VERIFICATION")
    print("=" * 80)

    video_path = Path("./data/videos/educational_lecture_5min.mp4")
    assert video_path.exists(), f"Video file not found at {video_path}"
    video_id = "test_lecture_5min"

    start_time = time.time()

    # -------------------------------------------------------------------------
    # STAGE 1: Audio Extraction (FFmpeg)
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] FFmpeg 16kHz Mono Audio Extraction...")
    audio_proc = AudioProcessor()
    print(f"  > FFmpeg executable: {audio_proc.ffmpeg_bin}")
    audio_wav = audio_proc.extract_audio_from_video(video_path, video_id)
    assert audio_wav.exists() and audio_wav.stat().st_size > 0
    print(f"  ✓ Audio extracted successfully: {audio_wav} ({audio_wav.stat().st_size / (1024*1024):.2f} MB)")

    # -------------------------------------------------------------------------
    # STAGE 2: Real Whisper Speech-to-Text Transcription
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 2] OpenAI Whisper Transcription (Model: {settings.WHISPER_MODEL})...")
    w_start = time.time()
    audio_segments = audio_proc.process_and_segment(audio_wav, video_id)
    w_dur = time.time() - w_start
    print(f"  ✓ Whisper produced {len(audio_segments)} timestamped transcript segments in {w_dur:.2f}s:")
    for idx, seg in enumerate(audio_segments):
        print(f"    - Segment {idx:02d} [{seg.start:05.1f}s - {seg.end:05.1f}s]: \"{seg.text}\"")

    # -------------------------------------------------------------------------
    # STAGE 3: 2-Second Frame Sampling (OpenCV)
    # -------------------------------------------------------------------------
    print("\n[STAGE 3] Video Frame Sampling at exact 2.0-second intervals...")
    frame_proc = FrameProcessor(sample_interval_seconds=2.0)
    frames = frame_proc.extract_frames(video_path, video_id)
    print(f"  ✓ Sampled {len(frames)} frames across {frames[-1].timestamp if frames else 0}s duration.")
    print(f"    - Frame 0: timestamp={frames[0].timestamp}s, path={frames[0].frame_path}")
    print(f"    - Frame 30: timestamp={frames[30].timestamp}s, path={frames[30].frame_path}")
    print(f"    - Frame 75: timestamp={frames[75].timestamp}s, path={frames[75].frame_path}")
    print(f"    - Frame 120: timestamp={frames[120].timestamp}s, path={frames[120].frame_path}")

    # -------------------------------------------------------------------------
    # STAGE 4: Real ResNet-50 CNN Visual Feature Extraction & Classification
    # -------------------------------------------------------------------------
    print("\n[STAGE 4] ResNet-50 CNN Visual Inference on All 2s Frames...")
    vis_encoder = VisualEncoder()
    frame_paths = [f.frame_path for f in frames]
    frame_timestamps = [f.timestamp for f in frames]
    
    cnn_start = time.time()
    visual_features = vis_encoder.batch_extract(frame_paths, frame_timestamps)
    cnn_dur = time.time() - cnn_start
    print(f"  ✓ ResNet-50 processed {len(visual_features)} frames in {cnn_dur:.2f}s (dim={len(visual_features[0].embedding)})")
    
    # Print sample detected classifications
    samples_idx = [0, 35, 80, 115, 140]
    for si in samples_idx:
        if si < len(visual_features):
            vf = visual_features[si]
            print(f"    - Frame @ {vf.timestamp:05.1f}s -> Predicted Class: '{vf.visual_label}' (conf: {vf.confidence:.2f})")

    # -------------------------------------------------------------------------
    # STAGE 5: PyTorch GRU Recurrent Sequence Audio Encoding
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] PyTorch GRU Recurrent Sequence Context Modeling...")
    gru_encoder = TemporalGRUEncoder()
    raw_embeds = [s.embedding for s in audio_segments]
    gru_out = gru_encoder.encode_sequence(raw_embeds)
    print(f"  ✓ PyTorch GRU processed sequence of shape ({len(raw_embeds)}, 512) -> Hidden Output: {gru_out.shape}")
    print(f"    - Vector L2-norm verification: {np.linalg.norm(gru_out[0]):.4f}")

    # -------------------------------------------------------------------------
    # STAGE 6: Spatiotemporal 2.0-Second Stream Alignment & Overlap Math
    # -------------------------------------------------------------------------
    print("\n[STAGE 6] Discretizing Streams into 2.0-Second Windows with Overlap Weighting...")
    aligner_2s = SpatiotemporalAligner2s(window_duration=2.0)
    aligned_windows = aligner_2s.align_streams(
        video_id=video_id,
        total_duration_seconds=300.0,
        visual_features=visual_features,
        audio_segments=audio_segments,
        gru_contextual_embeddings=gru_out
    )
    print(f"  ✓ Generated {len(aligned_windows)} synchronized 2.0s windows:")
    for w_idx in [0, 35, 80, 115, 145]:
        if w_idx < len(aligned_windows):
            w = aligned_windows[w_idx]
            print(f"    - Window {w.window_index:03d} [{w.timestamp_start:05.1f}s - {w.timestamp_end:05.1f}s]: visual='{w.visual_label}', transcript=\"{w.transcript[:55]}...\"")

    # -------------------------------------------------------------------------
    # STAGE 7: Pure NumPy Single-Head Attention Fusion (CLO01/CLO02)
    # -------------------------------------------------------------------------
    print("\n[STAGE 7] Pure NumPy Single-Head Cross-Modal Attention Fusion...")
    mm_aligner = SpatiotemporalAligner()
    chunks: List[SpatiotemporalChunk] = []

    fusion_start = time.time()
    for win in aligned_windows:
        fused_vec, attn_weights = mm_aligner.fuse_features(
            audio_embed=np.array(win.audio_feature, dtype=np.float32),
            visual_embed=np.array(win.visual_feature, dtype=np.float32),
            timestamp_seconds=win.timestamp_start
        )
        assert fused_vec.shape == (512,), f"Expected 512-D, got {fused_vec.shape}"
        assert np.isclose(np.linalg.norm(fused_vec), 1.0, atol=1e-5), "Fused vector must be L2 unit normalized"

        chunks.append(
            SpatiotemporalChunk(
                video_id=video_id,
                timestamp_start=win.timestamp_start,
                timestamp_end=win.timestamp_end,
                transcript=win.transcript,
                visual_label=win.visual_label,
                visual_labels=win.visual_labels,
                frame_reference=win.frame_path,
                audio_embedding=win.audio_feature,
                visual_embedding=win.visual_feature,
                multimodal_embedding=fused_vec.tolist()
            )
        )
    fusion_dur = time.time() - fusion_start
    print(f"  ✓ Generated {len(chunks)} 512-D fused multimodal embeddings in {fusion_dur:.3f}s")

    # -------------------------------------------------------------------------
    # STAGE 8: ChromaDB Persistent Storage & Indexing
    # -------------------------------------------------------------------------
    print("\n[STAGE 8] Indexing Multimodal Vectors into Persistent ChromaDB Collection...")
    chroma_store = ChromaVectorStore(persist_dir="./data/chromadb", collection_name="nexus_e2e_verification")
    indexed_count = chroma_store.insert_chunks(chunks)
    assert indexed_count == len(chunks), f"Expected {len(chunks)} indexed, got {indexed_count}"
    print(f"  ✓ Indexed {indexed_count} vectors into ChromaDB collection '{chroma_store.collection_name}' (Total collection count: {chroma_store.collection.count()})")

    # -------------------------------------------------------------------------
    # STAGE 9: Natural Language Spatiotemporal Grounding Retrieval Tests
    # -------------------------------------------------------------------------
    print("\n[STAGE 9] Natural Language Spatiotemporal Grounding Retrieval Queries...")
    retriever = SpatiotemporalRetriever(store=chroma_store)

    test_queries = [
        "When did the professor explain the microservice architecture diagram?",
        "Show me where the code snippet implementation with Redis cache is shown.",
        "When is the CAP theorem and partition tolerance discussed on the whiteboard?",
        "Where is the introduction to distributed cloud systems?"
    ]

    for q_idx, q_text in enumerate(test_queries, 1):
        print(f"\n  Query #{q_idx}: \"{q_text}\"")
        results = await retriever.query(query_text=q_text, video_id=video_id, top_k=3)
        for r_idx, res in enumerate(results, 1):
            print(f"    Match #{r_idx}:")
            print(f"      - Timestamp Window : {res.formatted_timestamp} ({res.timestamp_start:.1f}s - {res.timestamp_end:.1f}s)")
            print(f"      - Confidence Score : {res.similarity_score:.4f}")
            print(f"      - Visual Category  : {', '.join(res.visual_context)}")
            print(f"      - Spoken Transcript: \"{res.transcript}\"")
            print(f"      - Relevant Frame   : {res.relevant_frame}")

    total_pipeline_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" END-TO-END VERIFICATION COMPLETED IN {total_pipeline_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_end_to_end_verification())
