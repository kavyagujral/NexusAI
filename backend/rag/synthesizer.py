"""
Nexus Spatiotemporal RAG Engine - Topic Segmenter & Explanation Synthesizer
==========================================================================
Merges granular 2-second discretization windows into cohesive, human-interpretable
topic segments with clean start timestamps ("Starts at MM:SS") and synthesizes
structured, natural language explanations answering user questions from multimodal evidence.
"""

from typing import List, Dict, Any, Optional, Set
import re
from pydantic import BaseModel, Field


def format_timestamp(seconds: float) -> str:
    """Helper to convert float seconds into MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def deduplicate_sentences(text_list: List[str]) -> str:
    """
    Cleans and deduplicates overlapping spoken sentences and clauses across adjacent temporal windows.
    """
    seen: Set[str] = set()
    ordered_sentences: List[str] = []

    for raw in text_list:
        if not raw:
            continue
        parts = re.split(r'(?<=[.!?])\s+', raw.strip())
        for p in parts:
            p_clean = p.strip()
            if not p_clean:
                continue
            clauses = re.split(r'[,;]\s*', p_clean)
            unique_clauses = []
            for c in clauses:
                c_clean = c.strip()
                if len(c_clean) < 4:
                    continue
                key = re.sub(r'[^a-zA-Z0-9]', '', c_clean).lower()
                if key not in seen:
                    seen.add(key)
                    unique_clauses.append(c_clean)

            if unique_clauses:
                assembled = ", ".join(unique_clauses)
                if not assembled.endswith(('.', '!', '?')):
                    assembled += "."
                ordered_sentences.append(assembled)

    return " ".join(ordered_sentences)


class TopicSegment(BaseModel):
    """
    Unified topic-level grounded segment combining contiguous 2.0s windows.
    """
    video_id: str
    topic_start_seconds: float = Field(..., description="Time where the topic starts in seconds")
    topic_end_seconds: float = Field(..., description="Time where the topic segment concludes in seconds")
    topic_start_time: str = Field(..., description="e.g. '00:00'")
    display_timestamp: str = Field(..., description="e.g. 'Starts at 00:00'")
    explanation: str
    transcript: str
    visual_context: List[str]
    relevant_frame: str
    similarity_score: float


def group_windows_into_topics(
    raw_matches: List[Dict[str, Any]],
    proximity_seconds: float = 6.0
) -> List[TopicSegment]:
    """
    Merges granular 2-second windows from ChromaDB into cohesive topic segments.
    Nearby windows (within proximity_seconds) are combined into a single topic block.
    """
    if not raw_matches:
        return []

    # Sort matches chronologically by timestamp_start
    sorted_matches = sorted(raw_matches, key=lambda m: float(m.get("timestamp_start", 0.0)))

    clusters: List[List[Dict[str, Any]]] = []
    current_cluster: List[Dict[str, Any]] = [sorted_matches[0]]

    for match in sorted_matches[1:]:
        prev_end = float(current_cluster[-1].get("timestamp_end", 0.0))
        curr_start = float(match.get("timestamp_start", 0.0))

        # If window is contiguous or near the previous window, merge into current topic
        if curr_start <= prev_end + proximity_seconds:
            current_cluster.append(match)
        else:
            clusters.append(current_cluster)
            current_cluster = [match]

    if current_cluster:
        clusters.append(current_cluster)

    topic_segments: List[TopicSegment] = []

    for cluster in clusters:
        v_id = cluster[0].get("video_id", "unknown")
        t_start = float(min(m.get("timestamp_start", 0.0) for m in cluster))
        t_end = float(max(m.get("timestamp_end", t_start + 2.0) for m in cluster))

        # Deduplicate transcript across the cluster
        raw_transcripts = [m.get("transcript", "") for m in cluster]
        clean_text = deduplicate_sentences(raw_transcripts)
        if not clean_text:
            clean_text = cluster[0].get("transcript", "")

        # Collect unique visual labels
        all_labels: List[str] = []
        for m in cluster:
            for lbl in m.get("visual_labels", []):
                if lbl and lbl not in all_labels:
                    all_labels.append(lbl)
        if not all_labels:
            all_labels = ["video presentation"]

        # Select highest-similarity match for representative frame and score
        best_match = max(cluster, key=lambda m: float(m.get("similarity_score", 0.0)))
        frame_ref = best_match.get("frame_reference", cluster[0].get("frame_reference", ""))
        max_sim = float(best_match.get("similarity_score", 0.0))

        fmt_start = format_timestamp(t_start)

        topic_segments.append(
            TopicSegment(
                video_id=v_id,
                topic_start_seconds=round(t_start, 2),
                topic_end_seconds=round(t_end, 2),
                topic_start_time=fmt_start,
                display_timestamp=f"Starts at {fmt_start}",
                # The segment explanation is the cleaned transcript — the full synthesized
                # answer is produced separately by synthesize_comprehensive_explanation.
                explanation=clean_text,
                transcript=clean_text,
                visual_context=all_labels,
                relevant_frame=frame_ref,
                similarity_score=round(max_sim, 4)
            )
        )

    # Sort topic segments by similarity score descending (most relevant topic first)
    topic_segments.sort(key=lambda s: s.similarity_score, reverse=True)
    return topic_segments


def synthesize_comprehensive_explanation(
    query: str,
    topic_segments: List[TopicSegment]
) -> str:
    """
    Synthesizes a clear, structured natural language explanation answering the user's
    question based on the retrieved video segments and visual context.

    The explanation is built from the actual transcript content so it directly answers
    the query instead of just describing where the topic starts.
    """
    if not topic_segments:
        return f"No relevant content found in the video timeline for '{query}'."

    primary_segment = topic_segments[0]
    start_time = primary_segment.topic_start_time

    # ------------------------------------------------------------------ #
    # Collect all unique spoken sentences across retrieved topic segments  #
    # ------------------------------------------------------------------ #
    collected_sentences: List[str] = []
    for seg in topic_segments:
        parts = re.split(r'(?<=[.!?])\s+', seg.transcript.strip())
        for p in parts:
            p_clean = p.strip()
            if p_clean and p_clean not in collected_sentences and len(p_clean) > 8:
                collected_sentences.append(p_clean)

    # Collect visual context labels
    all_visuals: Set[str] = set()
    for seg in topic_segments:
        for v in seg.visual_context:
            all_visuals.add(v)
    visuals_str = ", ".join(sorted(all_visuals)) if all_visuals else "visual demonstrations"

    # ------------------------------------------------------------------ #
    # Build the answer: directly answer the question from transcript text  #
    # ------------------------------------------------------------------ #
    explanation_lines: List[str] = []
    explanation_lines.append(f"### Answer: *\"{query}\"*")
    explanation_lines.append("")

    if collected_sentences:
        # Primary explanation — first 1-3 sentences form the direct answer
        core = " ".join(collected_sentences[:3])
        explanation_lines.append(f"**Explanation:** {core}")
    else:
        explanation_lines.append(
            f"**Explanation:** This topic is discussed in the video starting at **{start_time}**."
        )

    # Additional detail from remaining sentences
    if len(collected_sentences) > 3:
        details = " ".join(collected_sentences[3:7])
        explanation_lines.append("")
        explanation_lines.append(f"**Further Details:** {details}")

    # Visual context grounding
    explanation_lines.append("")
    explanation_lines.append(
        f"**Visual Context:** The screen shows **{visuals_str}** aligned with the spoken content."
    )

    # Timestamp anchor — the single most important piece
    explanation_lines.append("")
    explanation_lines.append(f"📍 **This topic starts at {start_time}** in the video.")

    return "\n".join(explanation_lines)
