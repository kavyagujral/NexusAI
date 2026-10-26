"""
Nexus Spatiotemporal RAG Engine - ChromaDB Vector Store
======================================================
Implements persistent multimodal vector indexing and semantic retrieval using ChromaDB.
Each vector index entry represents a 2-second spatiotemporal window containing:
- Fused Multimodal Embedding (NumPy Cross-Modal Attention)
- Temporal boundaries (timestamp_start, timestamp_end)
- Spoken Audio Transcript
- CNN Visual Label & Confidence
- Synchronized Frame Reference
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.config import settings
from backend.utils.logger import logger
from backend.models.multimodal_attention import SpatiotemporalChunk


class ChromaVectorStore:
    """
    ChromaDB Client wrapper for persisting and querying multimodal spatiotemporal embeddings.
    """

    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = settings.resolve_path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        # Initialize ChromaDB Persistent Client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection with cosine similarity distance metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"Initialized ChromaDB Vector Store at '{self.persist_dir}' "
            f"(collection='{self.collection_name}', items_count={self.collection.count()})"
        )

    def insert_chunks(self, chunks: List[SpatiotemporalChunk]) -> int:
        """
        Inserts multimodal spatiotemporal chunks into the ChromaDB collection.

        Args:
            chunks: List of SpatiotemporalChunk instances containing 512-dim fused embeddings.

        Returns:
            Number of indexed vectors.
        """
        if not chunks:
            return 0

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{chunk.video_id}_win_{int(chunk.timestamp_start * 100):06d}_{idx}"
            
            # Use visual_labels joined as string for ChromaDB metadata (ChromaDB requires primitive metadata types)
            labels_str = ", ".join(chunk.visual_labels) if chunk.visual_labels else "general"

            ids.append(chunk_id)
            embeddings.append(chunk.multimodal_embedding or [0.0] * settings.EMBEDDING_DIM)
            documents.append(chunk.transcript if chunk.transcript else f"Visual context: {labels_str}")
            metadatas.append({
                "video_id": str(chunk.video_id),
                "timestamp_start": float(chunk.timestamp_start),
                "timestamp_end": float(chunk.timestamp_end),
                "transcript": str(chunk.transcript),
                "visual_label": str(labels_str),
                "frame_path": str(chunk.frame_reference)
            })

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"Successfully indexed {len(ids)} spatiotemporal vectors into ChromaDB (Total: {self.collection.count()})")
        return len(ids)

    def search_similar(
        self,
        query_embedding: List[float],
        video_id: Optional[str] = None,
        top_k: int = 3,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for top-k spatiotemporal segments closest to the query embedding.
        Filters out any matches that do not meet the minimum similarity threshold.

        Args:
            query_embedding: 512-dimensional query vector.
            video_id: Optional video ID filter.
            top_k: Number of nearest temporal chunks to retrieve.
            min_similarity: Minimum similarity threshold (default 0.0).

        Returns:
            List of matching chunk dictionaries with calculated similarity scores >= threshold.
        """
        if self.collection.count() == 0:
            logger.warning("ChromaDB collection is empty. Returning empty search results.")
            return []

        threshold = min_similarity

        where_filter = {"video_id": str(video_id)} if video_id else None

        actual_k = min(top_k, self.collection.count())
        if actual_k <= 0:
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                where=where_filter,
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query execution failed: {e}")
            return []

        matches: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return matches

        ids_list = results["ids"][0]
        metadatas_list = results["metadatas"][0] if results["metadatas"] else []
        distances_list = results["distances"][0] if results["distances"] else []

        for i in range(len(ids_list)):
            meta = metadatas_list[i] if i < len(metadatas_list) else {}
            dist = distances_list[i] if i < len(distances_list) else 0.0
            
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))

            # Strict relevance gate: only return results meeting the confidence threshold
            if similarity < threshold:
                logger.debug(
                    f"Chunk {ids_list[i]} similarity ({similarity:.4f}) below threshold ({threshold:.4f}). Filtered."
                )
                continue

            # Parse visual labels back into list
            raw_labels = meta.get("visual_label", "general")
            visual_labels = [lbl.strip() for lbl in raw_labels.split(",") if lbl.strip()]

            matches.append({
                "id": ids_list[i],
                "video_id": meta.get("video_id", "unknown"),
                "timestamp_start": float(meta.get("timestamp_start", 0.0)),
                "timestamp_end": float(meta.get("timestamp_end", 0.0)),
                "transcript": meta.get("transcript", ""),
                "visual_labels": visual_labels,
                "frame_reference": meta.get("frame_path", ""),
                "similarity_score": round(similarity, 4)
            })

        return matches


# Global singleton vector store
vector_store = ChromaVectorStore()
