"""RAG Package for Nexus with ChromaDB"""
from backend.rag.embeddings import QueryEmbedder
from backend.rag.vector_store import ChromaVectorStore, vector_store
from backend.rag.retriever import SpatiotemporalRetriever, RetrievalResult

# Alias for backwards compatibility
VectorStore = ChromaVectorStore

__all__ = [
    "QueryEmbedder",
    "ChromaVectorStore",
    "VectorStore",
    "vector_store",
    "SpatiotemporalRetriever",
    "RetrievalResult"
]
