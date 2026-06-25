"""
Retriever module combining embedder and vector store for semantic search.
Provides a high-level interface for querying the document collection
with relevance filtering and formatted context output.
"""
from typing import List, Dict
import numpy as np
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from utils.logger import get_logger
logger = get_logger(__name__)
class Retriever:
    """Semantic retriever combining embedding and vector search."""
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 4,
        relevance_threshold: float = 0.3,
    ) -> None:
        """
        Initialize the retriever.
        Args:
            embedder: Embedder instance for query embedding.
            vector_store: VectorStore instance for similarity search.
            top_k: Number of top results to retrieve.
            relevance_threshold: Minimum similarity score to include a result.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
    def retrieve(self, query: str) -> Dict:
        """
        Retrieve relevant document chunks for a query.
        Args:
            query: The user's search query.
        Returns:
            Dictionary containing:
                - context: Formatted context string with sources
                - sources: List of source filenames
                - confidence: Average relevance score of top results
                - chunks: List of individual chunk dictionaries
        """
        if not query.strip():
            logger.warning("Empty query provided to retriever.")
            return {
                "context": "",
                "sources": [],
                "confidence": 0.0,
                "chunks": [],
            }
        # Embed the query
        query_embedding = self.embedder.embed_query(query)
        if query_embedding.size == 0:
            logger.error("Failed to embed query.")
            return {
                "context": "",
                "sources": [],
                "confidence": 0.0,
                "chunks": [],
            }
        # Search the vector store
        results = self.vector_store.similarity_search(query_embedding, top_k=self.top_k)
        if not results:
            logger.info("No results found in vector store.")
            return {
                "context": "",
                "sources": [],
                "confidence": 0.0,
                "chunks": [],
            }
        # Filter by relevance threshold
        filtered_results = [
            (doc, score) for doc, score in results if score >= self.relevance_threshold
        ]
        if not filtered_results:
            logger.info(
                f"All results below relevance threshold ({self.relevance_threshold}). "
                f"Best score: {results[0][1]:.4f}"
            )
            return {
                "context": "",
                "sources": [],
                "confidence": float(results[0][1]) if results else 0.0,
                "chunks": [],
            }
        # Build context string
        context_parts: List[str] = []
        sources: List[str] = []
        chunks: List[Dict] = []
        scores: List[float] = []
        for i, (doc, score) in enumerate(filtered_results):
            source = doc.get("metadata", {}).get("source", "unknown")
            page = doc.get("metadata", {}).get("page", "?")
            content = doc.get("page_content", "")
            context_parts.append(
                f"[Source: {source}, Page: {page}, Relevance: {score:.2f}]\n{content}"
            )
            if source not in sources:
                sources.append(source)
            chunks.append({
                "content": content,
                "source": source,
                "page": page,
                "score": round(float(score), 4),
            })
            scores.append(float(score))
        context = "\n\n---\n\n".join(context_parts)
        avg_confidence = sum(scores) / len(scores) if scores else 0.0
        logger.info(
            f"Retrieved {len(filtered_results)} relevant chunks "
            f"(avg confidence: {avg_confidence:.3f}) from sources: {sources}"
        )
        return {
            "context": context,
            "sources": sources,
            "confidence": round(avg_confidence, 4),
            "chunks": chunks,
        }