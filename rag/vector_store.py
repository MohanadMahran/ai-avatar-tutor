"""
FAISS vector store module for persistent document storage and retrieval.
Manages the vector index for semantic similarity search with
automatic persistence to disk.
"""
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss
from langchain.schema import Document
from utils.logger import get_logger
logger = get_logger(__name__)
class VectorStore:
    """FAISS-based vector store with disk persistence."""
    def __init__(self, store_path: str = "./vector_store") -> None:
        """
        Initialize the vector store.
        Args:
            store_path: Directory path for persisting the index and metadata.
        """
        self.store_path = Path(store_path)
        self.index_file = self.store_path / "index.faiss"
        self.metadata_file = self.store_path / "metadata.json"
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[dict] = []
        self._dimension: Optional[int] = None
        self._load_from_disk()
    def _load_from_disk(self) -> None:
        """Load the existing vector store from disk if available."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                self._dimension = self.index.d
                logger.info(
                    f"Loaded vector store from disk: {self.index.ntotal} vectors, "
                    f"dimension={self._dimension}"
                )
            except Exception as e:
                logger.error(f"Failed to load vector store from disk: {e}")
                self.index = None
                self.documents = []
        else:
            logger.info("No existing vector store found on disk.")
    def _save_to_disk(self) -> None:
        """Save the vector store to disk."""
        self.store_path.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Vector store saved to disk ({self.index.ntotal} vectors).")
    def add_documents(
        self, documents: List[Document], embeddings: np.ndarray
    ) -> None:
        """
        Add document chunks and their embeddings to the vector store.
        Args:
            documents: List of LangChain Document objects.
            embeddings: NumPy array of embeddings corresponding to documents.
        """
        if len(documents) == 0 or len(embeddings) == 0:
            logger.warning("No documents or embeddings to add.")
            return
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Document count ({len(documents)}) does not match "
                f"embedding count ({len(embeddings)})"
            )
        dimension = embeddings.shape[1]
        if self.index is None:
            self._dimension = dimension
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"Created new FAISS index with dimension {dimension}.")
        if dimension != self._dimension:
            raise ValueError(
                f"Embedding dimension ({dimension}) does not match "
                f"index dimension ({self._dimension})"
            )
        embeddings_float32 = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings_float32)
        self.index.add(embeddings_float32)
        for doc in documents:
            self.documents.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })
        logger.info(f"Added {len(documents)} documents to vector store. Total: {self.index.ntotal}")
        self._save_to_disk()
    def similarity_search(
        self, query_embedding: np.ndarray, top_k: int = 4
    ) -> List[Tuple[dict, float]]:
        """
        Search for the most similar documents to the query embedding.
        Args:
            query_embedding: Query embedding vector of shape (1, dimension).
            top_k: Number of top results to return.
        Returns:
            List of tuples (document_dict, similarity_score) sorted by relevance.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty. No results to return.")
            return []
        query_float32 = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_float32)
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_float32, actual_k)
        results: List[Tuple[dict, float]] = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        logger.debug(f"Similarity search returned {len(results)} results.")
        return results
    def get_document_count(self) -> int:
        """Get the total number of document chunks in the store."""
        if self.index is None:
            return 0
        return self.index.ntotal
    def get_sources(self) -> List[str]:
        """Get a list of unique source filenames in the store."""
        sources = set()
        for doc in self.documents:
            source = doc.get("metadata", {}).get("source", "unknown")
            sources.add(source)
        return sorted(list(sources))
    def delete_by_source(self, source_name: str) -> bool:
        """
        Delete all document chunks from a specific source.
        Note: FAISS IndexFlatIP doesn't support direct deletion,
        so we rebuild the index without the deleted documents.
        Args:
            source_name: The source filename to delete.
        Returns:
            True if any documents were deleted.
        """
        if self.index is None or self.index.ntotal == 0:
            return False
        original_count = len(self.documents)
        remaining_docs = []
        remaining_indices = []
        for i, doc in enumerate(self.documents):
            if doc.get("metadata", {}).get("source") != source_name:
                remaining_docs.append(doc)
                remaining_indices.append(i)
        if len(remaining_docs) == original_count:
            logger.info(f"No documents found with source: {source_name}")
            return False
        if remaining_indices:
            all_vectors = faiss.rev_swig_ptr(
                self.index.get_xb(), self.index.ntotal * self._dimension
            )
            all_vectors = np.array(all_vectors).reshape(self.index.ntotal, self._dimension)
            remaining_vectors = all_vectors[remaining_indices]
            new_index = faiss.IndexFlatIP(self._dimension)
            new_index.add(remaining_vectors.astype(np.float32))
            self.index = new_index
        else:
            self.index = faiss.IndexFlatIP(self._dimension)
        self.documents = remaining_docs
        self._save_to_disk()
        deleted_count = original_count - len(remaining_docs)
        logger.info(f"Deleted {deleted_count} chunks from source: {source_name}")
        return True
    def clear(self) -> None:
        """Clear the entire vector store."""
        if self._dimension:
            self.index = faiss.IndexFlatIP(self._dimension)
        else:
            self.index = None
        self.documents = []
        self._save_to_disk()
        logger.info("Vector store cleared.")