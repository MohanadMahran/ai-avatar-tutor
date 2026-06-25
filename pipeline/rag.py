"""
RAG pipeline integration module.
Provides a high-level interface for the RAG system, combining
document loading, embedding, and retrieval into a single callable.
"""
from pathlib import Path
from typing import List, Optional
from config.settings import get_settings
from rag.document_loader import DocumentLoader
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from utils.logger import get_logger
logger = get_logger(__name__)
class RAGPipeline:
    """High-level RAG pipeline for document retrieval."""
    def __init__(self) -> None:
        """Initialize the RAG pipeline components."""
        self.settings = get_settings()
        self.embedder = Embedder(model_name=self.settings.embedding_model)
        self.vector_store = VectorStore(store_path=self.settings.vector_store_path)
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.settings.top_k_results,
            relevance_threshold=self.settings.relevance_threshold,
        )
        self.document_loader = DocumentLoader(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
    def retrieve_context(self, query: str) -> dict:
        """
        Retrieve relevant context from the vector store for a given query.
        Args:
            query: The user's question or search query.
        Returns:
            Dictionary containing:
                - context: Formatted context string
                - sources: List of source documents
                - confidence: Average relevance score
        """
        logger.info(f"RAG retrieval for query: '{query[:80]}...'")
        try:
            results = self.retriever.retrieve(query)
            return results
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return {
                "context": "",
                "sources": [],
                "confidence": 0.0,
            }
    def index_documents(self, directory_path: Optional[str] = None) -> int:
        """
        Load and index all documents from the specified directory.
        Args:
            directory_path: Path to directory containing documents.
                          Uses default docs path if not specified.
        Returns:
            Number of document chunks indexed.
        """
        if directory_path is None:
            directory_path = self.settings.docs_path
        path = Path(directory_path)
        if not path.exists():
            logger.warning(f"Documents directory does not exist: {directory_path}")
            return 0
        logger.info(f"Loading documents from {directory_path}...")
        documents = self.document_loader.load_directory(str(path))
        if not documents:
            logger.info("No documents found to index.")
            return 0
        logger.info(f"Embedding {len(documents)} document chunks...")
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_texts(texts)
        logger.info("Adding documents to vector store...")
        self.vector_store.add_documents(documents, embeddings)
        logger.info(f"Successfully indexed {len(documents)} document chunks.")
        return len(documents)
    def index_single_file(self, file_path: str) -> int:
        """
        Load and index a single document file.
        Args:
            file_path: Path to the document file.
        Returns:
            Number of document chunks indexed from this file.
        """
        logger.info(f"Indexing single file: {file_path}")
        documents = self.document_loader.load_file(file_path)
        if not documents:
            logger.warning(f"No content extracted from {file_path}")
            return 0
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_documents(documents, embeddings)
        logger.info(f"Indexed {len(documents)} chunks from {file_path}")
        return len(documents)
    def get_document_count(self) -> int:
        """Get the total number of indexed document chunks."""
        return self.vector_store.get_document_count()
    def get_indexed_sources(self) -> List[str]:
        """Get list of all indexed source filenames."""
        return self.vector_store.get_sources()
    def clear_index(self) -> None:
        """Clear the entire vector store index."""
        self.vector_store.clear()
        logger.info("Vector store cleared.")
    def delete_source(self, source_name: str) -> bool:
        """
        Delete all chunks from a specific source document.
        Args:
            source_name: The filename of the source to delete.
        Returns:
            True if deletion was successful.
        """
        return self.vector_store.delete_by_source(source_name)