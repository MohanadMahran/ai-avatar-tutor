"""
Tests for the RAG system components.
Tests document loading, embedding, vector store operations,
and retrieval with sample text data.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from langchain.schema import Document
from rag.document_loader import DocumentLoader
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)
@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    file_path = Path(temp_dir) / "sample.txt"
    content = """Machine learning is a subset of artificial intelligence.
It involves training algorithms on data to make predictions.
Deep learning uses neural networks with multiple layers.
Natural language processing deals with understanding human language.
Computer vision focuses on image and video analysis.
Reinforcement learning involves learning through trial and error."""
    file_path.write_text(content)
    return str(file_path)
@pytest.fixture
def document_loader():
    """Create a document loader instance."""
    return DocumentLoader(chunk_size=100, chunk_overlap=20)
@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns fixed-size vectors."""
    embedder = MagicMock(spec=Embedder)
    embedder.embed_texts.return_value = np.random.rand(5, 384).astype(np.float32)
    embedder.embed_query.return_value = np.random.rand(1, 384).astype(np.float32)
    return embedder
@pytest.fixture
def vector_store(temp_dir):
    """Create a vector store in a temporary directory."""
    store_path = Path(temp_dir) / "test_vector_store"
    return VectorStore(store_path=str(store_path))
class TestDocumentLoader:
    """Tests for the DocumentLoader class."""
    def test_load_text_file(self, document_loader, sample_text_file):
        """Test loading a text file produces document chunks."""
        docs = document_loader.load_file(sample_text_file)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
    def test_chunk_metadata(self, document_loader, sample_text_file):
        """Test that loaded documents have correct metadata."""
        docs = document_loader.load_file(sample_text_file)
        for doc in docs:
            assert "source" in doc.metadata
            assert "chunk_index" in doc.metadata
            assert "file_type" in doc.metadata
            assert doc.metadata["file_type"] == "txt"
    def test_load_directory(self, document_loader, temp_dir, sample_text_file):
        """Test loading all files from a directory."""
        # Create an additional file
        extra_file = Path(temp_dir) / "extra.txt"
        extra_file.write_text("Extra content for testing.")
        docs = document_loader.load_directory(temp_dir)
        assert len(docs) > 0
    def test_load_nonexistent_file(self, document_loader):
        """Test that loading a nonexistent file raises an error."""
        with pytest.raises(FileNotFoundError):
            document_loader.load_file("/nonexistent/path/file.txt")
    def test_load_empty_directory(self, document_loader, temp_dir):
        """Test loading from an empty directory returns empty list."""
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()
        docs = document_loader.load_directory(str(empty_dir))
        assert docs == []
    def test_chunk_size_respected(self, sample_text_file):
        """Test that chunk size configuration is respected."""
        loader = DocumentLoader(chunk_size=50, chunk_overlap=10)
        docs = loader.load_file(sample_text_file)
        for doc in docs:
            # Allow some overflow due to splitting at sentence boundaries
            assert len(doc.page_content) <= 100 # Allow 2x as buffer
class TestVectorStore:
    """Tests for the VectorStore class."""
    def test_add_documents(self, vector_store):
        """Test adding documents to the vector store."""
        docs = [
            Document(page_content="Test content 1", metadata={"source": "test.txt"}),
            Document(page_content="Test content 2", metadata={"source": "test.txt"}),
        ]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        vector_store.add_documents(docs, embeddings)
        assert vector_store.get_document_count() == 2
    def test_similarity_search(self, vector_store):
        """Test similarity search returns results."""
        docs = [
            Document(page_content="Machine learning basics", metadata={"source": "ml.txt"}),
            Document(page_content="Deep learning networks", metadata={"source": "dl.txt"}),
            Document(page_content="Cooking recipes", metadata={"source": "food.txt"}),
        ]
        embeddings = np.random.rand(3, 384).astype(np.float32)
        vector_store.add_documents(docs, embeddings)
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        results = vector_store.similarity_search(query_embedding, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)
    def test_persistence(self, temp_dir):
        """Test that the vector store persists to and loads from disk."""
        store_path = Path(temp_dir) / "persist_test"
        # Create and populate store
        store1 = VectorStore(store_path=str(store_path))
        docs = [Document(page_content="Persistent data", metadata={"source": "test.txt"})]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        store1.add_documents(docs, embeddings)
        # Load from disk
        store2 = VectorStore(store_path=str(store_path))
        assert store2.get_document_count() == 1
    def test_get_sources(self, vector_store):
        """Test getting unique source list."""
        docs = [
            Document(page_content="Content 1", metadata={"source": "file_a.txt"}),
            Document(page_content="Content 2", metadata={"source": "file_b.txt"}),
            Document(page_content="Content 3", metadata={"source": "file_a.txt"}),
        ]
        embeddings = np.random.rand(3, 384).astype(np.float32)
        vector_store.add_documents(docs, embeddings)
        sources = vector_store.get_sources()
        assert sorted(sources) == ["file_a.txt", "file_b.txt"]
    def test_empty_store_search(self, vector_store):
        """Test search on empty store returns empty list."""
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        results = vector_store.similarity_search(query_embedding, top_k=5)
        assert results == []
    def test_clear_store(self, vector_store):
        """Test clearing the vector store."""
        docs = [Document(page_content="Data", metadata={"source": "test.txt"})]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        vector_store.add_documents(docs, embeddings)
        vector_store.clear()
        assert vector_store.get_document_count() == 0
class TestRetriever:
    """Tests for the Retriever class."""
    def test_retrieve_with_results(self, mock_embedder, vector_store):
        """Test retrieval returns formatted context."""
        docs = [
            Document(page_content="ML is powerful", metadata={"source": "ml.txt", "page": 1}),
            Document(page_content="DL uses layers", metadata={"source": "dl.txt", "page": 2}),
        ]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        vector_store.add_documents(docs, embeddings)
        retriever = Retriever(
            embedder=mock_embedder,
            vector_store=vector_store,
            top_k=2,
            relevance_threshold=0.0, # Accept all results
        )
        result = retriever.retrieve("What is machine learning?")
        assert "context" in result
        assert "sources" in result
        assert "confidence" in result
    def test_retrieve_empty_query(self, mock_embedder, vector_store):
        """Test retrieval with empty query."""
        retriever = Retriever(
            embedder=mock_embedder,
            vector_store=vector_store,
            top_k=4,
            relevance_threshold=0.3,
        )
        result = retriever.retrieve("")
        assert result["context"] == ""
        assert result["sources"] == []
    def test_retrieve_no_results(self, mock_embedder, temp_dir):
        """Test retrieval from empty store."""
        empty_store = VectorStore(store_path=str(Path(temp_dir) / "empty_store"))
        retriever = Retriever(
            embedder=mock_embedder,
            vector_store=empty_store,
            top_k=4,
            relevance_threshold=0.3,
        )
        result = retriever.retrieve("test query")
        assert result["context"] == ""
        assert result["confidence"] == 0.0