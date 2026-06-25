"""
Text embedding module using sentence-transformers.
Converts text strings into dense vector representations
for semantic similarity search.
"""
from typing import List
import numpy as np
from utils.logger import get_logger
logger = get_logger(__name__)
_MODEL_CACHE = {}
class Embedder:
    """Sentence-transformer based text embedder with model caching."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedder with a specific model.
        Args:
            model_name: HuggingFace model identifier for sentence-transformers.
        """
        self.model_name = model_name
        self._model = None
    @property
    def model(self):
        """Lazy-load and cache the embedding model."""
        if self._model is None:
            if self.model_name in _MODEL_CACHE:
                self._model = _MODEL_CACHE[self.model_name]
                logger.info(f"Loaded embedding model from cache: {self.model_name}")
            else:
                logger.info(f"Loading embedding model: {self.model_name}...")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                _MODEL_CACHE[self.model_name] = self._model
                logger.info(f"Embedding model loaded and cached: {self.model_name}")
        return self._model
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.
        Args:
            texts: List of text strings to embed.
        Returns:
            NumPy array of shape (num_texts, embedding_dim) containing embedding vectors.
        """
        if not texts:
            logger.warning("Empty text list provided for embedding.")
            return np.array([])
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        Args:
            query: The query text to embed.
        Returns:
            NumPy array of shape (1, embedding_dim) containing the query embedding.
        """
        if not query.strip():
            logger.warning("Empty query provided for embedding.")
            return np.array([])
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()