"""
Application settings loaded from environment variables using Pydantic BaseSettings.
All pipeline components import their configuration from this module.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
class Settings(BaseSettings):
    """Application settings with type validation and default values."""
    # Groq API Configuration
    groq_api_key: str = Field(
        default="",
        description="API key for Groq LLM service",
    )
    groq_model: str = Field(
        default="llama3-70b-8192",
        description="Groq model identifier to use for text generation",
    )
    # ElevenLabs API Configuration
    elevenlabs_api_key: str = Field(
        default="",
        description="API key for ElevenLabs speech services",
    )
    elevenlabs_voice_id: str = Field(
        default="",
        description="ElevenLabs voice ID for TTS",
    )
    # HeyGen API Configuration
    heygen_api_key: str = Field(
        default="",
        description="API key for HeyGen avatar video generation",
    )
    heygen_avatar_id: str = Field(
        default="",
        description="HeyGen avatar ID for video generation",
    )
    heygen_voice_id: str = Field(
        default="",
        description="HeyGen voice ID for avatar speech",
    )
    # RAG Settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for document embeddings",
    )
    vector_store_path: str = Field(
        default="./vector_store",
        description="Path to store the FAISS vector index",
    )
    docs_path: str = Field(
        default="./docs",
        description="Path to the documents directory",
    )
    chunk_size: int = Field(
        default=500,
        description="Character count for document chunks",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Character overlap between consecutive chunks",
    )
    top_k_results: int = Field(
        default=4,
        description="Number of top results to retrieve from vector store",
    )
    relevance_threshold: float = Field(
        default=0.3,
        description="Minimum similarity score threshold for retrieval",
    )
    # App Settings
    app_host: str = Field(
        default="0.0.0.0",
        description="Host address for the FastAPI server",
    )
    app_port: int = Field(
        default=8000,
        description="Port number for the FastAPI server",
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode with auto-reload",
    )
    max_conversation_history: int = Field(
        default=10,
        description="Maximum number of conversation turns to maintain",
    )
    # Audio Settings
    audio_sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz for recording",
    )
    audio_channels: int = Field(
        default=1,
        description="Number of audio channels (1 for mono)",
    )
    max_audio_duration_seconds: int = Field(
        default=60,
        description="Maximum allowed audio recording duration in seconds",
    )
    # Cache Settings
    enable_cache: bool = Field(
        default=True,
        description="Enable response caching for identical queries",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache time-to-live in seconds",
    )
    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()