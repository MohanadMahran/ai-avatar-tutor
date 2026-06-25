"""
Pydantic schemas for API request and response models.
Defines the data structures used for API communication
between the frontend and backend.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
class InteractResponse(BaseModel):
    """Response model for the /interact endpoint."""
    transcription: str = Field(
        description="Transcribed text from the user's audio input"
    )
    response_text: str = Field(
        description="AI tutor's text response"
    )
    video_url: Optional[str] = Field(
        default=None,
        description="URL of the generated avatar video"
    )
    confidence: float = Field(
        default=0.0,
        description="RAG retrieval confidence score (0-1)"
    )
    language: str = Field(
        default="en",
        description="Detected language of the input"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of source documents used for context"
    )
    timing: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information for each pipeline step"
    )
class TextInteractRequest(BaseModel):
    """Request model for text-based interaction."""
    text: str = Field(description="User's text query")
    generate_video: bool = Field(
        default=True,
        description="Whether to generate avatar video"
    )
class UploadResponse(BaseModel):
    """Response model for document upload."""
    message: str = Field(description="Status message")
    files_processed: int = Field(description="Number of files successfully processed")
    total_chunks: int = Field(description="Total document chunks indexed")
    filenames: List[str] = Field(
        default_factory=list,
        description="List of processed filenames"
    )
class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(description="Application status")
    vector_store_loaded: bool = Field(description="Whether vector store is loaded")
    documents_indexed: int = Field(description="Number of indexed document chunks")
    indexed_sources: List[str] = Field(
        default_factory=list,
        description="List of indexed source filenames"
    )
class ConversationClearResponse(BaseModel):
    """Response model for conversation clear endpoint."""
    message: str = Field(description="Confirmation message")
    success: bool = Field(description="Whether the operation succeeded")
class DocumentListResponse(BaseModel):
    """Response model for listing indexed documents."""
    documents: List[str] = Field(
        default_factory=list,
        description="List of indexed document filenames"
    )
    total_chunks: int = Field(description="Total number of indexed chunks")
class DocumentDeleteRequest(BaseModel):
    """Request model for deleting a document."""
    source_name: str = Field(description="Filename of the document to delete")
class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion."""
    message: str = Field(description="Status message")
    success: bool = Field(description="Whether deletion was successful")
class StreamingChunk(BaseModel):
    """Model for a streaming response chunk."""
    type: str = Field(description="Type of chunk: 'token', 'complete', 'error'")
    content: str = Field(default="", description="Token content")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )
class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")