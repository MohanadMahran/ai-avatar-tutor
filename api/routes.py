"""
FastAPI route definitions for the AI Avatar Tutor API.
Provides endpoints for voice interaction, document management,
health checks, and conversation control.
"""
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
import json
from config.settings import get_settings
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.rag import RAGPipeline
from api.schemas import (
    InteractResponse,
    TextInteractRequest,
    UploadResponse,
    HealthResponse,
    ConversationClearResponse,
    DocumentListResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    ErrorResponse,
)
from utils.logger import get_logger
logger = get_logger(__name__)
router = APIRouter()
_orchestrator: PipelineOrchestrator = None
_rag_pipeline: RAGPipeline = None
def get_orchestrator() -> PipelineOrchestrator:
    """Get or create the pipeline orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator
def get_rag_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
@router.post("/interact", response_model=InteractResponse)
async def interact(
    audio: UploadFile = File(..., description="Audio file from microphone recording"),
    generate_video: bool = Form(default=True, description="Whether to generate avatar video"),
) -> InteractResponse:
    """
    Main interaction endpoint. Accepts audio, returns transcription,
    AI response, and avatar video URL.
    Args:
        audio: Uploaded audio file (webm, wav, mp3, ogg).
        generate_video: Whether to generate the avatar video.
    Returns:
        InteractResponse with all pipeline outputs.
    """
    logger.info(f"Received interaction request. Audio file: {audio.filename}, size: {audio.size}")
    try:
        audio_data = await audio.read()
        if not audio_data or len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file received.")
        if len(audio_data) > 10 * 1024 * 1024: # 10MB limit
            raise HTTPException(status_code=400, detail="Audio file too large. Maximum 10MB.")
        orchestrator = get_orchestrator()
        result = await orchestrator.process_audio(
            audio_data=audio_data,
            filename=audio.filename or "audio.webm",
            generate_video=generate_video,
        )
        return InteractResponse(
            transcription=result.get("transcription", ""),
            response_text=result.get("response_text", ""),
            video_url=result.get("video_url"),
            confidence=result.get("confidence", 0.0),
            language=result.get("language", "en"),
            sources=result.get("sources", []),
            timing=result.get("timing", {}),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interaction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
@router.post("/interact-text", response_model=InteractResponse)
async def interact_text(request: TextInteractRequest) -> InteractResponse:
    """
    Text-based interaction endpoint (skips STT).
    Args:
        request: TextInteractRequest with the user's text query.
    Returns:
        InteractResponse with pipeline outputs.
    """
    logger.info(f"Received text interaction: '{request.text[:60]}...'")
    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.process_text(
            text=request.text,
            generate_video=request.generate_video,
        )
        return InteractResponse(
            transcription=result.get("transcription", ""),
            response_text=result.get("response_text", ""),
            video_url=result.get("video_url"),
            confidence=result.get("confidence", 0.0),
            language=result.get("language", "en"),
            sources=result.get("sources", []),
            timing=result.get("timing", {}),
        )
    except Exception as e:
        logger.error(f"Text interaction error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
@router.post("/interact-stream")
async def interact_stream(
    audio: UploadFile = File(...),
) -> StreamingResponse:
    """
    Streaming interaction endpoint that returns tokens as they are generated.
    Args:
        audio: Uploaded audio file.
    Returns:
        StreamingResponse with server-sent events.
    """
    logger.info("Received streaming interaction request.")
    try:
        audio_data = await audio.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file.")
        orchestrator = get_orchestrator()
        # Transcribe
        transcription = await orchestrator.stt.transcribe_audio(
            audio_data, filename=audio.filename or "audio.webm"
        )
        if not transcription.strip():
            async def empty_stream():
                yield f"data: {json.dumps({'type': 'error', 'content': 'Could not transcribe audio.'})}\n\n"
            return StreamingResponse(empty_stream(), media_type="text/event-stream")
        # RAG retrieval
        rag_result = orchestrator.rag.retrieve_context(transcription)
        context = rag_result.get("context", "")
        from utils.helpers import detect_language
        language = detect_language(transcription)
        async def stream_generator():
            # Send transcription
            yield f"data: {json.dumps({'type': 'transcription', 'content': transcription})}\n\n"
            # Send metadata
            yield f"data: {json.dumps({'type': 'metadata', 'content': '', 'metadata': {'confidence': rag_result.get('confidence', 0.0), 'sources': rag_result.get('sources', []), 'language': language}})}\n\n"
            # Stream LLM response
            full_response = ""
            async for token in orchestrator.llm.generate_response_stream(
                query=transcription,
                context=context,
                conversation_history=orchestrator.conversation_history,
                language=language,
            ):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            # Update conversation history
            orchestrator.conversation_history = orchestrator.llm.manage_conversation_history(
                orchestrator.conversation_history, transcription, full_response
            )
            # Signal completion
            yield f"data: {json.dumps({'type': 'complete', 'content': full_response})}\n\n"
            # Start avatar generation in background
            if orchestrator.settings.heygen_api_key:
                try:
                    avatar_result = await orchestrator.avatar.generate_avatar_video(full_response)
                    if avatar_result.get("video_url"):
                        yield f"data: {json.dumps({'type': 'video', 'content': avatar_result['video_url']})}\n\n"
                except Exception as e:
                    logger.warning(f"Avatar generation failed during stream: {e}")
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Streaming interaction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/upload-docs", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(..., description="Document files to upload and index"),
) -> UploadResponse:
    """
    Upload one or more document files and index them in the vector store.
    Args:
        files: List of uploaded document files (PDF, TXT, MD).
    Returns:
        UploadResponse with processing results.
    """
    settings = get_settings()
    docs_path = Path(settings.docs_path)
    docs_path.mkdir(parents=True, exist_ok=True)
    supported_extensions = {".pdf", ".txt", ".md"}
    processed_files: List[str] = []
    total_chunks = 0
    rag = get_rag_pipeline()
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in supported_extensions:
            logger.warning(f"Skipping unsupported file: {file.filename}")
            continue
        file_path = docs_path / file.filename
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            chunks_count = rag.index_single_file(str(file_path))
            total_chunks += chunks_count
            processed_files.append(file.filename)
            logger.info(f"Uploaded and indexed: {file.filename} ({chunks_count} chunks)")
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            if file_path.exists():
                file_path.unlink()
    if not processed_files:
        raise HTTPException(
            status_code=400,
            detail="No valid documents were processed. Supported formats: PDF, TXT, MD."
        )
    return UploadResponse(
        message=f"Successfully processed {len(processed_files)} file(s).",
        files_processed=len(processed_files),
        total_chunks=total_chunks,
        filenames=processed_files,
    )
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check system health including vector store status.
    Returns:
        HealthResponse with system status information.
    """
    try:
        rag = get_rag_pipeline()
        doc_count = rag.get_document_count()
        sources = rag.get_indexed_sources()
        return HealthResponse(
            status="healthy",
            vector_store_loaded=doc_count > 0,
            documents_indexed=doc_count,
            indexed_sources=sources,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            vector_store_loaded=False,
            documents_indexed=0,
            indexed_sources=[],
        )
@router.delete("/conversation", response_model=ConversationClearResponse)
async def clear_conversation() -> ConversationClearResponse:
    """
    Clear the conversation history.
    Returns:
        ConversationClearResponse confirming the action.
    """
    try:
        orchestrator = get_orchestrator()
        orchestrator.clear_conversation()
        return ConversationClearResponse(
            message="Conversation history cleared successfully.",
            success=True,
        )
    except Exception as e:
        logger.error(f"Failed to clear conversation: {e}")
        return ConversationClearResponse(
            message=f"Failed to clear conversation: {str(e)}",
            success=False,
        )
@router.get("/docs-list", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """
    List all currently indexed documents.
    Returns:
        DocumentListResponse with document list and count.
    """
    try:
        rag = get_rag_pipeline()
        sources = rag.get_indexed_sources()
        total_chunks = rag.get_document_count()
        return DocumentListResponse(
            documents=sources,
            total_chunks=total_chunks,
        )
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return DocumentListResponse(documents=[], total_chunks=0)
@router.post("/delete-doc", response_model=DocumentDeleteResponse)
async def delete_document(request: DocumentDeleteRequest) -> DocumentDeleteResponse:
    """
    Delete a specific document from the vector store.
    Args:
        request: DocumentDeleteRequest with the source filename.
    Returns:
        DocumentDeleteResponse confirming the action.
    """
    try:
        rag = get_rag_pipeline()
        success = rag.delete_source(request.source_name)
        # Also delete the file from disk
        settings = get_settings()
        file_path = Path(settings.docs_path) / request.source_name
        if file_path.exists():
            file_path.unlink()
        if success:
            return DocumentDeleteResponse(
                message=f"Document '{request.source_name}' deleted successfully.",
                success=True,
            )
        else:
            return DocumentDeleteResponse(
                message=f"Document '{request.source_name}' not found in index.",
                success=False,
            )
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/conversation-history")
async def get_conversation_history():
    """Get the current conversation history."""
    orchestrator = get_orchestrator()
    return {"history": orchestrator.get_conversation_history()}