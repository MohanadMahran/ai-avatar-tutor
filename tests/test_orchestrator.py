import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pipeline.orchestrator import PipelineOrchestrator, STTError, LLMError, AvatarError

@pytest.fixture
def orchestrator():
    """Create a PipelineOrchestrator instance with mocked dependencies."""
    with patch("pipeline.orchestrator.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            heygen_api_key="test_heygen_key",
            enable_cache=False,
            cache_ttl_seconds=3600,
            max_conversation_history=10,
        )
        with patch("pipeline.orchestrator.SpeechToText"), \
             patch("pipeline.orchestrator.RAGPipeline"), \
             patch("pipeline.orchestrator.LLMService"), \
             patch("pipeline.orchestrator.AvatarGenerator"):
            orch = PipelineOrchestrator()
            
            # Setup mock services
            orch.stt.transcribe_audio = AsyncMock()
            orch.rag.retrieve_context = MagicMock()
            orch.llm.maybe_summarize_history = AsyncMock(side_effect=lambda x: x)
            orch.llm.generate_response = AsyncMock()
            orch.llm.manage_conversation_history = MagicMock(side_effect=lambda h, q, r: h + [{"role": "user", "content": q}, {"role": "assistant", "content": r}])
            orch.avatar.generate_avatar_video = AsyncMock()
            
    return orch

@pytest.mark.asyncio
async def test_process_audio_success(orchestrator):
    """Test full successful audio processing flow."""
    orchestrator.stt.transcribe_audio.return_value = "Hello"
    orchestrator.rag.retrieve_context.return_value = {"context": "context doc", "confidence": 0.9, "sources": ["doc.txt"]}
    orchestrator.llm.generate_response.return_value = "World"
    orchestrator.avatar.generate_avatar_video.return_value = {"status": "success", "video_url": "http://example.com/video.mp4"}
    
    result = await orchestrator.process_audio(b"audio", "test.webm", generate_video=True)
    
    assert result["transcription"] == "Hello"
    assert result["response_text"] == "World"
    assert result["video_url"] == "http://example.com/video.mp4"
    
    orchestrator.stt.transcribe_audio.assert_called_once()
    orchestrator.rag.retrieve_context.assert_called_once()
    orchestrator.llm.generate_response.assert_called_once()
    orchestrator.avatar.generate_avatar_video.assert_called_once()

@pytest.mark.asyncio
async def test_process_audio_stt_failure(orchestrator):
    """Test that STT failure raises STTError and stops further execution."""
    orchestrator.stt.transcribe_audio.side_effect = Exception("STT API crashed")
    
    with pytest.raises(STTError, match="STT failed: I couldn't understand the audio"):
        await orchestrator.process_audio(b"audio", "test.webm")
        
    orchestrator.rag.retrieve_context.assert_not_called()
    orchestrator.llm.generate_response.assert_not_called()
    orchestrator.avatar.generate_avatar_video.assert_not_called()

@pytest.mark.asyncio
async def test_process_audio_stt_empty(orchestrator):
    """Test that empty transcription raises STTError and stops execution."""
    orchestrator.stt.transcribe_audio.return_value = "   "
    
    with pytest.raises(STTError, match="I didn't catch anything"):
        await orchestrator.process_audio(b"audio", "test.webm")
        
    orchestrator.rag.retrieve_context.assert_not_called()
    orchestrator.llm.generate_response.assert_not_called()
    orchestrator.avatar.generate_avatar_video.assert_not_called()

@pytest.mark.asyncio
async def test_process_audio_llm_failure(orchestrator):
    """Test that LLM failure raises LLMError and stops before HeyGen."""
    orchestrator.stt.transcribe_audio.return_value = "Hello"
    orchestrator.rag.retrieve_context.return_value = {"context": "", "confidence": 0.0, "sources": []}
    orchestrator.llm.generate_response.side_effect = Exception("Groq offline")
    
    with pytest.raises(LLMError, match="LLM failed, cannot generate response"):
        await orchestrator.process_audio(b"audio", "test.webm")
        
    orchestrator.avatar.generate_avatar_video.assert_not_called()

@pytest.mark.asyncio
async def test_process_audio_llm_empty(orchestrator):
    """Test that empty LLM response raises LLMError and stops before HeyGen."""
    orchestrator.stt.transcribe_audio.return_value = "Hello"
    orchestrator.rag.retrieve_context.return_value = {"context": "", "confidence": 0.0, "sources": []}
    orchestrator.llm.generate_response.return_value = ""
    
    with pytest.raises(LLMError, match="LLM failed, cannot generate response"):
        await orchestrator.process_audio(b"audio", "test.webm")
        
    orchestrator.avatar.generate_avatar_video.assert_not_called()

@pytest.mark.asyncio
async def test_process_audio_heygen_failure(orchestrator):
    """Test that HeyGen failure raises AvatarError."""
    orchestrator.stt.transcribe_audio.return_value = "Hello"
    orchestrator.rag.retrieve_context.return_value = {"context": "", "confidence": 0.0, "sources": []}
    orchestrator.llm.generate_response.return_value = "World"
    orchestrator.avatar.generate_avatar_video.side_effect = Exception("HeyGen timeout")
    
    with pytest.raises(AvatarError, match="Heygen failed"):
        await orchestrator.process_audio(b"audio", "test.webm")

@pytest.mark.asyncio
async def test_process_audio_heygen_status_failed(orchestrator):
    """Test that HeyGen returning a non-success status raises AvatarError."""
    orchestrator.stt.transcribe_audio.return_value = "Hello"
    orchestrator.rag.retrieve_context.return_value = {"context": "", "confidence": 0.0, "sources": []}
    orchestrator.llm.generate_response.return_value = "World"
    orchestrator.avatar.generate_avatar_video.return_value = {"status": "failed", "error": "Voice not found"}
    
    with pytest.raises(AvatarError, match="Heygen failed: Voice not found"):
        await orchestrator.process_audio(b"audio", "test.webm")
