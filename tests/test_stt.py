"""
Tests for the Speech-to-Text pipeline module.
Tests transcription with valid audio, empty audio,
and API failure scenarios.
"""
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from pipeline.stt import SpeechToText
@pytest.fixture
def stt_service():
    """Create an STT service instance for testing."""
    with patch("pipeline.stt.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            elevenlabs_api_key="test_api_key",
        )
        service = SpeechToText()
    return service
@pytest.fixture
def valid_audio_data():
    """Generate valid test audio data."""
    return b"\x00" * 5000 # 5KB of dummy audio data
@pytest.fixture
def empty_audio_data():
    """Generate empty audio data."""
    return b""
@pytest.mark.asyncio
async def test_transcribe_valid_audio(stt_service, valid_audio_data):
    """Test successful transcription with valid audio input."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": "Hello, how does photosynthesis work?"}
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        result = await stt_service.transcribe_audio(valid_audio_data, "test.webm")
    assert result == "Hello, how does photosynthesis work?"
@pytest.mark.asyncio
async def test_transcribe_empty_audio(stt_service, empty_audio_data):
    """Test that empty audio raises ValueError."""
    with pytest.raises(ValueError, match="Audio data is empty"):
        await stt_service.transcribe_audio(empty_audio_data)
@pytest.mark.asyncio
async def test_transcribe_api_failure(stt_service, valid_audio_data):
    """Test graceful handling of API failures."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        with pytest.raises(RuntimeError, match="ElevenLabs STT API failed"):
            await stt_service.transcribe_audio(valid_audio_data, "test.webm")
@pytest.mark.asyncio
async def test_transcribe_timeout(stt_service, valid_audio_data):
    """Test handling of request timeout."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client_class.return_value = mock_client
        with pytest.raises(RuntimeError, match="timed out"):
            await stt_service.transcribe_audio(valid_audio_data, "test.webm")
@pytest.mark.asyncio
async def test_transcribe_empty_response(stt_service, valid_audio_data):
    """Test handling of empty transcription response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": ""}
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        result = await stt_service.transcribe_audio(valid_audio_data, "test.webm")
    assert result == ""
@pytest.mark.asyncio
async def test_transcribe_rate_limit_retry(stt_service, valid_audio_data):
    """Test retry logic on rate limiting."""
    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.text = "Rate limited"
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {"text": "Retry succeeded"}
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            side_effect=[rate_limit_response, success_response]
        )
        mock_client_class.return_value = mock_client
        result = await stt_service.transcribe_audio(valid_audio_data, "test.webm")
    assert result == "Retry succeeded"