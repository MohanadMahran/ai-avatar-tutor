"""
Tests for the avatar video generation module.
Tests video generation, polling, and fallback behavior
when HeyGen API encounters issues.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pipeline.tts_avatar import AvatarGenerator
@pytest.fixture
def avatar_generator():
    """Create an avatar generator instance for testing."""
    with patch("pipeline.tts_avatar.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            heygen_api_key="test_api_key",
            heygen_avatar_id="test_avatar_id",
            heygen_voice_id="test_voice_id",
        )
        generator = AvatarGenerator()
        generator.poll_interval = 0.1 # Speed up tests
        generator.max_poll_attempts = 3
    return generator
@pytest.mark.asyncio
async def test_generate_video_success(avatar_generator):
    """Test successful video generation flow."""
    create_response = MagicMock()
    create_response.status_code = 200
    create_response.json.return_value = {"data": {"video_id": "vid_123"}}
    status_response = MagicMock()
    status_response.status_code = 200
    status_response.json.return_value = {
        "data": {
            "status": "completed",
            "video_url": "https://example.com/video.mp4",
        }
    }
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=status_response)
        mock_client_class.return_value = mock_client
        result = await avatar_generator.generate_avatar_video("Hello, I am your tutor.")
    assert result["status"] == "success"
    assert result["video_url"] == "https://example.com/video.mp4"
    assert result["error"] is None
@pytest.mark.asyncio
async def test_generate_video_empty_text(avatar_generator):
    """Test that empty text returns a failure response."""
    result = await avatar_generator.generate_avatar_video("")
    assert result["status"] == "failed"
    assert result["video_url"] is None
    assert "No text" in result["error"]
@pytest.mark.asyncio
async def test_generate_video_no_api_key():
    """Test fallback when no API key is configured."""
    with patch("pipeline.tts_avatar.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            heygen_api_key="",
            heygen_avatar_id="",
            heygen_voice_id="",
        )
        generator = AvatarGenerator()
    result = await generator.generate_avatar_video("Test text")
    assert result["status"] == "failed"
    assert "not configured" in result["error"]
@pytest.mark.asyncio
async def test_generate_video_create_failure(avatar_generator):
    """Test handling of video creation failure."""
    error_response = MagicMock()
    error_response.status_code = 403
    error_response.text = "Forbidden"
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=error_response)
        mock_client_class.return_value = mock_client
        result = await avatar_generator.generate_avatar_video("Test text")
    assert result["status"] == "failed"
    assert result["video_url"] is None
@pytest.mark.asyncio
async def test_generate_video_polling_timeout(avatar_generator):
    """Test handling of video generation timeout during polling."""
    create_response = MagicMock()
    create_response.status_code = 200
    create_response.json.return_value = {"data": {"video_id": "vid_timeout"}}
    processing_response = MagicMock()
    processing_response.status_code = 200
    processing_response.json.return_value = {
        "data": {"status": "processing"}
    }
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=processing_response)
        mock_client_class.return_value = mock_client
        result = await avatar_generator.generate_avatar_video("Test text")
    assert result["status"] == "failed"
    assert "timed out" in result["error"] or "failed" in result["error"].lower()
@pytest.mark.asyncio
async def test_generate_video_polling_failure_status(avatar_generator):
    """Test handling of video generation failure status from HeyGen."""
    create_response = MagicMock()
    create_response.status_code = 200
    create_response.json.return_value = {"data": {"video_id": "vid_fail"}}
    failed_response = MagicMock()
    failed_response.status_code = 200
    failed_response.json.return_value = {
        "data": {
            "status": "failed",
            "error": "Text too long for avatar.",
        }
    }
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=failed_response)
        mock_client_class.return_value = mock_client
        result = await avatar_generator.generate_avatar_video("Test text")
    assert result["status"] == "failed"
    assert result["video_url"] is None
@pytest.mark.asyncio
async def test_text_truncation(avatar_generator):
    """Test that long text is truncated before sending to API."""
    long_text = "A" * 3000 # Exceeds the 1500 char limit
    create_response = MagicMock()
    create_response.status_code = 200
    create_response.json.return_value = {"data": {"video_id": "vid_trunc"}}
    status_response = MagicMock()
    status_response.status_code = 200
    status_response.json.return_value = {
        "data": {"status": "completed", "video_url": "https://example.com/v.mp4"}
    }
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=status_response)
        mock_client_class.return_value = mock_client
        result = await avatar_generator.generate_avatar_video(long_text)
    # Should still succeed despite truncation
    assert result["status"] == "success"
    # Verify the text was truncated in the API call
    call_args = mock_client.post.call_args
    payload = call_args.kwargs.get("json", {})
    input_text = payload["video_inputs"][0]["voice"]["input_text"]
    assert len(input_text) <= 1500