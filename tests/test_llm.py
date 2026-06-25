"""
Tests for the LLM pipeline module.
Tests response generation with context, conversation history,
and error handling scenarios.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pipeline.llm import LLMService
@pytest.fixture
def llm_service():
    """Create an LLM service instance for testing."""
    with patch("pipeline.llm.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            groq_api_key="test_key",
            groq_model="llama3-70b-8192",
            max_conversation_history=10,
        )
        with patch("pipeline.llm.Groq"):
            with patch("pipeline.llm.AsyncGroq") as mock_async_groq:
                service = LLMService()
                service.async_client = AsyncMock()
    return service
@pytest.fixture
def sample_context():
    """Sample RAG context for testing."""
    return """[Source: biology.pdf, Page: 5, Relevance: 0.85]
Photosynthesis is the process by which plants convert light energy into chemical energy.
It occurs in the chloroplasts of plant cells."""
@pytest.fixture
def sample_history():
    """Sample conversation history."""
    return [
        {"role": "user", "content": "What is biology?"},
        {"role": "assistant", "content": "Biology is the study of living organisms."},
    ]
@pytest.mark.asyncio
async def test_generate_response_with_context(llm_service, sample_context, sample_history):
    """Test successful response generation with context."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Photosynthesis is how plants make food from sunlight."
    llm_service.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
    result = await llm_service.generate_response(
        query="How does photosynthesis work?",
        context=sample_context,
        conversation_history=sample_history,
        language="en",
    )
    assert result == "Photosynthesis is how plants make food from sunlight."
    llm_service.async_client.chat.completions.create.assert_called_once()
@pytest.mark.asyncio
async def test_generate_response_without_context(llm_service):
    """Test response generation without RAG context."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I'll answer based on my general knowledge."
    llm_service.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
    result = await llm_service.generate_response(
        query="What is the capital of France?",
        context="",
        conversation_history=[],
        language="en",
    )
    assert "general knowledge" in result.lower() or len(result) > 0
@pytest.mark.asyncio
async def test_generate_response_arabic(llm_service, sample_context):
    """Test response generation with Arabic language."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "تاتابنلا لوحت ةيلمع وه يئوضلا ليثمتلا"
    llm_service.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
    result = await llm_service.generate_response(
        query="؟يئوضلا ليثمتلا لمعي فيك",
        context=sample_context,
        conversation_history=[],
        language="ar",
    )
    assert len(result) > 0
@pytest.mark.asyncio
async def test_generate_response_api_error(llm_service):
    """Test handling of API errors."""
    llm_service.async_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API connection failed")
    )
    with pytest.raises(RuntimeError, match="Failed to generate LLM response"):
        await llm_service.generate_response(
            query="Test query",
            context="Test context",
            conversation_history=[],
            language="en",
        )
@pytest.mark.asyncio
async def test_conversation_history_management(llm_service):
    """Test conversation history is properly managed."""
    history = []
    history = llm_service.manage_conversation_history(
        history, "Hello!", "Hi there, how can I help you?"
    )
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello!"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there, how can I help you?"
@pytest.mark.asyncio
async def test_summarize_history(llm_service):
    """Test conversation history summarization."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "The student asked about photosynthesis and biology."
    llm_service.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
    long_history = [
        {"role": "user", "content": f"Question {i}"}
        for i in range(20)
    ]
    summary = await llm_service.summarize_history(long_history)
    assert len(summary) > 0
@pytest.mark.asyncio
async def test_maybe_summarize_short_history(llm_service):
    """Test that short history is not summarized."""
    short_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    result = await llm_service.maybe_summarize_history(short_history)
    assert result == short_history
@pytest.mark.asyncio
async def test_system_prompt_includes_context(llm_service, sample_context):
    """Test that the system prompt correctly includes context."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    llm_service.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
    await llm_service.generate_response(
        query="Test",
        context=sample_context,
        conversation_history=[],
        language="en",
    )
    call_args = llm_service.async_client.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages", [])
    system_msg = messages[0]
    assert system_msg["role"] == "system"
    assert "Photosynthesis" in system_msg["content"]