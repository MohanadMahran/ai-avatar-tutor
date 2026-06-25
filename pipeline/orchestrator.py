"""
Pipeline orchestrator that coordinates all components.
Manages the full flow from audio input through transcription,
retrieval, generation, and avatar video creation.
"""
import time
import hashlib
import asyncio
from typing import Dict, List, Optional, Any
from config.settings import get_settings
from pipeline.stt import SpeechToText
from pipeline.rag import RAGPipeline
from pipeline.llm import LLMService
from pipeline.tts_avatar import AvatarGenerator
from utils.logger import get_logger
from utils.helpers import detect_language
logger = get_logger(__name__)
class STTError(Exception):
    """Exception raised when Speech-to-Text conversion fails."""
    pass


class LLMError(Exception):
    """Exception raised when LLM generation fails."""
    pass


class AvatarError(Exception):
    """Exception raised when HeyGen avatar video generation fails."""
    pass


class ResponseCache:
    """Simple in-memory cache for query responses."""
    def __init__(self, ttl_seconds: int = 3600) -> None:
        """Initialize the response cache."""
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
    def _hash_query(self, query: str, context_hash: str) -> str:
        """Generate a hash key for a query-context combination."""
        combined = f"{query.strip().lower()}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    def get(self, query: str, context_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if available and not expired."""
        key = self._hash_query(query, context_hash)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                logger.info("Cache hit for query.")
                return entry["response"]
            else:
                del self.cache[key]
        return None
    def set(self, query: str, context_hash: str, response: Dict[str, Any]) -> None:
        """Store a response in the cache."""
        key = self._hash_query(query, context_hash)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
        }
    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()
class PipelineOrchestrator:
    """Main pipeline orchestrator coordinating all AI tutor components."""
    def __init__(self) -> None:
        """Initialize all pipeline components."""
        self.settings = get_settings()
        self.stt = SpeechToText()
        self.rag = RAGPipeline()
        self.llm = LLMService()
        self.avatar = AvatarGenerator()
        self.conversation_history: List[Dict[str, str]] = []
        self.cache = ResponseCache(ttl_seconds=self.settings.cache_ttl_seconds)
    async def process_audio(
        self,
        audio_data: bytes,
        filename: str = "audio.webm",
        generate_video: bool = True,
    ) -> Dict[str, Any]:
        """
        Process audio input through the full pipeline.
        Args:
            audio_data: Raw audio bytes from the user's microphone.
            filename: Audio filename with extension.
            generate_video: Whether to generate avatar video.
        Returns:
            Dictionary containing:
                - transcription: The transcribed user query
                - response_text: The LLM-generated tutor response
                - video_url: URL of avatar video (or None)
                - confidence: RAG retrieval confidence score
                - language: Detected language
                - sources: List of source documents used
                - timing: Dictionary of step durations
        """
        timing = {}
        result = {
            "transcription": "",
            "response_text": "",
            "video_url": None,
            "confidence": 0.0,
            "language": "en",
            "sources": [],
            "timing": timing,
        }
        # Step 1: Speech to Text
        step_start = time.time()
        try:
            transcription = await self.stt.transcribe_audio(audio_data, filename)
            result["transcription"] = transcription
            timing["stt"] = round(time.time() - step_start, 2)
            logger.info(f"STT completed in {timing['stt']}s: '{transcription[:60]}...'")
        except Exception as e:
            logger.error(f"STT failed: {e}")
            raise STTError("STT failed: I couldn't understand the audio. Please try speaking again.") from e
        if not transcription.strip():
            logger.error("STT returned empty transcription.")
            raise STTError("STT failed: I didn't catch anything. Could you please speak again?")
        # Step 2: Language Detection
        language = detect_language(transcription)
        result["language"] = language
        logger.info(f"Detected language: {language}")
        # Step 3: RAG Retrieval
        step_start = time.time()
        try:
            rag_result = self.rag.retrieve_context(transcription)
            context = rag_result.get("context", "")
            result["confidence"] = rag_result.get("confidence", 0.0)
            result["sources"] = rag_result.get("sources", [])
            timing["rag"] = round(time.time() - step_start, 2)
            logger.info(f"RAG retrieval completed in {timing['rag']}s (confidence: {result['confidence']:.2f})")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            context = ""
            timing["rag"] = round(time.time() - step_start, 2)
        # Step 4: Check cache
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        if self.settings.enable_cache:
            cached = self.cache.get(transcription, context_hash)
            if cached:
                cached["timing"] = timing
                cached["transcription"] = transcription
                logger.info("Returning cached response.")
                return cached
        # Step 5: LLM Response Generation
        step_start = time.time()
        try:
            self.conversation_history = await self.llm.maybe_summarize_history(
                self.conversation_history
            )
            response_text = await self.llm.generate_response(
                query=transcription,
                context=context,
                conversation_history=self.conversation_history,
                language=language,
            )
            if not response_text or not response_text.strip():
                raise LLMError("LLM failed, cannot generate response")
            result["response_text"] = response_text
            timing["llm"] = round(time.time() - step_start, 2)
            logger.info(f"LLM response generated in {timing['llm']}s")
            # Update conversation history
            self.conversation_history = self.llm.manage_conversation_history(
                self.conversation_history,
                transcription,
                response_text,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            timing["llm"] = round(time.time() - step_start, 2)
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"LLM failed, cannot generate response: {str(e)}") from e
        # Step 6: Avatar Video Generation
        if generate_video and self.settings.heygen_api_key:
            step_start = time.time()
            try:
                avatar_result = await self.avatar.generate_avatar_video(response_text)
                if avatar_result.get("status") != "success":
                    raise AvatarError(f"Heygen failed: {avatar_result.get('error', 'Unknown avatar generation error')}")
                result["video_url"] = avatar_result.get("video_url")
                timing["avatar"] = round(time.time() - step_start, 2)
                logger.info(f"Avatar video generated in {timing['avatar']}s")
            except Exception as e:
                logger.error(f"Avatar generation failed: {e}")
                timing["avatar"] = round(time.time() - step_start, 2)
                if isinstance(e, AvatarError):
                    raise
                raise AvatarError(f"Heygen failed: {str(e)}") from e
        else:
            timing["avatar"] = 0.0
        # Cache the result
        if self.settings.enable_cache:
            self.cache.set(transcription, context_hash, result)
        total_time = sum(timing.values())
        timing["total"] = round(total_time, 2)
        logger.info(f"Full pipeline completed in {total_time:.2f}s")
        return result
    async def process_text(
        self,
        text: str,
        generate_video: bool = True,
    ) -> Dict[str, Any]:
        """
        Process text input through the pipeline (skipping STT).
        Args:
            text: User's text query.
            generate_video: Whether to generate avatar video.
        Returns:
            Pipeline result dictionary.
        """
        timing = {}
        result = {
            "transcription": text,
            "response_text": "",
            "video_url": None,
            "confidence": 0.0,
            "language": detect_language(text),
            "sources": [],
            "timing": timing,
        }
        # RAG Retrieval
        step_start = time.time()
        try:
            rag_result = self.rag.retrieve_context(text)
            context = rag_result.get("context", "")
            result["confidence"] = rag_result.get("confidence", 0.0)
            result["sources"] = rag_result.get("sources", [])
            timing["rag"] = round(time.time() - step_start, 2)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            context = ""
            timing["rag"] = round(time.time() - step_start, 2)
        # LLM Generation
        step_start = time.time()
        try:
            self.conversation_history = await self.llm.maybe_summarize_history(
                self.conversation_history
            )
            response_text = await self.llm.generate_response(
                query=text,
                context=context,
                conversation_history=self.conversation_history,
                language=result["language"],
            )
            if not response_text or not response_text.strip():
                raise LLMError("LLM failed, cannot generate response")
            result["response_text"] = response_text
            timing["llm"] = round(time.time() - step_start, 2)
            self.conversation_history = self.llm.manage_conversation_history(
                self.conversation_history, text, response_text
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            timing["llm"] = round(time.time() - step_start, 2)
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"LLM failed, cannot generate response: {str(e)}") from e
        # Avatar Generation
        if generate_video and self.settings.heygen_api_key:
            step_start = time.time()
            try:
                avatar_result = await self.avatar.generate_avatar_video(response_text)
                if avatar_result.get("status") != "success":
                    raise AvatarError(f"Heygen failed: {avatar_result.get('error', 'Unknown avatar generation error')}")
                result["video_url"] = avatar_result.get("video_url")
                timing["avatar"] = round(time.time() - step_start, 2)
            except Exception as e:
                logger.error(f"Avatar generation failed: {e}")
                timing["avatar"] = round(time.time() - step_start, 2)
                if isinstance(e, AvatarError):
                    raise
                raise AvatarError(f"Heygen failed: {str(e)}") from e
        else:
            timing["avatar"] = 0.0
        timing["total"] = round(sum(timing.values()), 2)
        return result
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        self.cache.clear()
        logger.info("Conversation history and cache cleared.")
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()