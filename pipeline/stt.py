"""
Speech-to-Text module using ElevenLabs API.
Handles transcription of audio input from the user's microphone
into text for further processing by the pipeline.
"""
import httpx
from pathlib import Path
from typing import Optional
from config.settings import get_settings
from utils.logger import get_logger
logger = get_logger(__name__)
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
class SpeechToText:
    """ElevenLabs Speech-to-Text transcription service."""
    def __init__(self) -> None:
        """Initialize the STT service with API credentials."""
        self.settings = get_settings()
        self.api_key = self.settings.elevenlabs_api_key
        self.timeout = 30.0
    async def transcribe_audio(
        self,
        audio_data: bytes,
        filename: str = "audio.webm",
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio bytes to text using ElevenLabs API.
        Args:
            audio_data: Raw audio bytes from the microphone recording.
            filename: Name of the audio file with extension for format detection.
            language: Optional language code hint (e.g., 'en', 'ar').
        Returns:
            Transcribed text string.
        Raises:
            ValueError: If audio data is empty.
            RuntimeError: If the API call fails after retries.
        """
        if not audio_data or len(audio_data) == 0:
            logger.error("Empty audio data received for transcription.")
            raise ValueError("Audio data is empty. Please record audio before sending.")
        logger.info(f"Starting transcription of {len(audio_data)} bytes of audio...")
        headers = {
            "xi-api-key": self.api_key,
        }
        content_type_map = {
            ".webm": "audio/webm",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
        }
        ext = Path(filename).suffix.lower()
        content_type = content_type_map.get(ext, "audio/webm")
        files = {
            "file": (filename, audio_data, content_type),
        }
        data = {}
        if language:
            data["language"] = language
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        ELEVENLABS_STT_URL,
                        headers=headers,
                        files=files,
                        data=data,
                    )
                if response.status_code == 200:
                    result = response.json()
                    transcribed_text = result.get("text", "").strip()
                    if not transcribed_text:
                        logger.warning("Transcription returned empty text.")
                        return ""
                    logger.info(f"Transcription successful: '{transcribed_text[:50]}...'")
                    return transcribed_text
                elif response.status_code == 429:
                    logger.warning(f"Rate limited on attempt {attempt + 1}. Retrying...")
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    error_detail = response.text
                    logger.error(
                        f"ElevenLabs STT API error (status {response.status_code}): {error_detail}"
                    )
                    if attempt < max_retries - 1:
                        continue
                    raise RuntimeError(
                        f"ElevenLabs STT API failed with status {response.status_code}: {error_detail}"
                    )
            except httpx.TimeoutException:
                logger.error(f"Timeout on transcription attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError("ElevenLabs STT API timed out after all retries.")
            except httpx.ConnectError as e:
                logger.error(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(1)
                    continue
                raise RuntimeError(f"Cannot connect to ElevenLabs API: {e}")
        raise RuntimeError("Transcription failed after all retry attempts.")
    async def transcribe_file(self, file_path: str) -> str:
        """
        Transcribe an audio file from disk.
        Args:
            file_path: Path to the audio file on disk.
        Returns:
            Transcribed text string.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        audio_data = path.read_bytes()
        return await self.transcribe_audio(audio_data, filename=path.name)