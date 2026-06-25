"""
Avatar video generation module using HeyGen API.
Generates talking avatar videos from text responses,
providing a visual representation of the AI tutor.
"""
import asyncio
import httpx
from typing import Optional, Dict
from config.settings import get_settings
from utils.logger import get_logger
logger = get_logger(__name__)
HEYGEN_API_BASE = "https://api.heygen.com"
HEYGEN_VIDEO_GENERATE_URL = f"{HEYGEN_API_BASE}/v2/video/generate"
HEYGEN_VIDEO_STATUS_URL = f"{HEYGEN_API_BASE}/v1/video_status.get"
class AvatarGenerator:
    """HeyGen avatar video generation service."""
    def __init__(self) -> None:
        """Initialize the avatar generator with HeyGen API credentials."""
        self.settings = get_settings()
        self.api_key = self.settings.heygen_api_key
        self.avatar_id = self.settings.heygen_avatar_id
        self.voice_id = self.settings.heygen_voice_id
        self.timeout = 120.0
        self.poll_interval = 5.0
        self.max_poll_attempts = 60
    async def generate_avatar_video(self, text: str) -> Dict[str, Optional[str]]:
        """
        Generate a talking avatar video from text using HeyGen API.
        Args:
            text: The text for the avatar to speak.
        Returns:
            Dictionary containing:
                - video_url: URL of the generated video or None
                - status: 'success', 'pending', or 'failed'
                - error: Error message if failed, None otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for avatar generation.")
            return {
                "video_url": None,
                "status": "failed",
                "error": "No text provided for avatar generation.",
            }
        if not self.api_key:
            logger.warning("HeyGen API key not configured. Skipping avatar generation.")
            return {
                "video_url": None,
                "status": "failed",
                "error": "HeyGen API key not configured.",
            }
        logger.info(f"Generating avatar video for text ({len(text)} chars)...")
        try:
            video_id = await self._create_video(text)
            if not video_id:
                return {
                    "video_url": None,
                    "status": "failed",
                    "error": "Failed to create video generation task.",
                }
            video_url = await self._poll_video_status(video_id)
            if video_url:
                logger.info(f"Avatar video generated successfully: {video_url}")
                return {
                    "video_url": video_url,
                    "status": "success",
                    "error": None,
                }
            else:
                return {
                    "video_url": None,
                    "status": "failed",
                    "error": "Video generation timed out or failed.",
                }
        except Exception as e:
            logger.error(f"Avatar generation failed: {e}")
            return {
                "video_url": None,
                "status": "failed",
                "error": str(e),
            }
    async def _create_video(self, text: str) -> Optional[str]:
        """
        Submit a video generation request to HeyGen.
        Args:
            text: Text for the avatar to speak.
        Returns:
            Video ID for status polling, or None on failure.
        """
        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }
        truncated_text = text[:1500] if len(text) > 1500 else text
        
        voice_config = {
            "type": "text",
            "input_text": truncated_text,
            "speed": 1.0,
        }
        # Only add voice_id if it is configured and not the default placeholder
        if self.voice_id and not self.voice_id.startswith("your_"):
            voice_config["voice_id"] = self.voice_id

        payload = {
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": self.avatar_id,
                        "avatar_style": "normal",
                    },
                    "voice": voice_config,
                    "background": {
                        "type": "color",
                        "value": "#1a1a2e",
                    },
                }
            ],
            "dimension": {
                "width": 512,
                "height": 512,
            },
            "test": True,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    HEYGEN_VIDEO_GENERATE_URL,
                    headers=headers,
                    json=payload,
                )
            if response.status_code == 200:
                result = response.json()
                video_id = result.get("data", {}).get("video_id")
                if video_id:
                    logger.info(f"Video generation started. ID: {video_id}")
                    return video_id
                else:
                    logger.error(f"No video_id in response: {result}")
                    return None
            else:
                logger.error(
                    f"HeyGen video creation failed (status {response.status_code}): {response.text}"
                )
                return None
        except httpx.TimeoutException:
            logger.error("Timeout while creating HeyGen video.")
            return None
        except Exception as e:
            logger.error(f"Error creating HeyGen video: {e}")
            return None
    async def _poll_video_status(self, video_id: str) -> Optional[str]:
        """
        Poll HeyGen API for video generation completion.
        Args:
            video_id: The video ID to check status for.
        Returns:
            Video URL when complete, or None on timeout/failure.
        """
        headers = {
            "X-Api-Key": self.api_key,
        }
        for attempt in range(self.max_poll_attempts):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.get(
                        HEYGEN_VIDEO_STATUS_URL,
                        headers=headers,
                        params={"video_id": video_id},
                    )
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("data", {}).get("status", "")
                    if status == "completed":
                        video_url = result.get("data", {}).get("video_url")
                        return video_url
                    elif status == "failed":
                        error = result.get("data", {}).get("error", "Unknown error")
                        logger.error(f"Video generation failed: {error}")
                        return None
                    elif status in ("processing", "pending", "waiting"):
                        logger.debug(
                            f"Video still processing (attempt {attempt + 1}/{self.max_poll_attempts})..."
                        )
                    else:
                        logger.warning(f"Unknown video status: {status}")
                else:
                    logger.warning(
                        f"Status poll failed (status {response.status_code}): {response.text}"
                    )
            except Exception as e:
                logger.warning(f"Error polling video status: {e}")
            await asyncio.sleep(self.poll_interval)
        logger.error(f"Video generation timed out after {self.max_poll_attempts} polls.")
        return None
    async def get_available_avatars(self) -> list:
        """
        Get list of available avatars from HeyGen.
        Returns:
            List of avatar information dictionaries.
        """
        headers = {
            "X-Api-Key": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{HEYGEN_API_BASE}/v2/avatars",
                    headers=headers,
                )
            if response.status_code == 200:
                result = response.json()
                return result.get("data", {}).get("avatars", [])
            else:
                logger.error(f"Failed to fetch avatars: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error fetching avatars: {e}")
            return []