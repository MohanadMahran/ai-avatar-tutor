"""
Helper utility functions used across the application.
Provides language detection, text processing, timing utilities,
and other shared functionality.
"""
import hashlib
import time
from typing import Callable, Any
from functools import wraps
from utils.logger import get_logger
logger = get_logger(__name__)
def detect_language(text: str) -> str:
    """
    Detect whether the input text is in Arabic or English.
    Uses character analysis and the langdetect library as fallback
    to determine the language of the input text.
    Args:
        text: Input text to analyze.
    Returns:
        Language code: 'ar' for Arabic, 'en' for English (default).
    """
    if not text or not text.strip():
        return "en"
    # Quick check: Arabic Unicode range detection
    arabic_chars = 0
    total_alpha = 0
    for char in text:
        if char.isalpha():
            total_alpha += 1
            if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or '\uFB50' <= char <= '\uFDFF' or '\uFE70' <= char <= '\uFEFF':
                arabic_chars += 1
    if total_alpha > 0 and (arabic_chars / total_alpha) > 0.3:
        return "ar"
    # Fallback to langdetect library
    try:
        from langdetect import detect
        detected = detect(text)
        if detected == "ar":
            return "ar"
        return "en"
    except Exception:
        return "en"
def compute_text_hash(text: str) -> str:
    """
    Compute a hash for text content for caching purposes.
    Args:
        text: Input text to hash.
    Returns:
        MD5 hex digest of the text.
    """
    normalized = text.strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()
def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with a suffix.
    Args:
        text: Input text to truncate.
        max_length: Maximum allowed length.
        suffix: Suffix to append when truncated.
    Returns:
        Truncated text string.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
def timing_decorator(func: Callable) -> Callable:
    """
    Decorator that logs the execution time of a function.
    Args:
        func: Function to wrap.
    Returns:
        Wrapped function with timing logging.
    """
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to a human-readable string.
    Args:
        size_bytes: File size in bytes.
    Returns:
        Human-readable size string (e.g., "2.5 MB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to remove potentially dangerous characters.
    Args:
        filename: Original filename.
    Returns:
        Sanitized filename safe for filesystem use.
    """
    import re
    # Remove path separators and null bytes
    sanitized = re.sub(r'[/\\:\0]', '', filename)
    # Remove leading dots
    sanitized = sanitized.lstrip('.')
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed_file"
    return sanitized
def chunk_text(text: str, chunk_size: int = 1000) -> list:
    """
    Split text into chunks of approximately equal size.
    Args:
        text: Input text to split.
        chunk_size: Target chunk size in characters.
    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Try to break at a sentence boundary
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            break_point = max(last_period, last_newline)
            if break_point > start:
                end = break_point + 1
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]