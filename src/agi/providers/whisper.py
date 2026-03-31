from __future__ import annotations

"""Whisper voice transcription provider.

Backends (tried in order based on config):
  1. openai-whisper  — local model via whisper package
  2. faster-whisper  — faster local inference
  3. Groq API        — fast cloud (free tier available)
  4. OpenAI API      — fallback cloud
"""

import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WhisperProvider:
    def __init__(self, backend: str = "auto", model: str = "base", api_key: str = "") -> None:
        """
        backend: auto / local / groq / openai
        model:   for local: tiny/base/small/medium/large
                 for cloud: whisper-1 / whisper-large-v3
        """
        self._backend = backend
        self._model = model
        self._api_key = api_key
        self._local_model: Any = None

    async def transcribe(self, audio_data: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe audio bytes to text."""
        backend = self._backend
        if backend == "auto":
            backend = await self._detect_backend()

        if backend in ("local", "faster"):
            return await self._transcribe_local(audio_data, mime_type)
        elif backend == "groq":
            return await self._transcribe_groq(audio_data, mime_type)
        elif backend == "openai":
            return await self._transcribe_openai(audio_data, mime_type)
        else:
            logger.warning("No whisper backend available")
            return ""

    async def _detect_backend(self) -> str:
        # Prefer cloud if API key available (faster, no GPU needed)
        import os
        if self._api_key or os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        # Fall back to local
        try:
            import whisper  # noqa: F401
            return "local"
        except ImportError:
            pass
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            return "faster"
        except ImportError:
            pass
        return "none"

    async def _transcribe_local(self, audio_data: bytes, mime_type: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._local_transcribe_sync, audio_data, mime_type)

    def _local_transcribe_sync(self, audio_data: bytes, mime_type: str) -> str:
        try:
            import whisper
            if self._local_model is None:
                self._local_model = whisper.load_model(self._model)

            ext = _mime_to_ext(mime_type)
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(audio_data)
                tmp_path = f.name

            result = self._local_model.transcribe(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            return result["text"].strip()
        except Exception as e:
            logger.warning("Local whisper failed: %s", e)
            return ""

    async def _transcribe_groq(self, audio_data: bytes, mime_type: str) -> str:
        import os
        import httpx

        api_key = self._api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return ""

        ext = _mime_to_ext(mime_type)
        filename = f"audio{ext}"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (filename, audio_data, mime_type)},
                data={"model": "whisper-large-v3-turbo", "response_format": "text"},
            )
            resp.raise_for_status()
            return resp.text.strip()

    async def _transcribe_openai(self, audio_data: bytes, mime_type: str) -> str:
        import os
        import httpx

        api_key = self._api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return ""

        ext = _mime_to_ext(mime_type)
        filename = f"audio{ext}"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (filename, audio_data, mime_type)},
                data={"model": "whisper-1", "response_format": "text"},
            )
            resp.raise_for_status()
            return resp.text.strip()


def _mime_to_ext(mime: str) -> str:
    return {
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/wav": ".wav",
        "audio/webm": ".webm",
        "audio/flac": ".flac",
    }.get(mime.split(";")[0].strip(), ".ogg")
