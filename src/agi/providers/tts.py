from __future__ import annotations

"""TTS (Text-to-Speech) provider — mirrors openclaw's TTS backends.

Backends:
  edge      — Microsoft Edge TTS (free, no API key)  pip install edge-tts
  openai    — OpenAI TTS API  (tts-1 / tts-1-hd / gpt-4o-mini-tts)
  elevenlabs — ElevenLabs API

Agent directive syntax (same as openclaw):
  [[tts:provider=edge voice=en-US-AriaNeural]]    ← inline directive
  [[tts:text]]Only speak this part.[[/tts:text]]  ← custom TTS text block
  [[tts:voice=en-GB-RyanNeural model=tts-1-hd]]   ← multiple params

Directives are stripped from the displayed text; the cleaned text (or
the [[tts:text]] block content) is passed to the TTS backend.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Directive parsing  (mirrors openclaw's parseTtsDirectives)
# ------------------------------------------------------------------

_BLOCK_RE = re.compile(r"\[\[tts:text\]\]([\s\S]*?)\[\[/tts:text\]\]", re.IGNORECASE)
_DIR_RE = re.compile(r"\[\[tts:([^\]]+)\]\]", re.IGNORECASE)


def parse_tts_directives(text: str) -> tuple[str, dict[str, str]]:
    """Parse [[tts:...]] directives. Returns (cleaned_text, overrides).

    overrides keys: provider, voice, model, tts_text
    """
    overrides: dict[str, str] = {}

    def _replace_block(m: re.Match) -> str:
        overrides["tts_text"] = m.group(1).strip()
        return ""

    cleaned = _BLOCK_RE.sub(_replace_block, text)

    def _replace_dir(m: re.Match) -> str:
        for token in m.group(1).split():
            if "=" in token:
                k, v = token.split("=", 1)
                key = k.strip().lower()
                val = v.strip()
                if key == "provider":
                    overrides["provider"] = val
                elif key in ("voice", "voiceid", "voice_id"):
                    overrides["voice"] = val
                elif key in ("model", "modelid", "model_id"):
                    overrides["model"] = val
        return ""

    cleaned = _DIR_RE.sub(_replace_dir, cleaned).strip()
    return cleaned, overrides


# ------------------------------------------------------------------
# TtsProvider
# ------------------------------------------------------------------

class TtsProvider:
    """Convert text to audio bytes (mp3/opus) using configured backend.

    provider: edge / openai / elevenlabs / none
    voice:    edge example  → en-US-AriaNeural
              openai example → alloy / echo / nova / shimmer / onyx / fable
              elevenlabs     → voice_id string
    model:    openai  → tts-1 / tts-1-hd / gpt-4o-mini-tts (default tts-1)
              elevenlabs → eleven_monolingual_v1 etc.
    """

    def __init__(
        self,
        provider: str = "edge",
        voice: str = "",
        api_key: str = "",
        model: str = "",
    ) -> None:
        self._provider = provider
        self._voice = voice
        self._api_key = api_key
        self._model = model

    async def speak(
        self,
        text: str,
        override_provider: str = "",
        override_voice: str = "",
        override_model: str = "",
    ) -> bytes | None:
        """Generate audio from text. Returns bytes or None if disabled."""
        provider = override_provider or self._provider
        voice = override_voice or self._voice
        model = override_model or self._model

        if not text or provider in ("none", "", "disabled"):
            return None

        try:
            if provider == "edge":
                return await self._edge(text, voice or "en-US-AriaNeural")
            elif provider == "openai":
                return await self._openai(text, voice or "alloy", model or "tts-1")
            elif provider == "elevenlabs":
                return await self._elevenlabs(text, voice, model)
            else:
                logger.warning("Unknown TTS provider: %s", provider)
                return None
        except Exception as e:
            logger.warning("TTS failed (%s): %s", provider, e)
            return None

    # ------------------------------------------------------------------
    # Edge TTS  (free — uses Microsoft Edge cloud voices)
    # ------------------------------------------------------------------

    async def _edge(self, text: str, voice: str) -> bytes:
        try:
            import edge_tts
        except ImportError:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

        communicate = edge_tts.Communicate(text, voice)
        chunks: list[bytes] = []
        async for item in communicate.stream():
            if item["type"] == "audio":
                chunks.append(item["data"])
        if not chunks:
            raise RuntimeError("Edge TTS returned no audio")
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # OpenAI TTS
    # ------------------------------------------------------------------

    async def _openai(self, text: str, voice: str, model: str) -> bytes:
        import os
        import httpx

        api_key = self._api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        base_url = (os.getenv("OPENAI_TTS_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url}/audio/speech",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "input": text,
                    "voice": voice,
                    "response_format": "mp3",
                },
            )
            resp.raise_for_status()
            return resp.content

    # ------------------------------------------------------------------
    # ElevenLabs TTS
    # ------------------------------------------------------------------

    async def _elevenlabs(self, text: str, voice_id: str, model_id: str) -> bytes:
        import os
        import httpx

        api_key = self._api_key or os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")
        if not voice_id:
            raise RuntimeError("voice (voice_id) required for ElevenLabs")

        model = model_id or "eleven_monolingual_v1"
        base_url = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").rstrip("/")

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url}/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
                json={
                    "text": text,
                    "model_id": model,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True,
                    },
                },
            )
            resp.raise_for_status()
            return resp.content


# ------------------------------------------------------------------
# Helper: list available Edge voices (useful for config discovery)
# ------------------------------------------------------------------

async def list_edge_voices(locale: str = "") -> list[dict[str, Any]]:
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        if locale:
            voices = [v for v in voices if v.get("Locale", "").startswith(locale)]
        return voices
    except ImportError:
        return []
