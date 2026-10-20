"""Handles ACS (Azure Communication Services) clients via JSON-wrapped audio."""

import base64
import json
import logging

from app.handler.voicelive_media_handler import DEFAULT_CHUNK_SIZE, VoiceLiveMediaHandler
from . import ooh_capture

logger = logging.getLogger(__name__)


class ACSMediaHandler(VoiceLiveMediaHandler):
    """Bridges ACS Call Automation WebSocket to Voice Live.

    Overrides only the JSON wrapping/unwrapping; ambient mixing and Voice Live
    connection are inherited from the base class.
    """
    def __init__(self, config):
        super().__init__(config)
        ooh_capture.reset()          # fresh capture state for this call

    async def on_user_transcript(self, transcript: str):
        ooh_capture.add_caller(transcript)

    async def on_transcript_done(self, transcript: str):
        await super().on_transcript_done(transcript)
        ooh_capture.add_agent(transcript)

    async def on_callout_raised(self, result: dict):
        ooh_capture.set_callout_ref(result.get("calloutRef") if isinstance(result, dict) else "")

    async def cleanup(self):
        try:
            await ooh_capture.flush_transcript()
        except Exception:
            logger.exception("[ACSMediaHandler] transcript flush failed")
        await super().cleanup()
    # ------------------------------------------------------------------
    # Audio output — wrap in ACS JSON protocol
    # ------------------------------------------------------------------

    async def _send_audio_to_client(self, audio_bytes: bytes):
        """Wrap audio in ACS AudioData JSON format before sending."""
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        data = {
            "Kind": "AudioData",
            "AudioData": {"Data": audio_b64},
            "StopAudio": None,
        }
        await self.send_message(json.dumps(data))

    # ------------------------------------------------------------------
    # Inbound audio — parse ACS JSON protocol
    # ------------------------------------------------------------------

    def _receive_audio_from_client(self, data) -> tuple:
        """Parse ACS JSON and extract PCM audio bytes."""
        try:
            msg = json.loads(data)
            if msg.get("kind") == "AudioData":
                audio_data = msg.get("audioData", {})
                incoming_data = audio_data.get("data", "")

                if incoming_data:
                    pcm_bytes = base64.b64decode(incoming_data)
                    chunk_size = len(pcm_bytes)
                else:
                    pcm_bytes = None
                    chunk_size = DEFAULT_CHUNK_SIZE

                if audio_data.get("silent", True):
                    return None, chunk_size
                return pcm_bytes, chunk_size
        except Exception:
            logger.exception("[ACSMediaHandler] Error parsing ACS audio")
        return None, DEFAULT_CHUNK_SIZE
