"""Base handler for Azure Voice Live API connections using the official SDK.

Provides the shared Voice Live connection, event processing, web client
audio handling with ambient mixing, and cleanup logic. Telephony subclasses
override on_message() and hook methods to implement protocol-specific behavior.
"""

import asyncio
import base64
import json
import logging
import time
import os
import aiohttp
from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import ManagedIdentityCredential
from azure.ai.voicelive.aio import connect as voicelive_connect
from azure.ai.voicelive.models import (
    AudioEchoCancellation,
    AudioNoiseReduction,
    AzureSemanticVad,
    AzureStandardVoice,
    InputAudioFormat,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerEventType,
    FunctionTool,
    FunctionCallOutputItem,
    ToolChoiceLiteral,
)

from .ambient_mixer import AmbientMixer

# Data type for WebSocket messages (str or bytes) sent to client
Data = Union[str, bytes]

logger = logging.getLogger(__name__)

# Default chunk size in bytes (100ms of audio at 24kHz, 16-bit mono)
DEFAULT_CHUNK_SIZE = 4800  # 24000 samples/sec * 0.1 sec * 2 bytes


ZENCORE_OOH_URL = os.getenv("ZENCORE_OOH_URL", "https://zencore.zenengineering.co.uk/api/ooh-callout")
OOH_INGEST_KEY = os.getenv("OOH_INGEST_KEY", "")

RAISE_CALLOUT_TOOL = FunctionTool(
    name="raise_callout",
    description="Log an out-of-hours callout into ZenCore and alert the on-call engineer. Call this once you have what's wrong and a phone number to reach the caller on.",
    parameters={
        "type": "object",
        "properties": {
            "issue": {"type": "string", "description": "What's wrong, in the caller's words."},
            "callbackNumber": {"type": "string", "description": "A phone number to call the caller back on."},
            "site": {"type": "string", "description": "The site, company or address NAME, as the caller gives it. Do NOT put the postcode here."},
            "postcode": {"type": "string", "description": "The site postcode, captured as its own answer and read back to confirm (e.g. 'HG1 5DG'). Send it here, separate from the site name."},
            "callerName": {"type": "string", "description": "The caller's name."},
            "reference": {"type": "string", "description": "Any site/customer reference or account number, if given."},
            "urgencyNote": {"type": "string", "description": "Anything the caller says about how urgent it is."},
        },
        "required": ["issue", "callbackNumber"],
    },
)


async def raise_callout(args: dict) -> dict:
    """POST the captured callout to ZenCore. Never raises — returns a dict."""
    payload = {
        "issue": args.get("issue", ""),
        "callbackNumber": args.get("callbackNumber", ""),
        "site": args.get("site", ""),
        "postcode": args.get("postcode", ""),
        "callerName": args.get("callerName", ""),
        "reference": args.get("reference", ""),
        "urgencyNote": args.get("urgencyNote", ""),
        "callStartedAt": datetime.now(timezone.utc).isoformat(),
    }
    headers = {"x-ooh-key": OOH_INGEST_KEY, "Content-Type": "application/json"}
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(ZENCORE_OOH_URL, json=payload, headers=headers) as r:
                data = await r.json()
                return data if isinstance(data, dict) else {"ok": False}
    except Exception as e:
        logger.error("raise_callout failed: %s", e)
        return {"ok": False, "error": str(e)}
    
class VoiceLiveMediaHandler:
    """"Handles the connection to Azure Voice Live API and web clients.

    Uses the azure-ai-voicelive SDK for typed session config, event handling,
    and audio streaming. Provides web client audio handling (raw PCM + ambient
    mixing) by default. Telephony subclasses override on_message() and hooks
    for their specific protocols.
    """

    def __init__(self, config):
        self.endpoint = config["AZURE_VOICE_LIVE_ENDPOINT"]
        self.model = config["VOICE_LIVE_MODEL"]    
        self.api_key = config["AZURE_VOICE_LIVE_API_KEY"]
        self.client_id = config["AZURE_USER_ASSIGNED_IDENTITY_CLIENT_ID"]

        # Voice Live audio format. Defaults to PCM16 (24kHz)
        self.input_audio_format = InputAudioFormat.PCM16
        self.output_audio_format = OutputAudioFormat.PCM16

        self.conn = None
        self._conn_ctx = None  # async context manager from SDK connect()
        self._credential = None  # kept alive for token refresh
        self._receiver_task = None
        self._voicelive_connected = False  # True while Voice Live WS is healthy

        self._turn_speech_stopped_ts: Optional[float] = None
        self._awaiting_first_audio = False
        self._turn_index = 0

        # Client WebSocket
        self.client_ws = None

        # TTS output buffering for continuous ambient mixing
        self._tts_output_buffer = bytearray()
        self._tts_buffer_lock = asyncio.Lock()
        self._max_buffer_size = 480000  # 10 seconds of audio
        self._buffer_warning_logged = False
        self._tts_playback_started = False
        self._min_buffer_to_start = 9600  # 200ms buffer before starting TTS playback

        # Ambient mixer initialization
        self._ambient_mixer: Optional[AmbientMixer] = None
        ambient_preset = config.get("AMBIENT_PRESET", "none")
        if ambient_preset and ambient_preset != "none":
            try:
                self._ambient_mixer = AmbientMixer(preset=ambient_preset)
            except Exception as e:
                logger.error(f"Failed to initialize AmbientMixer: {e}")

    async def _handle_raise_callout(self, event):
        try:
            args = json.loads(event.arguments or "{}")
        except Exception:
            args = {}
        result = await raise_callout(args)
        await self.on_callout_raised(result)
        model_output = {
            "ok": bool(result.get("ok")),
            "reference": result.get("calloutRef"),
            "status": ("Callout logged and the on-call engineer has been alerted."
                       if result.get("ok") else
                       "Could not log automatically; reassure the caller and an engineer will be contacted."),
        }
        item = FunctionCallOutputItem(call_id=event.call_id, output=json.dumps(model_output))
        await self.conn.conversation.item.create(item=item)
        await self.conn.response.create()

    def _session_config(self) -> RequestSession:
        """Return the typed session configuration for Voice Live."""
        return RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions="""You are the out-of-hours telephone assistant for Zen Engineering, a fire and security systems maintenance company. Callers reach you outside office hours because something has gone wrong on a site we look after — a fire alarm, intruder alarm, access control, emergency lighting or similar — and they need an engineer.

Your job is simple: answer warmly, find out what's wrong and how to reach them, and log the callout. You do NOT decide whether it's urgent, and you never ask them to wait until morning — every call is passed to the on-call engineer.

How to handle the call:
- Open with: "You've reached Zen Engineering's out-of-hours line. Can I take some details and get an engineer onto it?"
- Speak in plain UK English. Be calm, brief and reassuring — the caller may be stressed. Ask one thing at a time.
- Find out, roughly in this order:
  1. What's wrong — let them describe the fault in their own words.
  2. The site — ask in two short steps, not one run-on question:
     (a) The site name: "What's the name of the site or company?" Confirm it once; if they correct you, use their version. Put it in `site`.
     (b) Then the postcode on its own: "And what's the postcode there?" If they use the phonetic alphabet (Alpha, Bravo, Charlie…), convert it to the letters. Read it back ONCE to confirm — phonetically is fine ("Hotel, Golf, one, five, Delta, Golf") — and if they correct it use their version, then move on. Do NOT keep re-confirming. Put it in `postcode`, not `site`.
  3. A phone number to call them back on — read it back once, digit by digit ("zero, seven, seven, zero, zero…"), never as whole numbers like "seven hundred"; if they correct a digit, use it.
  4. Their name.
  5. Any site or account reference (don't push if they don't have one).
- The raise_callout tool is the ONLY thing that logs the callout and alerts the engineer. Nothing is logged until you actually call it — saying "it's logged" or "an engineer will be in touch" WITHOUT calling the tool leaves the caller with no help at all. So the moment you have the fault and a callback number (plus site name and postcode if given), CALL the raise_callout tool. Never ask permission to log — they rang for help, logging is the whole point.
- ONLY AFTER the raise_callout tool has actually returned may you tell the caller it's logged, the on-call engineer has been alerted, and will call them back shortly on the number they gave. If you have not called the tool yet, do not say any of that. Do NOT promise a specific arrival time.
- Don't get stuck confirming details — one read-back each is enough. If something's unclear, make one short attempt, then log what you have; a logged callout with a couple of imperfect details is far better than a call that logs nothing.
- Ask if there's anything else, then close politely.

Important rules:
- If the caller describes an actual fire, smoke, a gas leak, or anyone hurt or in danger, tell them immediately to hang up and dial 999 — then still log it for the engineer. Life safety comes first.
- Don't give technical advice or talk them through fixing anything — just capture and pass it on.
- Don't make anything up. If something's unclear, ask a short follow-up. If they genuinely can't give a callback number, still log the call with what you have.
- Keep it short — you're taking a message and reassuring them an engineer is coming, not having a long chat.""",
            tools=[RAISE_CALLOUT_TOOL],
            tool_choice=ToolChoiceLiteral.AUTO,
            turn_detection=AzureSemanticVad(),
            input_audio_format=self.input_audio_format,
            output_audio_format=self.output_audio_format,
            input_audio_noise_reduction=AudioNoiseReduction(type="azure_deep_noise_suppression"),
            input_audio_echo_cancellation=AudioEchoCancellation(),
            voice=AzureStandardVoice(name="en-GB-SoniaNeural", temperature=0.8),
        )

    # ------------------------------------------------------------------
    # Voice Live connection
    # ------------------------------------------------------------------

    async def connect_voicelive(self):
        """Connect to Azure Voice Live API using the SDK."""
        t0 = time.perf_counter()

        if self.client_id:
            self._credential = ManagedIdentityCredential(client_id=self.client_id)
            credential = self._credential
        else:
            credential = AzureKeyCredential(self.api_key)

        t1 = time.perf_counter()
        logger.info("[VoiceLive] Credential prepared in %.2fs", t1 - t0)

        self._conn_ctx = voicelive_connect(
            endpoint=self.endpoint,
            credential=credential,
            model=self.model.strip(),
        )
        self.conn = await self._conn_ctx.__aenter__()

        t2 = time.perf_counter()
        logger.info("[VoiceLive] SDK connected in %.2fs (total %.2fs)", t2 - t1, t2 - t0)
        logger.info("[VoiceLive] Session model=%s", self.model.strip())
        self._voicelive_connected = True

        await self.conn.session.update(session=self._session_config())
        await self.conn.response.create()

        self._receiver_task = asyncio.create_task(self._receiver_loop())

    async def send_audio(self, audio_b64: str):
        """Send PCM 24kHz 16-bit mono audio (base64) to Voice Live."""
        if not self._voicelive_connected:
            return
        await self.conn.input_audio_buffer.append(audio=audio_b64)

    async def _receiver_loop(self):
        """Receives typed events from Voice Live and dispatches to hook methods."""
        cancelled = False
        try:
            async for event in self.conn:
                event_type = event.type

                match event_type:
                    case ServerEventType.SESSION_CREATED:
                        session_id = event.session.id if hasattr(event, "session") else None
                        logger.info("[VoiceLive] Session ID: %s", session_id)

                    case ServerEventType.SESSION_UPDATED:
                        logger.info("[VoiceLive] Session updated")

                    case ServerEventType.INPUT_AUDIO_BUFFER_CLEARED:
                        logger.debug("[VoiceLive] Input audio buffer cleared")

                    case ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                        logger.info(
                            "[VoiceLive] Speech started at %s ms",
                            event.audio_start_ms,
                        )
                        await self.on_speech_started()

                    case ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
                        logger.info("[VoiceLive] Speech stopped")
                        self._turn_speech_stopped_ts = time.perf_counter()
                        self._awaiting_first_audio = True
                        self._turn_index += 1

                    case ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                        transcript = event.transcript
                        logger.debug("[VoiceLive] User: %s", transcript)
                        await self.on_user_transcript(transcript)

                    case ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED:
                        logger.warning(
                            "[VoiceLive] Transcription error: %s", event.error if hasattr(event, "error") else "unknown"
                        )

                    case ServerEventType.RESPONSE_AUDIO_DELTA:
                        delta = event.delta
                        if delta:
                            if self._awaiting_first_audio and self._turn_speech_stopped_ts is not None:
                                ttfa_ms = (time.perf_counter() - self._turn_speech_stopped_ts) * 1000.0
                                self._awaiting_first_audio = False
                                logger.info(
                                    "[Metric] ttfa model=%s turn=%d ttfa_ms=%.0f",
                                    self.model.strip(),
                                    self._turn_index,
                                    ttfa_ms,
                                )
                            await self.on_audio_delta(delta)

                    case ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
                        transcript = event.transcript
                        logger.debug("[VoiceLive] AI: %s", transcript)
                        await self.on_transcript_done(transcript)

                    case ServerEventType.RESPONSE_DONE:
                        response_id = event.response.id if hasattr(event, "response") else None
                        logger.info("[VoiceLive] Response done: id=%s", response_id)
                        await self.on_response_done()

                    case ServerEventType.ERROR:
                        logger.error("[VoiceLive] Error: %s", event.error)

                    case ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
                        await self._handle_raise_callout(event)
                    case _:
                        logger.debug("[VoiceLive] Event: %s", event_type)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            logger.exception("[VoiceLive] Receiver loop error")
        finally:
            self._voicelive_connected = False
            # If Voice Live dropped unexpectedly (not a normal cancellation),
            # close the client WebSocket so the caller-side loop exits cleanly.
            if not cancelled and self.client_ws:
                try:
                    logger.warning("[VoiceLive] Voice Live disconnected — closing client WebSocket")
                    await self.client_ws.close(1001)  # Going Away
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Client WebSocket
    # ------------------------------------------------------------------

    async def init_websocket(self, socket):
        """Sets up the client WebSocket."""
        self.client_ws = socket

    async def send_message(self, message: Data):
        """Sends data back to client WebSocket."""
        try:
            await self.client_ws.send(message)
        except Exception:
            logger.exception("[VoiceLive] Failed to send message to client")

    # ------------------------------------------------------------------
    # Hooks — web client implementations (override in telephony subclasses)
    # ------------------------------------------------------------------

    async def on_speech_started(self):
        """Barge-in: send StopAudio to client and clear TTS buffer."""
        stop_audio_data = {"Kind": "StopAudio", "AudioData": None, "StopAudio": {}}
        await self.send_message(json.dumps(stop_audio_data))

        if self._ambient_mixer is not None:
            async with self._tts_buffer_lock:
                self._tts_output_buffer.clear()
                self._tts_playback_started = False

    async def on_audio_delta(self, audio_bytes: bytes):
        """Handle audio from Voice Live — buffer for ambient or send directly."""
        if self._ambient_mixer is not None and self._ambient_mixer.is_enabled():
            async with self._tts_buffer_lock:
                self._tts_output_buffer.extend(audio_bytes)
                if len(self._tts_output_buffer) > self._max_buffer_size:
                    if not self._buffer_warning_logged:
                        logger.warning(
                            f"TTS buffer large: {len(self._tts_output_buffer)} bytes. "
                            "Speech may be delayed but will not be cut."
                        )
                        self._buffer_warning_logged = True
                elif self._buffer_warning_logged and len(self._tts_output_buffer) < self._max_buffer_size // 2:
                    self._buffer_warning_logged = False
        else:
            await self._send_audio_to_client(audio_bytes)

    async def on_transcript_done(self, transcript: str):
        """Forward transcript to client."""
        await self.send_message(
            json.dumps({"Kind": "Transcription", "Text": transcript})
        )

    async def on_response_done(self):
        """Called when Voice Live finishes a response. No-op by default."""
        pass
    async def on_user_transcript(self, transcript: str):
        """A completed caller transcript line. No-op by default; the OOH ACS
        handler overrides this to capture it."""
        pass

    async def on_callout_raised(self, result: dict):
        """Called right after raise_callout files a callout (result carries
        calloutRef). No-op by default; the OOH ACS handler overrides this."""
        pass

    # ------------------------------------------------------------------
    # Audio output to client
    # ------------------------------------------------------------------

    async def _send_audio_to_client(self, audio_bytes: bytes):
        """Send audio bytes to the client. Override in subclasses for wrapping."""
        await self.send_message(audio_bytes)

    # ------------------------------------------------------------------
    # Inbound audio from client
    # ------------------------------------------------------------------

    def _receive_audio_from_client(self, data) -> tuple:
        """Convert client audio to PCM 24kHz. Override for format conversion.

        Returns (pcm_bytes | None, chunk_size). Return None for silent frames.
        """
        return data, len(data)

    async def on_message(self, msg):
        """Process one incoming WebSocket message. Override in subclasses for protocol handling."""
        await self.handle_audio(msg)

    async def handle_audio(self, data):
        """Process inbound audio: convert, mix ambient, forward to Voice Live."""
        pcm_bytes, chunk_size = self._receive_audio_from_client(data)
        await self._send_continuous_audio(chunk_size)
        if pcm_bytes:
            audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
            await self.send_audio(audio_b64)

    # ------------------------------------------------------------------
    # Ambient mixing
    # ------------------------------------------------------------------

    async def _send_continuous_audio(self, chunk_size: int) -> None:
        """Send continuous audio (ambient + TTS if available) back to client."""
        if self._ambient_mixer is None or not self._ambient_mixer.is_enabled():
            return

        try:
            async with self._tts_buffer_lock:
                buffer_len = len(self._tts_output_buffer)
                ambient_bytes = self._ambient_mixer.get_ambient_only_chunk(chunk_size)

                should_play_tts = False
                if self._tts_playback_started:
                    if buffer_len >= chunk_size:
                        should_play_tts = True
                    elif buffer_len > 0:
                        should_play_tts = True
                    else:
                        self._tts_playback_started = False
                else:
                    if buffer_len >= self._min_buffer_to_start:
                        self._tts_playback_started = True
                        should_play_tts = True

                if should_play_tts and buffer_len >= chunk_size:
                    tts_chunk = bytes(self._tts_output_buffer[:chunk_size])
                    del self._tts_output_buffer[:chunk_size]

                    ambient = np.frombuffer(ambient_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    tts = np.frombuffer(tts_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    mixed = np.clip(ambient + tts, -0.95, 0.95)
                    output_bytes = (mixed * 32767).astype(np.int16).tobytes()

                elif should_play_tts and buffer_len > 0:
                    tts_chunk = bytes(self._tts_output_buffer[:])
                    self._tts_output_buffer.clear()
                    self._tts_playback_started = False

                    ambient = np.frombuffer(ambient_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    tts_samples = len(tts_chunk) // 2
                    tts = np.frombuffer(tts_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    ambient[:tts_samples] += tts
                    mixed = np.clip(ambient, -0.95, 0.95)
                    output_bytes = (mixed * 32767).astype(np.int16).tobytes()

                else:
                    output_bytes = ambient_bytes

            await self._send_audio_to_client(output_bytes)

        except Exception:
            logger.exception("[VoiceLive] Error in _send_continuous_audio")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Cancel background tasks and close the Voice Live connection."""
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receiver_task = None
        if self._conn_ctx:
            try:
                await self._conn_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._conn_ctx = None
            self.conn = None
        if self._credential:
            try:
                await self._credential.close()
            except Exception:
                pass
            self._credential = None
        logger.info("[VoiceLive] Cleaned up")
