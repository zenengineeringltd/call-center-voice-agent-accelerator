"""OOH call recording for the ACS provider.

Records the inbound out-of-hours call, then — after hang-up, once ACS has
finalised the audio — downloads it, stores it in blob storage, mints a
time-limited SAS link, and POSTs that link to ZenCore's
/api/ooh-callout-recording, which merges it onto the callout next to the
transcript (ooh_capture does the transcript half).

Flow / wiring (one in-flight OOH call at a time, matching ooh_capture):
  • start(ccid)        — on the inbound CallConnected (event_handler): starts the
                         recording and remembers its recording_id.
  • note_ref(ref)      — on raise_callout (media_handler.on_callout_raised): binds
                         the in-flight recording_id to the callout ref, keyed BY
                         recording_id so it survives past the call (the recording
                         only finalises after hang-up, maybe after the next call
                         has started).
  • on_recording_ready — on the ACS RecordingFileStatusUpdated Event Grid event:
                         downloads the finalised file, stores it, POSTs the link.

Everything here is best-effort: a storage or recording hiccup must never affect
the live call or the callout. Needs azure-storage-blob (imported lazily so a
missing dependency can't crash app start-up) and AZURE_STORAGE_CONNECTION_STRING;
with no storage connection string set, it simply doesn't record.
"""

import datetime as dt
import inspect
import logging
import os

import aiohttp
from azure.communication.callautomation.aio import CallAutomationClient
from azure.communication.callautomation import (
    RecordingContent,
    RecordingChannel,
    RecordingFormat,
)

logger = logging.getLogger(__name__)

ACS_CONNECTION_STRING     = os.getenv("ACS_CONNECTION_STRING", "")
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
RECORDING_CONTAINER       = os.getenv("OOH_RECORDING_CONTAINER", "ooh-recordings")
SAS_VALID_DAYS            = int(os.getenv("OOH_RECORDING_SAS_DAYS", "30"))

ZENCORE_OOH_RECORDING_URL = os.getenv("ZENCORE_OOH_RECORDING_URL", "https://zencore.zenengineering.co.uk/api/ooh-callout-recording")
OOH_INGEST_KEY            = os.getenv("OOH_INGEST_KEY", "")

_client: CallAutomationClient | None = None
_recording_id = ""                  # the in-flight call's recording id (set at start)
_PENDING: dict[str, str] = {}       # recording_id -> callout_ref (survives past hang-up)


def _acs() -> CallAutomationClient:
    global _client
    if _client is None:
        _client = CallAutomationClient.from_connection_string(ACS_CONNECTION_STRING)
    return _client


async def start(call_connection_id: str) -> None:
    """Start recording the inbound call (best-effort). No storage configured → skip."""
    global _recording_id
    _recording_id = ""
    if not call_connection_id or not ACS_CONNECTION_STRING:
        return
    if not STORAGE_CONNECTION_STRING:
        logger.info("ooh recording: AZURE_STORAGE_CONNECTION_STRING unset — not recording this call")
        return
    try:
        # SDK 1.5.0: start by call_connection_id (no ServerCallLocator needed); the
        # *_type kwargs are the current names. Mixed mono MP3 is fine for a voice note.
        result = await _acs().start_recording(
            call_connection_id=call_connection_id,
            recording_content_type=RecordingContent.AUDIO,
            recording_channel_type=RecordingChannel.MIXED,
            recording_format_type=RecordingFormat.MP3,
        )
        _recording_id = getattr(result, "recording_id", "") or ""
        logger.info("ooh recording started rec=%s ccid=%s", _recording_id, call_connection_id)
    except Exception as e:                           # noqa: BLE001 — transcript still lands
        logger.warning("ooh start_recording failed for %s: %s", call_connection_id, e)


def note_ref(callout_ref: str) -> None:
    """Bind the in-flight recording to the callout ref (called when raise_callout
    returns). Keyed by recording_id so the later recording-ready event can find it."""
    if _recording_id and callout_ref:
        _PENDING[_recording_id] = callout_ref


async def on_recording_ready(event: dict) -> None:
    """ACS RecordingFileStatusUpdated: download the finalised audio, store it, and
    attach a shareable SAS link to the callout."""
    data = event.get("data") or {}
    rec_id = data.get("recordingId") or _rec_id_from_subject(event.get("subject", ""))
    ref = _PENDING.pop(rec_id or "", "")
    if not ref:
        logger.warning("ooh recording ready but no callout bound (rec=%s)", rec_id)
        return
    chunks = ((data.get("recordingStorageInfo") or {}).get("recordingChunks")) or []
    content_location = chunks[0].get("contentLocation") if chunks else ""
    if not content_location:
        logger.warning("ooh recording ready with no contentLocation (rec=%s)", rec_id)
        return
    try:
        audio = await _download(content_location)
        url = await _upload_and_sign(f"{ref}-{rec_id or 'rec'}.mp3", audio)
        await _post(ref, url)
        logger.info("ooh recording attached to %s (%d bytes)", ref, len(audio))
    except Exception as e:                           # noqa: BLE001
        logger.warning("ooh recording handling failed for %s: %s", ref, e)


def _rec_id_from_subject(subject: str) -> str:
    # subject: /recording/call/{serverCallId}/recordingId/{recordingId}
    marker = "/recordingId/"
    return subject.split(marker, 1)[1] if marker in subject else ""


async def _download(content_location: str) -> bytes:
    # download_recording(recording_url) → bytes/iterable; the aio client's exact
    # return shape varies by version, so cope with awaitable / async-iter / sync.
    res = _acs().download_recording(content_location)
    if inspect.isawaitable(res):
        res = await res
    if hasattr(res, "__aiter__"):
        buf = bytearray()
        async for chunk in res:
            buf.extend(chunk)
        return bytes(buf)
    if hasattr(res, "read"):
        data = res.read()
        if inspect.isawaitable(data):
            data = await data
        return bytes(data)
    buf = bytearray()
    for chunk in res:
        buf.extend(chunk)
    return bytes(buf)


async def _upload_and_sign(blob_name: str, data: bytes) -> str:
    # Imported lazily so a missing azure-storage-blob can't break module import.
    from azure.storage.blob.aio import BlobServiceClient
    from azure.storage.blob import generate_blob_sas, BlobSasPermissions
    svc = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    async with svc:
        container = svc.get_container_client(RECORDING_CONTAINER)
        try:
            await container.create_container()
        except Exception:                            # noqa: BLE001 — already exists
            pass
        await container.upload_blob(name=blob_name, data=data, overwrite=True, content_type="audio/mpeg")
        sas = generate_blob_sas(
            account_name=svc.account_name,
            container_name=RECORDING_CONTAINER,
            blob_name=blob_name,
            account_key=svc.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=SAS_VALID_DAYS),
        )
        return f"{container.url}/{blob_name}?{sas}"


async def _post(callout_ref: str, recording_url: str) -> None:
    headers = {"x-ooh-key": OOH_INGEST_KEY, "Content-Type": "application/json"}
    body = {"calloutRef": callout_ref, "recordingUrl": recording_url}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
            async with s.post(ZENCORE_OOH_RECORDING_URL, json=body, headers=headers) as r:
                if r.status >= 300:
                    logger.warning("ooh recording POST -> %s", r.status)
    except Exception as e:                           # noqa: BLE001
        logger.warning("ooh recording POST failed: %s", e)
