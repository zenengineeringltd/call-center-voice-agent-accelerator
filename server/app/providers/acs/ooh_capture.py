"""OOH transcript + recording capture for the ACS provider.

Posts to ZenCore's /api/ooh-callout-recording, which MERGES fields — so the
transcript (ready at hang-up) and the recording URL (ready later) go as two
separate POSTs. This file does the transcript half; recording is added next.

One in-flight OOH call at a time (out-of-hours volume). If you ever need
concurrent OOH calls, key this state by callConnectionId instead.
"""

import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

ZENCORE_OOH_RECORDING_URL = os.getenv(
    "ZENCORE_OOH_RECORDING_URL",
    "https://zencore.zenengineering.co.uk/api/ooh-callout-recording",
)
OOH_INGEST_KEY = os.getenv("OOH_INGEST_KEY", "")

_callout_ref = ""
_lines: list[str] = []
_flushed = False


def reset() -> None:
    """Clear state at the start of a call."""
    global _callout_ref, _lines, _flushed
    _callout_ref = ""
    _lines = []
    _flushed = False


def set_callout_ref(ref: str) -> None:
    global _callout_ref
    _callout_ref = ref or ""


def add_caller(text: str) -> None:
    t = (text or "").strip()
    if t:
        _lines.append(f"Caller: {t}")


def add_agent(text: str) -> None:
    t = (text or "").strip()
    if t:
        _lines.append(f"Agent: {t}")


async def _post(body: dict) -> None:
    headers = {"x-ooh-key": OOH_INGEST_KEY, "Content-Type": "application/json"}
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(ZENCORE_OOH_RECORDING_URL, json=body, headers=headers) as r:
                if r.status >= 300:
                    logger.warning("ooh recording POST -> %s", r.status)
    except Exception as e:
        logger.warning("ooh recording POST failed: %s", e)


async def flush_transcript() -> None:
    """POST the transcript to ZenCore at call end. Safe to call once."""
    global _flushed
    if _flushed:
        return
    _flushed = True
    ref = _callout_ref
    text = "\n".join(_lines).strip()
    if ref and text:
        await _post({"calloutRef": ref, "transcript": text})
        logger.info("ooh transcript posted for %s (%d lines)", ref, len(_lines))