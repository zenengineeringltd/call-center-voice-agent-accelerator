"""OOH outbound escalation — keep ringing until answered + accept on the phone.

Kicked off from ACSMediaHandler.on_callout_raised once raise_callout returns a
calloutRef. Rings the on-call engineer; if they press 1 it accepts the callout in
ZenCore and stops; if they don't answer (voicemail/no key) it redials, then drops
to the next rung of the ladder. Polls ZenCore so it also stops the moment the
callout is accepted via the email/SMS link.

Reuses the provider's existing /acs/callbacks/<id> route: outbound-call events
come back there and process_callback_events hands each one to
handle_callback_event(), which owns it only if the call is one of ours.

ACS has no reliable answering-machine detection, so we require a keypress —
voicemail never presses 1, so a no-input timeout IS the "not answered" signal.

Env (Container App settings):
  ACS_CONNECTION_STRING   the ACS resource (already set)
  ACS_SOURCE_NUMBER       the FIXED ACS number the AI owns, E.164 (+44…) — the
                          outbound caller id. NOT the rotating engineer; the
                          weekly rotation is the ZenCore on-call rota.
  OOH_PUBLIC_BASE_URL     this app's public https base (the Container App ingress)
  OOH_INGEST_KEY          shared secret (already set)
  ZENCORE_OOH_STATUS_URL / ZENCORE_OOH_ACCEPT_URL  (default to prod)

Written for azure-communication-callautomation 1.5.0: create_call takes the target
identifier directly (no CallInvite) and returns CallConnectionProperties.
"""

import asyncio
import logging
import os
import uuid

import aiohttp
from azure.communication.callautomation.aio import CallAutomationClient
from azure.communication.callautomation import (
    PhoneNumberIdentifier,
    TextSource,
    RecognizeInputType,
)

logger = logging.getLogger(__name__)

ACS_CONNECTION_STRING = os.getenv("ACS_CONNECTION_STRING", "")
ACS_SOURCE_NUMBER     = os.getenv("ACS_SOURCE_NUMBER", "")
PUBLIC_BASE_URL       = os.getenv("OOH_PUBLIC_BASE_URL", os.getenv("ACS_DEV_TUNNEL", "")).rstrip("/")
VOICE                 = os.getenv("OOH_TTS_VOICE", "en-GB-SoniaNeural")

ZENCORE_OOH_STATUS_URL = os.getenv("ZENCORE_OOH_STATUS_URL", "https://zencore.zenengineering.co.uk/api/ooh-callout-status")
ZENCORE_OOH_ACCEPT_URL = os.getenv("ZENCORE_OOH_ACCEPT_URL", "https://zencore.zenengineering.co.uk/api/ooh-callout-accept")
OOH_INGEST_KEY         = os.getenv("OOH_INGEST_KEY", "")

MAX_ATTEMPTS_PER_RUNG = int(os.getenv("OOH_MAX_ATTEMPTS_PER_RUNG", "2"))
REDIAL_DELAY_SECONDS  = int(os.getenv("OOH_REDIAL_DELAY_SECONDS", "20"))
ACCEPT_KEY            = os.getenv("OOH_ACCEPT_KEY", "1")

_client: CallAutomationClient | None = None


def _acs() -> CallAutomationClient:
    global _client
    if _client is None:
        _client = CallAutomationClient.from_connection_string(ACS_CONNECTION_STRING)
    return _client


async def _zc_status(ref: str) -> dict:
    headers = {"x-ooh-key": OOH_INGEST_KEY}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
        async with s.get(ZENCORE_OOH_STATUS_URL, params={"ref": ref}, headers=headers) as r:
            return await r.json()


async def _zc_accept(ref: str, by: str = "") -> dict:
    headers = {"x-ooh-key": OOH_INGEST_KEY, "Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
        async with s.post(ZENCORE_OOH_ACCEPT_URL, json={"calloutRef": ref, "by": by}, headers=headers) as r:
            return await r.json()


class _Esc:
    def __init__(self, ref: str, ladder: list[dict], site: str = ""):
        self.ref = ref
        self.ladder = ladder          # [{role, name, number}, …]
        self.site = site
        self.rung = 0
        self.attempts = 0
        self.ccid: str | None = None
        self.awaiting = False
        self.done = False


ACTIVE: dict[str, _Esc] = {}          # calloutRef -> _Esc
_BY_CCID: dict[str, str] = {}         # call_connection_id -> calloutRef


def _prompt(esc: _Esc) -> str:
    where = f" at {esc.site}" if esc.site else ""
    return (f"This is the Zen out of hours line. A callout has come in{where}. "
            f"Press {ACCEPT_KEY} to accept it, or hang up and it will be passed on.")


async def start(callout_ref: str) -> None:
    """Begin the ladder. Fire-and-forget from on_callout_raised."""
    if not callout_ref or callout_ref in ACTIVE:
        return
    if not ACS_SOURCE_NUMBER or not PUBLIC_BASE_URL:
        logger.warning("ooh escalation not configured (ACS_SOURCE_NUMBER / OOH_PUBLIC_BASE_URL)")
        return
    try:
        status = await _zc_status(callout_ref)
    except Exception as e:                      # noqa: BLE001
        logger.warning("ooh status fetch failed for %s: %s", callout_ref, e)
        return
    if status.get("acknowledged"):
        return
    # Normalise every number to E.164 (+44…) up front. ACS create_call can't
    # route a number stored as 07… or with spaces, and fails the call — so a
    # number saved in national form is exactly the "ringing rung 0 then
    # CreateCallFailed" symptom. _e164 fixes the common UK forms; a number it
    # can't fix is dropped rather than dialled into a guaranteed failure.
    ladder = []
    for c in status.get("escalation", []):
        num = _e164(c.get("number"))
        if num:
            ladder.append({**c, "number": num})
    if not ladder:
        logger.warning("ooh %s: empty ladder — office already emailed", callout_ref)
        return
    esc = _Esc(callout_ref, ladder, site=status.get("site", ""))
    ACTIVE[callout_ref] = esc
    logger.info("ooh %s: escalation started, %d rung(s)", callout_ref, len(ladder))
    await _dial(esc)


async def _dial(esc: _Esc) -> None:
    if esc.done:
        return
    try:
        if (await _zc_status(esc.ref)).get("acknowledged"):
            return _finish(esc)
    except Exception:                            # noqa: BLE001
        pass
    if esc.rung >= len(esc.ladder):
        logger.warning("ooh %s: ladder exhausted — leaving to the office backstop", esc.ref)
        return _finish(esc)

    target = esc.ladder[esc.rung]
    cb_url = f"{PUBLIC_BASE_URL}/acs/callbacks/{uuid.uuid4()}"
    try:
        # SDK 1.5.0: the target is a CommunicationIdentifier directly (no
        # CallInvite), the caller id is a keyword, and create_call returns a
        # CallConnectionProperties (so the id is .call_connection_id, not nested).
        result = await _acs().create_call(
            target_participant=PhoneNumberIdentifier(target["number"]),
            callback_url=cb_url,
            source_caller_id_number=PhoneNumberIdentifier(ACS_SOURCE_NUMBER),
            operation_context=f"ooh:{esc.ref}:{esc.rung}",
        )
        esc.ccid = result.call_connection_id
        _BY_CCID[esc.ccid] = esc.ref
        esc.awaiting = True
        logger.info("ooh %s: ringing rung %d (%s) ccid=%s",
                    esc.ref, esc.rung, target.get("name") or target["number"], esc.ccid)
    except Exception as e:                        # noqa: BLE001
        logger.warning("ooh %s: create_call failed: %s", esc.ref, e)
        await _advance(esc)


async def handle_callback_event(event: dict) -> bool:
    """Handle one ACS callback event IF it belongs to an escalation call.
    Returns True when it owned the event (caller should skip its own handling)."""
    data = event.get("data") or {}
    ccid = data.get("callConnectionId") or ""
    ref = _BY_CCID.get(ccid)
    if not ref:
        return False
    esc = ACTIVE.get(ref)
    if not esc or esc.done:
        return True

    etype = (event.get("type") or "").split(".")[-1]
    conn = _acs().get_call_connection(ccid)

    if etype == "CallConnected":
        try:
            await conn.start_recognizing_media(
                input_type=RecognizeInputType.DTMF,
                target_participant=PhoneNumberIdentifier(esc.ladder[esc.rung]["number"]),
                play_prompt=TextSource(text=_prompt(esc), voice_name=VOICE),
                initial_silence_timeout=12,
                dtmf_max_tones_to_collect=1,
                interrupt_prompt=True,
                operation_context=f"ooh:{esc.ref}:{esc.rung}",
            )
        except Exception as e:                     # noqa: BLE001
            logger.warning("ooh %s: recognize start failed: %s", esc.ref, e)
            await _end_attempt(esc, conn, accepted=False)

    elif etype == "RecognizeCompleted":
        tones = ((data.get("dtmfResult") or {}).get("tones")) or []
        pressed = any(str(t).lower() in (ACCEPT_KEY, _tone_word(ACCEPT_KEY)) for t in tones)
        if pressed:
            who = esc.ladder[esc.rung].get("name") or ""
            try:
                await _zc_accept(esc.ref, by=who)
            except Exception as e:                 # noqa: BLE001
                logger.warning("ooh %s: accept POST failed: %s", esc.ref, e)
            try:
                await conn.play_media_to_all(TextSource(text="Thanks, that's booked to you. Goodbye.", voice_name=VOICE))
            except Exception:                      # noqa: BLE001
                pass
            esc.awaiting = False
            _finish(esc)
            await _hang_up(conn)
        else:
            await _end_attempt(esc, conn, accepted=False)

    elif etype in ("RecognizeFailed", "PlayFailed"):
        await _end_attempt(esc, conn, accepted=False)

    elif etype == "CreateCallFailed":
        # The outbound call never connected (bad/unroutable number, or the source
        # number isn't allowed to dial this destination). Log ACS's own reason so
        # a persistent failure is diagnosable, then treat it as a failed attempt
        # and move on rather than leaving the ladder hung.
        info = data.get("resultInformation") or {}
        logger.warning("ooh %s: CreateCallFailed rung %d code=%s subCode=%s message=%s",
                       esc.ref, esc.rung, info.get("code"), info.get("subCode"), info.get("message"))
        esc.awaiting = False
        _BY_CCID.pop(ccid, None)
        await _advance(esc)

    elif etype == "CallDisconnected":
        if esc.awaiting:
            esc.awaiting = False
            _BY_CCID.pop(ccid, None)
            await _advance(esc)

    return True


async def _end_attempt(esc: _Esc, conn, accepted: bool) -> None:
    esc.awaiting = False
    await _hang_up(conn)
    if esc.ccid:
        _BY_CCID.pop(esc.ccid, None)
    if not accepted:
        await _advance(esc)


async def _advance(esc: _Esc) -> None:
    if esc.done:
        return
    esc.attempts += 1
    if esc.attempts >= MAX_ATTEMPTS_PER_RUNG:
        esc.rung += 1
        esc.attempts = 0
    await asyncio.sleep(REDIAL_DELAY_SECONDS)
    await _dial(esc)


def _finish(esc: _Esc) -> None:
    esc.done = True
    ACTIVE.pop(esc.ref, None)
    if esc.ccid:
        _BY_CCID.pop(esc.ccid, None)


async def _hang_up(conn) -> None:
    try:
        await conn.hang_up(is_for_everyone=True)
    except Exception:                              # noqa: BLE001
        pass


def _tone_word(digit: str) -> str:
    return {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}.get(digit, digit)


def _e164(number: str) -> str:
    """Best-effort UK E.164 normalisation. ACS needs +44…; a number stored as
    07… or with spaces/brackets won't route and create_call fails. Keeps an
    already-+ number as-is, maps 0044/44/0 prefixes to +44, and returns "" for
    anything left that clearly isn't a dialable number."""
    raw = str(number or "").strip()
    n = "+" + "".join(ch for ch in raw if ch.isdigit()) if raw.startswith("+") \
        else "".join(ch for ch in raw if ch.isdigit())
    if not n or n == "+":
        return ""
    if n.startswith("+"):
        return n
    if n.startswith("0044"):
        return "+" + n[2:]
    if n.startswith("44"):
        return "+" + n
    if n.startswith("0"):
        return "+44" + n[1:]
    return n
