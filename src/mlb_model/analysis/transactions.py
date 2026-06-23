"""Authoritative roster-status verification from the MLB transactions feed.

The transactions endpoint is the system of record for *why* a player isn't
producing: it carries the official "placed on the 10-day injured list" /
"activated from the injured list" / "optioned to Triple-A" status changes,
each with a date and (for IL moves) the injury itself in plain text:

    "New York Yankees placed RF Aaron Judge on the 10-day injured list
     retroactive to June 2, 2026. Right rib stress fracture."

We replay a player's status changes in order to derive their *current*
state — on the IL (verified injury), optioned / DFA'd / released / suspended
(verified external factor), or neither (unverified → the caller reports it as
UNCLEAR). This is deliberately stricter than headline keyword-matching: a
cold streak is only ever labelled injury/external when MLB itself logged the
move.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

import statsapi

from mlb_model.logging import get_logger

log = get_logger("analysis.transactions")

_IL_PLACED = re.compile(r"on the (\d+)-day injured list", re.IGNORECASE)
_IL_TRANSFER = re.compile(r"to the (\d+)-day injured list", re.IGNORECASE)
_RETRO = re.compile(r"retroactive to ([A-Z][a-z]+ \d{1,2}, \d{4})", re.IGNORECASE)


@dataclass(slots=True)
class StatusVerdict:
    """Verified current roster status for one player, derived from transactions."""

    cause: str  # "injury-verified" | "external-verified" | "unverified"
    il_type: str | None = None          # e.g. "10-day"
    il_start: date | None = None        # effective / retroactive start
    injury_note: str | None = None      # e.g. "Right rib stress fracture"
    external_move: str | None = None     # e.g. "optioned to Triple-A"
    move_date: date | None = None

    @property
    def is_verified(self) -> bool:
        return self.cause != "unverified"


def fetch_team_transactions(team_id: int, season: int) -> list[dict]:
    """All transactions for a team across the season window; [] on failure."""
    try:
        res = statsapi.get(
            "transactions",
            {
                "teamId": int(team_id),
                "startDate": f"{int(season)}-01-01",
                "endDate": f"{int(season)}-12-31",
            },
        )
        return res.get("transactions", []) or []
    except Exception:  # noqa: BLE001 -- best-effort; verification degrades to UNCLEAR
        log.warning("transactions.fetch.failed", team_id=int(team_id), season=int(season))
        return []


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _injury_note(description: str) -> str | None:
    """The trailing clause after the IL sentence is the injury itself."""
    # "...injured list[ retroactive to <date>]. <Injury note>."
    idx = description.lower().find("injured list")
    if idx == -1:
        return None
    tail = description[idx + len("injured list"):]
    # Drop the optional "retroactive to <date>." then take what remains.
    tail = re.sub(r"^[^.]*\.", "", tail, count=1).strip() if "." in tail else ""
    note = tail.strip(" .")
    return note or None


def verify_status(
    transactions: list[dict], player_id: int, *, as_of: date | None = None
) -> StatusVerdict:
    """Replay a player's status changes to derive current verified status."""
    as_of = as_of or date.today()
    mine = [t for t in transactions if (t.get("person") or {}).get("id") == player_id]

    def _tx_date(t: dict) -> date:
        return _parse_date(t.get("effectiveDate") or t.get("date")) or date.min

    mine.sort(key=_tx_date)

    il: dict | None = None
    ext: dict | None = None
    for t in mine:
        desc = (t.get("description") or "")
        low = desc.lower()
        tdate = _tx_date(t)
        if tdate > as_of:
            continue

        placed = _IL_PLACED.search(desc)
        if "injured list" in low and ("placed" in low or "transferred" in low):
            m = placed or _IL_TRANSFER.search(desc)
            il_type = f"{m.group(1)}-day" if m else (il["il_type"] if il else "IL")
            retro = _RETRO.search(desc)
            start = _parse_date(retro.group(1)) if retro else tdate
            # A transfer (e.g. 15-day → 60-day) keeps the original start.
            if "transferred" in low and il:
                start = il["il_start"]
            il = {"il_type": il_type, "il_start": start, "note": _injury_note(desc)}
            continue
        if "injured list" in low and ("activated" in low or "reinstated" in low):
            il = None
            continue

        # --- external (non-injury) roster moves ---
        if "optioned" in low or "outrighted" in low or ("sent" in low and " to " in low and "rehab" not in low):
            ext = {"move": "optioned/sent to minors", "date": tdate}
        elif "designated" in low and "assignment" in low:
            ext = {"move": "designated for assignment", "date": tdate}
        elif "released" in low:
            ext = {"move": "released", "date": tdate}
        elif "suspended" in low:
            ext = {"move": "suspended", "date": tdate}
        elif "recalled" in low or "selected the contract" in low or "reinstated" in low:
            ext = None

    if il is not None:
        return StatusVerdict(
            cause="injury-verified",
            il_type=il["il_type"],
            il_start=il["il_start"],
            injury_note=il["note"],
        )
    if ext is not None:
        return StatusVerdict(
            cause="external-verified",
            external_move=ext["move"],
            move_date=ext["date"],
        )
    return StatusVerdict(cause="unverified")
