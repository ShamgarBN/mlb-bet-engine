"""Lightweight, key-less news lookup via Google News RSS.

Used by :mod:`slugger_slump` to explain *why* a big home-run hitter has gone
cold: is it an injury (IL stint, strain, fracture...) or some other external
factor (benched, optioned, personal leave, suspension)?

No API key, no third-party deps — just the stdlib RSS feed at
``news.google.com/rss/search``. Best-effort: any network/parse failure returns
an empty result so the caller degrades gracefully to gamelog-only signals.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("analysis.news")

# Terms that, if present *near the player's name* in a recent headline, point
# at a physical injury. Curated for precision — bare common words ("back",
# "hand") are avoided in favour of injury-specific phrasings.
INJURY_TERMS = (
    "injur", "injured list", "il stint", "10-day il", "15-day il", "60-day il",
    "to the il", "on the il", "disabled list", "strain", "sprain", "surgery",
    "soreness", "fractur", "hamstring", "oblique", "rib", "knee", "shoulder",
    "wrist", "elbow", "forearm", "lower back", "back injury", "back spasms",
    "calf", "quad", "groin", "ankle", "hip injury", "thumb", "concussion",
    "day-to-day", "placed on", "mri", "out for the", "sidelined",
    "left the game", "exited the game", "x-ray", "inflammation", "setback",
)

# Terms that point at a non-injury external factor.
EXTERNAL_TERMS = (
    "benched", "demoted", "optioned", "sent down", "paternity", "bereavement",
    "suspended", "suspension", "personal leave", "family matter", "released",
    "designated for assignment", "dfa", "platoon", "out of the lineup",
    "off day", "day off", "rest day", "getting a breather",
)

# If a negation sits near the injury term, the player likely AVOIDED the
# injury or is returning from it — don't blame it for the slump.
NEGATION_TERMS = (
    "dodge", "avoid", "no injury", "not injured", "good news", "cleared",
    "returns", "return from", "activated", "reinstated", "back in the lineup",
    "out of the woods", "escapes", "won't need", "no structural",
    "shuts down", "shut down injury", "injury concern", "stays hot",
    "optimistic", "breaking out", "come back clean", "comes back clean",
)


@dataclass(slots=True)
class Headline:
    title: str
    url: str
    source: str
    published: datetime | None

    def age_days(self, *, now: datetime | None = None) -> float | None:
        if self.published is None:
            return None
        ref = now or datetime.now(timezone.utc)
        return (ref - self.published).total_seconds() / 86400.0


@dataclass(slots=True)
class NewsVerdict:
    """What the recent news says about a player's cold streak."""

    query: str
    headlines: list[Headline] = field(default_factory=list)
    injury_related: bool = False
    external_related: bool = False

    @property
    def cause(self) -> str:
        if self.injury_related:
            return "injury"
        if self.external_related:
            return "external"
        if self.headlines:
            return "unclear"
        return "no-news"


def _parse_pubdate(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def fetch_headlines(query: str, *, limit: int = 8) -> list[Headline]:
    """Fetch up to ``limit`` recent headlines for a free-text query."""
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(url, headers={"User-Agent": settings.http_user_agent})
    try:
        with urllib.request.urlopen(req, timeout=settings.http_timeout_seconds) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
    except Exception:  # noqa: BLE001 -- news is best-effort; never fatal
        log.warning("news.fetch.failed", query=query)
        return []

    out: list[Headline] = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        # Google News encodes the publisher in a <source> child element.
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        published = _parse_pubdate(item.findtext("pubDate"))
        out.append(Headline(title=title, url=link, source=source, published=published))
    return out


_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def _surname(player_name: str) -> str:
    """Best-effort surname for proximity matching (drops Jr./III suffixes)."""
    tokens = [t for t in player_name.split() if t.strip(".").lower() not in _SUFFIXES]
    return (tokens[-1] if tokens else player_name).lower()


def _term_near_name(text: str, surname: str, terms: tuple[str, ...], window: int = 45) -> bool:
    """True if any ``term`` occurs within ``window`` chars of the surname, and
    is not negated by a nearby negation word.

    Requiring the term to sit beside the player's own name is what stops a
    headline about a *teammate's* injury (that merely mentions our player)
    from being misread as our player's injury.
    """
    name_positions = [i for i in range(len(text)) if text.startswith(surname, i)]
    if not name_positions:
        return False
    for term in terms:
        start = 0
        while (idx := text.find(term, start)) != -1:
            start = idx + 1
            if not any(abs(idx - npos) <= window for npos in name_positions):
                continue
            # Reject if a negation word sits in the same neighbourhood.
            lo, hi = max(0, idx - window), idx + len(term) + window
            window_text = text[lo:hi]
            if any(neg in window_text for neg in NEGATION_TERMS):
                continue
            return True
    return False


def assess(
    player_name: str,
    *,
    team: str | None = None,
    max_age_days: float = 30.0,
    limit: int = 10,
) -> NewsVerdict:
    """Search recent news for a player and classify the cause of their cold streak.

    Only headlines newer than ``max_age_days`` are considered, and an injury /
    external term only counts when it sits *near the player's surname* and
    isn't negated — so a stale April story or a teammate's injury doesn't get
    blamed for our player's June slump.
    """
    query = f"{player_name} injury OR IL OR lineup OR slump"
    headlines = fetch_headlines(query, limit=limit)
    surname = _surname(player_name)

    verdict = NewsVerdict(query=query, headlines=headlines)
    for h in headlines:
        age = h.age_days()
        if age is not None and age > max_age_days:
            continue
        text = h.title.lower()
        if _term_near_name(text, surname, INJURY_TERMS):
            verdict.injury_related = True
        if _term_near_name(text, surname, EXTERNAL_TERMS):
            verdict.external_related = True
    return verdict
