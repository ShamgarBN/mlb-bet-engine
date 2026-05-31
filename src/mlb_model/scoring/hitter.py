"""Hitter prop scoring: hit / HR / total bases / strikeout.

Each score is a 0-10 number where **5.0 = league-average matchup**. Higher
scores mean a more favorable matchup for the *batter side* of that prop:

  * ``hit``       — probability of 1+ hit in the game
  * ``hr``        — probability of a home run
  * ``total_bases`` — composite of hit + power
  * ``strikeout`` — probability of 1+ strikeout (so HIGH = good K prop, LOW = batter avoids K)

Design choices, lifted from the public ``mlb-scout`` work and refined:

* **Delta-from-5.0 model.** Each input contributes a signed delta that's added
  to 5.0 and clamped to [0, 10]. Avoids the "everything multiplies to zero"
  problem of weighted-average designs.
* **Composite power index for HR.** Barrel%, xSLG, HardHit%, EV are heavily
  correlated; we collapse them to a single 0-1 power index before scoring so
  any solid power hitter doesn't auto-stack to 10.
* **BIP-based shrinkage.** Per Carleton (BP 2016), Statcast metrics stabilize
  around 50 BIP. We regress small-sample stats toward league mean via
  ``val * conf + lg * (1 - conf)`` where ``conf = bip / (bip + 50)``.
* **Pitcher OPS allowed vs handedness is the dominant signal** (weight 3.0).
  Matches sharp-money intuition: matchup quality dwarfs batter form for
  single-game props.
* **Graceful degradation.** Every input is optional; the function uses what
  it has and returns ``None`` only when nothing usable is available.

Inputs are plain dataclasses, not warehouse objects -- this module has zero
DuckDB dependency and is trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# League-average constants (calibrated from 2018-2025 warehouse means)
# ---------------------------------------------------------------------------

LG_AVG = 0.248
LG_OBP = 0.318
LG_SLG = 0.413
LG_OPS = 0.731
LG_ISO = 0.155
LG_K_PCT = 0.225   # batter K rate (PA-based)
LG_BB_PCT = 0.085
LG_XBA = 0.248
LG_XSLG = 0.415
LG_BARREL = 0.080
LG_HARDHIT = 0.355
LG_EV = 88.5

# Pitcher-side league averages (per-game / per-9 / per-PA)
LG_FIP = 4.20
LG_OPS_ALLOWED = 0.731
LG_K9 = 8.8
LG_BB9 = 3.2
LG_HR9 = 1.20
LG_K_PCT_PIT = 0.225  # pitcher K rate vs batters faced


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

Hand = Literal["L", "R"]


@dataclass
class BatterInputs:
    """Everything we know about a batter heading into today's game.

    All fields optional. Most fields are season-to-date; ``split_*`` fields
    are filtered to PA against the opposing starter's handedness.
    """
    bats: Hand | None = None

    # Season triple-slash + plate-discipline rates
    season_avg: float | None = None
    season_obp: float | None = None
    season_slg: float | None = None
    season_ops: float | None = None
    season_iso: float | None = None
    season_k_pct: float | None = None
    season_bb_pct: float | None = None
    season_pa: int = 0

    # Same metrics split against today's opposing starter's hand
    split_avg: float | None = None
    split_obp: float | None = None
    split_slg: float | None = None
    split_ops: float | None = None
    split_iso: float | None = None
    split_k_pct: float | None = None
    split_pa: int = 0

    # Statcast quality (season-to-date; optional)
    xba: float | None = None
    xslg: float | None = None
    xwoba: float | None = None
    barrel_pct: float | None = None
    hardhit_pct: float | None = None
    avg_ev: float | None = None
    bip: int | None = None  # batted balls in play (for shrinkage)


@dataclass
class PitcherInputs:
    """Everything we know about today's opposing starter."""
    throws: Hand | None = None

    # Headline stats
    era: float | None = None
    fip: float | None = None
    k_pct: float | None = None   # K / batters faced
    bb_pct: float | None = None
    hr9: float | None = None
    batters_faced: int = 0

    # Stats allowed split by *batter's* hand (vs_L = LHB facing him)
    vs_lhb_ops: float | None = None
    vs_rhb_ops: float | None = None
    vs_lhb_avg: float | None = None
    vs_rhb_avg: float | None = None
    vs_lhb_hr9: float | None = None
    vs_rhb_hr9: float | None = None
    vs_lhb_k_pct: float | None = None
    vs_rhb_k_pct: float | None = None


@dataclass
class Scores:
    """The four scores plus inline reasons for the UI."""
    hit: float | None
    hr: float | None
    total_bases: float | None
    strikeout: float | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _shrink(val: float | None, lg: float, n: int | None, n_stab: int) -> float | None:
    """Beta-binomial-style shrinkage toward league mean.

    ``n`` is the relevant sample size (PA for rates, BIP for Statcast).
    At ``n = n_stab`` we trust the observed value 50%; rises smoothly to 100%.
    """
    if val is None:
        return None
    if n is None or n <= 0:
        return lg
    conf = n / (n + n_stab)
    return val * conf + lg * (1.0 - conf)


def _norm01(val: float, lo: float, hi: float) -> float:
    """Linear normalize to [0, 1] with clipping."""
    if hi <= lo:
        return 0.5
    return _clip((val - lo) / (hi - lo), 0.0, 1.0)


def _pitcher_vs_hand(pitcher: PitcherInputs, bat_hand: Hand | None, attr: str) -> float | None:
    """Return ``vs_lhb_<attr>`` or ``vs_rhb_<attr>`` based on batter hand."""
    if bat_hand is None:
        return None
    field = f"vs_lhb_{attr}" if bat_hand == "L" else f"vs_rhb_{attr}"
    return getattr(pitcher, field, None)


# ---------------------------------------------------------------------------
# Hit score (probability of 1+ H)
# ---------------------------------------------------------------------------

def score_hit(
    batter: BatterInputs,
    pitcher: PitcherInputs,
    park_factor: float | None = None,
) -> float | None:
    """0-10, anchored at 5.0 = league-average hit-probability matchup.

    Weights (delta caps):
      * Split AVG vs hand          ±2.0
      * xBA quality + luck         ±1.5 / ±1.0
      * HardHit%                   ±0.5
      * Pitcher OPS allowed vs hand ±3.0   (primary signal)
      * FIP fallback (no split)    ±1.5
      * Park factor                ±0.5
    """
    delta = 0.0
    has_batter = False
    has_pitcher = False
    bat_hand = batter.bats

    # ── Batter quality ──────────────────────────────────────────────────────
    # Thresholds are intentionally low: shrinkage will pull tiny samples
    # toward league mean automatically, so we'd rather feed a noisy 15-PA
    # signal through the regressor than emit None and let the score
    # collapse entirely to the pitcher's contribution.
    split_pa_safe = batter.split_pa or 0
    if batter.split_avg is not None and split_pa_safe >= 10:
        avg = _shrink(batter.split_avg, LG_AVG, split_pa_safe, 100)
        if avg is not None:
            has_batter = True
            delta += _clip((avg - LG_AVG) / 0.072 * 2.0, -2.0, 2.0)
    elif batter.season_avg is not None and batter.season_pa >= 10:
        avg = _shrink(batter.season_avg, LG_AVG, batter.season_pa, 120)
        if avg is not None:
            has_batter = True
            delta += _clip((avg - LG_AVG) / 0.072 * 1.5, -1.5, 1.5)

    if batter.xba is not None:
        # xBA stabilizes at ~50 BIP
        xba = _shrink(batter.xba, LG_XBA, batter.bip, 50)
        if xba is not None:
            has_batter = True
            delta += _clip((xba - LG_XBA) / 0.060 * 1.5, -1.5, 1.5)
            # Luck correction (xBA - actual AVG): persistent gap = regression candidate
            if batter.season_avg is not None and batter.season_pa >= 40:
                luck = xba - batter.season_avg
                delta += _clip(luck / 0.040 * 1.0, -1.0, 1.0)

    if batter.hardhit_pct is not None:
        hh = _shrink(batter.hardhit_pct, LG_HARDHIT, batter.bip, 50)
        if hh is not None:
            has_batter = True
            delta += _clip((hh - LG_HARDHIT) / 0.15 * 0.5, -0.5, 0.5)

    # ── Pitcher quality (dominant signal) ────────────────────────────────────
    # Convention: HIGHER OPS-allowed / HIGHER FIP = worse pitcher = easier
    # for batter = POSITIVE delta to the batter's hit score. (The original
    # mlb-scout FIP fallback had this sign inverted — corrected here.)
    p_ops = _pitcher_vs_hand(pitcher, bat_hand, "ops")
    if p_ops is not None:
        has_pitcher = True
        delta += _clip((p_ops - LG_OPS_ALLOWED) / 0.120 * 3.0, -3.0, 3.0)
    elif pitcher.fip is not None:
        has_pitcher = True
        delta += _clip((pitcher.fip - LG_FIP) / 1.0 * 1.5, -1.5, 1.5)

    # ── Park ─────────────────────────────────────────────────────────────────
    if park_factor is not None:
        delta += _clip((park_factor - 1.0) / 0.10 * 0.5, -0.5, 0.5)

    if not has_batter and not has_pitcher:
        return None
    return round(_clip(5.0 + delta, 0.0, 10.0), 1)


# ---------------------------------------------------------------------------
# HR score
# ---------------------------------------------------------------------------

def score_hr(
    batter: BatterInputs,
    pitcher: PitcherInputs,
    park_factor: float | None = None,
) -> float | None:
    """0-10. Composite-power-index design from mlb-scout v4.

    Returns ``None`` if Statcast and season ISO are both missing.
    """
    has_signal = (
        batter.barrel_pct is not None or batter.xslg is not None
        or batter.hardhit_pct is not None or batter.avg_ev is not None
        or (batter.season_iso is not None and batter.season_pa >= 80)
    )
    if not has_signal:
        return None

    # ── Step 1: BIP shrinkage on Statcast quality metrics ────────────────────
    bip = batter.bip
    weights = {"barrel": 0.56, "xslg": 0.16, "hh": 0.13, "ev": 0.15}
    bounds = {
        "barrel": (0.02, 0.18, LG_BARREL),
        "xslg":   (0.310, 0.580, LG_XSLG),
        "hh":     (0.20, 0.52, LG_HARDHIT),
        "ev":     (83.0, 93.0, LG_EV),
    }
    raw = {
        "barrel": _shrink(batter.barrel_pct, LG_BARREL, bip, 50),
        "xslg":   _shrink(batter.xslg,       LG_XSLG,   bip, 50),
        "hh":     _shrink(batter.hardhit_pct, LG_HARDHIT, bip, 50),
        "ev":     _shrink(batter.avg_ev,     LG_EV,     bip, 50),
    }
    norm = {}
    for k, v in raw.items():
        if v is None:
            continue
        lo, hi, _ = bounds[k]
        norm[k] = _norm01(v, lo, hi)

    if norm:
        total_w = sum(weights[k] for k in norm)
        power_idx = sum(norm[k] * weights[k] for k in norm) / total_w
    else:
        # Fall back to regressed season ISO as a single-input power index
        iso = _shrink(batter.season_iso, LG_ISO, batter.season_pa, 200)
        if iso is None:
            return None
        # Map ISO range .080 (weak) → .250 (elite) into 0..1
        power_idx = _norm01(iso, 0.080, 0.250)

    # ── Step 4: Base from power_idx — weak=2, avg=4.5, elite=7 ──────────────
    base = 1.0 + power_idx * 7.0

    # ── Step 5: Pitcher HR/9 adjustment ──────────────────────────────────────
    p_hr9 = _pitcher_vs_hand(pitcher, batter.bats, "hr9")
    if p_hr9 is None:
        p_hr9 = pitcher.hr9
    pitch_adj = _clip((p_hr9 - LG_HR9) / 0.90 * 1.5, -1.5, 1.5) if p_hr9 is not None else 0.0

    # ── Step 6: Park ─────────────────────────────────────────────────────────
    park_adj = _clip((park_factor - 1.0) / 0.18 * 1.0, -0.5, 1.0) if park_factor is not None else 0.0

    # ── Step 7: Season-split ISO (small supplementary input) ─────────────────
    iso_adj = 0.0
    if batter.split_iso is not None and (batter.split_pa or 0) >= 30:
        iso_reg = _shrink(batter.split_iso, LG_ISO, batter.split_pa, 200)
        if iso_reg is not None:
            iso_adj = _clip((iso_reg - LG_ISO) / 0.060 * 0.2, -0.2, 0.2)

    return round(_clip(base + pitch_adj + park_adj + iso_adj, 0.0, 10.0), 1)


# ---------------------------------------------------------------------------
# Total bases (composite of hit + power)
# ---------------------------------------------------------------------------

def score_tb(
    batter: BatterInputs,
    pitcher: PitcherInputs,
    park_factor: float | None = None,
) -> float | None:
    """0-10. Blended hit + HR signal.

    Total bases (over 1.5) hits when the batter gets a double, two singles,
    a HR, etc. So it rewards both contact (hit-score) and power (HR-score).
    Weighted 60% hit, 40% HR -- pure singles count for hit but not TB, and
    a HR counts heavily for both.
    """
    h = score_hit(batter, pitcher, park_factor)
    hr = score_hr(batter, pitcher, park_factor)
    if h is None and hr is None:
        return None
    if h is None:
        return hr
    if hr is None:
        return h
    return round(0.60 * h + 0.40 * hr, 1)


# ---------------------------------------------------------------------------
# Strikeout score (high = K likely; LOW = batter unlikely to K)
# ---------------------------------------------------------------------------

def score_k(batter: BatterInputs, pitcher: PitcherInputs) -> float | None:
    """0-10, anchored at 5.0 = league-average K-probability matchup.

    HIGH score = K is more likely (good for K-over bet on the batter).
    LOW score  = batter is contact-oriented vs a contact pitcher (K-under).

    Weights:
      * Pitcher K% vs hand   ±3.0  (dominant — strikeout is a pitcher skill)
      * Pitcher overall K%   ±2.0  (fallback if no split)
      * Batter K% (split)    ±2.5
      * Batter K% (season)   ±1.5  (fallback)
    """
    delta = 0.0
    has_pitch = False
    has_bat = False

    p_k = _pitcher_vs_hand(pitcher, batter.bats, "k_pct")
    if p_k is not None:
        has_pitch = True
        delta += _clip((p_k - LG_K_PCT_PIT) / 0.08 * 3.0, -3.0, 3.0)
    elif pitcher.k_pct is not None:
        has_pitch = True
        delta += _clip((pitcher.k_pct - LG_K_PCT_PIT) / 0.08 * 2.0, -2.0, 2.0)

    if batter.split_k_pct is not None and (batter.split_pa or 0) >= 10:
        bk = _shrink(batter.split_k_pct, LG_K_PCT, batter.split_pa, 100)
        if bk is not None:
            has_bat = True
            delta += _clip((bk - LG_K_PCT) / 0.08 * 2.5, -2.5, 2.5)
    elif batter.season_k_pct is not None and batter.season_pa >= 10:
        bk = _shrink(batter.season_k_pct, LG_K_PCT, batter.season_pa, 120)
        if bk is not None:
            has_bat = True
            delta += _clip((bk - LG_K_PCT) / 0.08 * 1.5, -1.5, 1.5)

    if not has_pitch and not has_bat:
        return None
    return round(_clip(5.0 + delta, 0.0, 10.0), 1)


# ---------------------------------------------------------------------------
# All four scores in one call
# ---------------------------------------------------------------------------

def score_matchup(
    batter: BatterInputs,
    pitcher: PitcherInputs,
    park_factor: float | None = None,
) -> Scores:
    """Compute all four prop scores for one (batter, pitcher) matchup."""
    return Scores(
        hit=score_hit(batter, pitcher, park_factor),
        hr=score_hr(batter, pitcher, park_factor),
        total_bases=score_tb(batter, pitcher, park_factor),
        strikeout=score_k(batter, pitcher),
    )
