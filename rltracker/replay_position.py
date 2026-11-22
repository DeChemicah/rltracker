"""Phase 3: positioning, rotation and heatmaps from a track file.

Everything is "team-relative": y is flipped for orange so that, for every
player, their own goal is at negative y and the opponent's goal at positive y.
x is flipped too (a 180° turn), so left/right stay the same from the player's
point of view and heatmaps from both colours can be added together.
"""
from __future__ import annotations

import base64
import warnings

import numpy as np

from . import replay_extract as X

FIELD_X = 4096          # side wall
FIELD_Y = 5120          # back wall (goal line)
THIRD = FIELD_Y / 3     # defensive / neutral / offensive split
HEAT_W, HEAT_H = 32, 40  # heatmap cells: across (x) × along the field (y)
HEAT_SCALE = 100000      # stored as share of time × this, uint16

GROUP = [("pct_def_third", "% defensive third"), ("pct_neutral_third", "% neutral third"), ("pct_off_third", "% offensive third"),
         ("pct_behind_ball", "% behind ball"), ("pct_last_back", "% last back"), ("pct_first_forward", "% most forward"),
         ("avg_dist_ball", "Avg distance to ball"), ("avg_dist_mate", "Avg distance to teammates")]
ROTATION = [("pct_double_up", "% two+ in offensive third"), ("pct_double_back", "% two+ in defensive third"),
            ("pct_no_last_back", "% nobody behind ball"), ("pct_all_back", "% everyone behind ball")]


def encode_heat(h: np.ndarray) -> str:
    """(HEAT_H, HEAT_W) share-of-time grid → compact base64 string."""
    v = np.clip(np.round(h * HEAT_SCALE), 0, 65535).astype("<u2")
    return base64.b64encode(v.tobytes()).decode()


def decode_heat(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<u2").astype(np.float64).reshape(HEAT_H, HEAT_W) / HEAT_SCALE


def compute(meta: dict, A: dict, dt: np.ndarray) -> tuple[list[dict], dict, list[str | None]]:
    """→ (per-player stats, per-team rotation {team: {...}}, per-player heat strings)."""
    state = A["car_state"]
    if state.ndim != 2 or not len(dt):
        return [], {}, []
    P = state.shape[1]
    present = (state & X.PRESENT) > 0
    pos = A["car_pos"].astype(np.float64)
    ball = A["ball_pos"].astype(np.float64)
    teams = [meta["players"][i].get("team") for i in range(P)]
    flip = np.array([-1.0 if t == 1 else 1.0 for t in teams])
    yr = pos[:, :, 1] * flip            # team-relative y (own goal negative)
    xr = pos[:, :, 0] * flip
    ball_yr = ball[:, 1][:, None] * flip  # ball y in each player's frame

    players, heats = [], []
    for i in range(P):
        w = dt * present[:, i]
        T = float(w.sum())
        if T < 5:
            players.append({})
            heats.append(None)
            continue

        def frac(mask, base=w, total=T):
            return round(100 * float((base * mask).sum()) / total, 1) if total > 0 else None

        y = yr[:, i]
        mates = [j for j in range(P) if j != i and teams[j] == teams[i]]
        d_ball = np.linalg.norm(pos[:, i] - ball, axis=1)
        st = {
            "pct_def_third": frac(y < -THIRD), "pct_neutral_third": frac((y >= -THIRD) & (y <= THIRD)), "pct_off_third": frac(y > THIRD),
            "pct_behind_ball": frac(y < ball_yr[:, i]),
            "avg_dist_ball": round(float((w * d_ball).sum()) / T),
            "pct_last_back": None, "pct_first_forward": None, "avg_dist_mate": None,
        }
        if mates:
            # only frames where this player and at least one teammate are on the field
            mp = present[:, mates]
            both = present[:, i] & mp.any(axis=1)
            wb = dt * both
            Tb = float(wb.sum())
            ym = np.where(mp, yr[:, mates], np.nan)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo = np.nanmin(ym, axis=1)
                hi = np.nanmax(ym, axis=1)
                dm = np.nanmean(np.where(mp, np.linalg.norm(pos[:, mates] - pos[:, i][:, None], axis=2), np.nan), axis=1)
            if Tb > 5:
                st["pct_last_back"] = frac(y <= np.nan_to_num(lo, nan=np.inf), wb, Tb)
                st["pct_first_forward"] = frac(y >= np.nan_to_num(hi, nan=-np.inf), wb, Tb)
                st["avg_dist_mate"] = round(float((wb * np.nan_to_num(dm)).sum()) / Tb)
        players.append(st)

        on = present[:, i]
        h, _, _ = np.histogram2d(np.clip(yr[on, i], -FIELD_Y, FIELD_Y - 1e-6), np.clip(xr[on, i], -FIELD_X, FIELD_X - 1e-6),
                                 bins=[HEAT_H, HEAT_W], range=[[-FIELD_Y, FIELD_Y], [-FIELD_X, FIELD_X]], weights=dt[on])
        heats.append(encode_heat(h / T))

    rotation = {}
    for team in (0, 1):
        idx = [i for i in range(P) if teams[i] == team]
        if len(idx) < 2:
            continue
        tp = present[:, idx]
        full = tp.sum(axis=1) >= 2          # at least two of the team on the field
        w = dt * full
        T = float(w.sum())
        if T < 5:
            continue
        yt = np.where(tp, yr[:, idx], np.nan)
        by = ball_yr[:, idx[0]]
        n_off = np.sum(yt > THIRD, axis=1)
        n_def = np.sum(yt < -THIRD, axis=1)
        n_behind = np.sum(yt < by[:, None], axis=1)
        n_on = tp.sum(axis=1)

        def tfrac(mask):
            return round(100 * float((w * mask).sum()) / T, 1)

        rotation[team] = {"pct_double_up": tfrac(n_off >= 2), "pct_double_back": tfrac(n_def >= 2),
                          "pct_no_last_back": tfrac(n_behind == 0), "pct_all_back": tfrac(n_behind == n_on), "seconds": round(T, 1)}
    return players, rotation, heats


def avg_heat(strings: list[str]) -> str | None:
    """Average several games' heatmaps (each is a share-of-time grid)."""
    hs = [decode_heat(x) for x in strings if x]
    return encode_heat(sum(hs) / len(hs)) if hs else None


def avg_rotation(rots: list[dict]) -> dict:
    rots = [r for r in rots if r]
    if not rots:
        return {}
    from .replay_stats import ROTATION as ALL  # includes possession and kickoffs
    out = {}
    for k, _l in ALL:
        vals = [r[k] for r in rots if r.get(k) is not None]
        if vals:
            out[k] = round(sum(vals) / len(vals), 1)
    return out
