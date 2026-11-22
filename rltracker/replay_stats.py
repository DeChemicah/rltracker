"""Phase 2: boost, movement and demo stats from a track file.

Everything is computed from data/tracks/<match_id>.npz, so it can be recomputed
for every stored game whenever these definitions change (bump STATS_VERSION).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from . import replay_extract as X
from . import replay_position as RP
from . import replay_ball as RB

STATS_VERSION = 11  # 11: powerslide count, crossbar hits · 10: fastest ball hit per player · 9: every shot the game counted (shot moments from the replay, or ranked up to the scoreboard count) · 8: goal timeline, overtime, momentum · 7: each shot's ball path + outcome · 6: live time only (no kickoff countdowns / goal celebrations), boost pads by location · 5: demos as a main stat · 2: ball hits from touches · 3: positioning, rotation, heatmaps · 4: ball play, shots, kickoffs
SUPERSONIC = 2190      # uu/s (game threshold is 2200; small margin for rounding)
BOOST_SPEED = 1410     # uu/s: top speed without boosting
GROUND_Z = 25          # car resting height is ~17
HIGH_AIR_Z = 300       # above this counts as high in the air

# Field: blue (team 0) defends negative y, orange (team 1) positive y.

# The 34 boost pads of a standard soccar field: (x, y, big).
PADS = [(0, -4240, 0), (-1792, -4184, 0), (1792, -4184, 0), (-3072, -4096, 1), (3072, -4096, 1), (-940, -3308, 0), (940, -3308, 0),
        (0, -2816, 0), (-3584, -2484, 0), (3584, -2484, 0), (-1788, -2300, 0), (1788, -2300, 0), (-2048, -1036, 0), (0, -1024, 0),
        (2048, -1036, 0), (-3584, 0, 1), (-1024, 0, 0), (1024, 0, 0), (3584, 0, 1), (-2048, 1036, 0), (0, 1024, 0), (2048, 1036, 0),
        (-1788, 2300, 0), (1788, 2300, 0), (-3584, 2484, 0), (3584, 2484, 0), (0, 2816, 0), (-940, 3310, 0), (940, 3308, 0),
        (-3072, 4096, 1), (3072, 4096, 1), (-1792, 4184, 0), (1792, 4184, 0), (0, 4240, 0)]
_PAD_XY = np.array([(x, y) for x, y, _b in PADS], dtype=np.float64)


def pad_map(A: dict, pickups: list[dict]) -> dict[str, int] | None:
    """Which pad each pickup actor is. A single pickup's car position is noisy (the
    event arrives a frame or two late), so every actor gets a majority vote over all
    its pickups, then actors and pads are matched one-to-one. None = not a standard
    pad layout (the votes don't agree), so callers fall back to boost-gain guesses."""
    pos = A["car_pos"]
    votes: dict[str, Counter] = defaultdict(Counter)
    for e in pickups:
        f, i = e["frame"], e["player"]
        if f >= len(pos) or i >= pos.shape[1]:
            continue
        d = np.linalg.norm(_PAD_XY - pos[f, i, :2].astype(np.float64), axis=1)
        j = int(d.argmin())
        if d[j] < 900:
            votes[e["pad"]][j] += 1
    total = sum(sum(c.values()) for c in votes.values())
    if not total:
        return None
    cand = sorted(((n, a, j) for a, c in votes.items() for j, n in c.items()), reverse=True)
    out: dict[str, int] = {}
    used = set()
    for n, a, j in cand:
        if a in out or j in used:
            continue
        out[a] = j
        used.add(j)
    agree = sum(votes[a][j] for a, j in out.items()) / total
    return out if agree >= 0.6 else None


def dedupe_pickups(pickups: list[dict], pads: dict[str, int] | None, t: np.ndarray) -> list[dict]:
    """Older track files stored every re-send of a pad's state as a pickup. A pad can't
    be taken again before it respawns (small 4 s, big 10 s), so a second pickup of the
    same pad by the same player inside that time is the same pickup. Track files that
    carry the game's pickup counter ("n") are already exact."""
    if all("n" in e for e in pickups):
        return pickups
    last: dict[str, tuple[int, float]] = {}
    out = []
    for e in sorted(pickups, key=lambda e: e["frame"]):
        tt = float(t[min(e["frame"], len(t) - 1)])
        j = pads.get(e["pad"]) if pads else None
        respawn = 10.0 if (j is not None and PADS[j][2]) else 4.0
        prev = last.get(e["pad"])
        if prev and prev[0] == e["player"] and tt - prev[1] < respawn:
            continue
        last[e["pad"]] = (e["player"], tt)
        out.append(e)
    return out


MOMENTUM_STEP = 30  # seconds of game clock per momentum bucket


def timeline(meta: dict, A: dict, events: list[dict], dt: np.ndarray, touch_list: list) -> dict:
    """Goals with game-clock times, whether it went to overtime, and momentum per 30 s of
    game clock: share of live time the ball was in each half, and possession (last touch)."""
    t = A["t"].astype(np.float64)
    clock = A["clock"].astype(np.int64) if "clock" in A else np.zeros(len(t), np.int64)
    F = len(t)
    if not F:
        return {"goals": [], "overtime": False, "momentum": []}
    zero = np.flatnonzero(clock <= 0)
    ot_from = None
    if len(zero):
        z0 = int(zero[0])
        later = np.flatnonzero(clock[z0:] > 0)
        if len(later):  # the clock counts up again after 0:00 → overtime
            ot_from = z0 + int(later[0]) - 1
    ot = np.zeros(F, bool)
    if ot_from is not None:
        ot[ot_from:] = True
    teams = [p.get("team") for p in meta["players"]]
    names = [p.get("name") for p in meta["players"]]
    goals = []
    for e in sorted((e for e in events if e["type"] == "goal"), key=lambda e: e["frame"]):
        f = min(e["frame"], F - 1)
        sc = e.get("player")
        goals.append({"t": round(float(t[f] - t[0]), 1), "clock": int(clock[f]), "ot": bool(ot[f]), "team": e["team"],
                      "scorer": names[sc] if sc is not None and 0 <= sc < len(names) else None})
    # bucket index per frame: regulation counts down from 5:00, overtime counts up
    reg_idx = np.clip((300 - clock) // MOMENTUM_STEP, 0, 300 // MOMENTUM_STEP - 1)
    idx = np.where(ot, 300 // MOMENTUM_STEP + np.clip(clock, 0, None) // MOMENTUM_STEP, reg_idx)
    by = A["ball_pos"][:, 1].astype(np.float64)
    last_team = np.full(F, -1)
    for f, i in sorted(touch_list):
        if 0 <= f < F and i < len(teams):
            last_team[f:] = teams[i]
    momentum = []
    for b in range(int(idx.max()) + 1 if F else 0):
        sel = idx == b
        w = dt * sel
        live = float(w.sum())
        if live < 3:
            continue
        in_orange = float((w * (by > 0)).sum())
        poss = w * (last_team >= 0)
        pt = float(poss.sum())
        is_ot = b >= 300 // MOMENTUM_STEP
        start = (b - 300 // MOMENTUM_STEP) * MOMENTUM_STEP if is_ot else 300 - b * MOMENTUM_STEP
        momentum.append({"b": b, "ot": is_ot, "clock": int(start), "live_s": round(live, 1),
                         "blue_attack": round(100 * in_orange / live, 1),          # ball in orange's half = blue attacking
                         "blue_poss": round(100 * float((w * (last_team == 0)).sum()) / pt, 1) if pt > 1 else None})
    return {"goals": goals, "overtime": ot_from is not None, "momentum": momentum}


def live_mask(A: dict, events: list[dict]) -> np.ndarray:
    """True while the ball is in play: not during kickoff countdowns (ball parked on
    the centre spot) and not from a goal until the next kickoff is released."""
    b = A["ball_pos"].astype(np.float64)
    v = np.linalg.norm(A["ball_vel"].astype(np.float64), axis=1)
    still = (np.abs(b[:, 0]) < 5) & (np.abs(b[:, 1]) < 5) & (v < 5)
    dead = still.copy()
    F = len(b)
    for g in sorted(e["frame"] for e in events if e["type"] == "goal"):
        nxt = np.flatnonzero(still[g:])
        if not len(nxt):
            dead[g:] = True
            continue
        s = g + int(nxt[0])
        rel = np.flatnonzero(~still[s:])
        dead[g:(s + int(rel[0])) if len(rel) else F] = True
    return ~dead

GROUPS = {
    "boost": [("bpm", "Boost / min"), ("avg_boost", "Avg boost"), ("pct_zero", "% at 0 boost"), ("pct_full", "% at 100 boost"),
              ("big_pads", "Big pads"), ("small_pads", "Small pads"), ("stolen", "Pads stolen"), ("overfill", "Boost wasted")],
    "movement": [("avg_speed", "Avg speed"), ("pct_supersonic", "% supersonic"), ("pct_boost_speed", "% boost speed"),
                 ("pct_slow", "% slow"), ("pct_ground", "% on ground"), ("pct_low_air", "% low air"), ("pct_high_air", "% high air"),
                 ("jumps", "Jumps"), ("double_jumps", "Double jumps"), ("dodges", "Dodges"), ("powerslides", "Powerslides"), ("powerslide_s", "Powerslide (s)"),
                 ("distance_km", "Distance (km)")],
    "demos": [("demos", "Demos"), ("demoed", "Demoed")],
    "ball": RB.GROUP,
    "positioning": RP.GROUP,
    "touches": [("ball_hits", "Ball hits (from replay)")],
}

# Which way is "better" when comparing (default: higher is better).
LOWER_BETTER = {"pct_zero", "overfill", "demoed", "pct_double_up", "pct_no_last_back"}
ROTATION = RP.ROTATION + RB.TEAM  # team-level, stored on each match team as "rotation"
NEUTRAL = {"kickoffs", "clears", "tracked_shots", "pct_def_third", "pct_neutral_third", "pct_off_third", "pct_behind_ball", "pct_last_back", "pct_first_forward",
           "avg_dist_ball", "avg_dist_mate", "pct_double_back", "pct_all_back",
           "avg_boost", "pct_full", "small_pads", "pct_boost_speed", "pct_slow", "pct_ground", "pct_low_air",
           "pct_high_air", "jumps", "double_jumps", "powerslide_s", "powerslides", "crossbars"}
# Single-game records worth showing (highest value wins).
RECORD_KEYS = ["max_hit_speed", "crossbars", "powerslides", "centres", "max_shot_speed", "max_goal_speed", "aerial_touches", "pct_possession", "clears", "bpm", "big_pads", "stolen",
               "avg_speed", "pct_supersonic", "distance_km", "pct_high_air", "dodges"]  # demos: a main stat now (stat_records)
KEYS = [k for grp in GROUPS.values() for k, _l in grp if k != "ball_hits"]  # ball hits live in the normal stats
LABELS = {k: l for grp in GROUPS.values() for k, l in grp}


def has(p: dict) -> bool:
    """Player has full replay stats (not just a 'barely played' stub)."""
    return bool(p.get("rstats")) and "bpm" in p["rstats"]


def meta() -> dict:
    """Labels and comparison direction for the browser."""
    return {"groups": [[g.capitalize(), [[k, l] for k, l in items]] for g, items in GROUPS.items() if g != "touches"],  # list keeps the order
            "rotation": [[k, l] for k, l in ROTATION],
            "lower_better": sorted(LOWER_BETTER), "neutral": sorted(NEUTRAL),
            "team_sum": sorted(TEAM_SUM), "team_max": sorted(TEAM_MAX), "pads": PADS}


# How a team's value is built from its players' values in one game (anything else: the average).
TEAM_SUM = {"big_pads", "small_pads", "stolen", "overfill", "jumps", "double_jumps", "dodges", "powerslide_s", "powerslides", "crossbars", "distance_km",
            "demos", "demoed", "aerial_touches", "pct_possession", "clears", "centres", "kickoffs", "kickoff_wins"}
TEAM_MAX = {"max_hit_speed", "max_shot_speed", "max_goal_speed"}


def avg_of(players: list[dict]) -> dict:
    """Per-player average of every replay stat over the given player entries."""
    rows = [p["rstats"] for p in players if has(p)]
    if not rows:
        return {}
    out = {}
    for k in KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]  # e.g. "% last back" doesn't exist in 1v1
        out[k] = round(sum(vals) / len(vals), 2) if vals else None
    return out


def _rising(bits: np.ndarray, flag: int) -> int:
    on = (bits & flag) > 0
    return int(np.count_nonzero(on[1:] & ~on[:-1]))


def _count_starts(on: np.ndarray, t: np.ndarray, merge_s: float = 0.2) -> int:
    """How many separate stretches a condition is true for. Stretches less than merge_s apart
    count as one (feathering powerslide is one slide, not several)."""
    if not len(on):
        return 0
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    ends = np.flatnonzero(on & ~np.r_[on[1:], False])  # last frame of each stretch
    if not len(starts):
        return 0
    gaps = t[starts[1:]] - t[ends[:-1]]
    return 1 + int(np.count_nonzero(gaps > merge_s))


def touches(A: dict) -> list[tuple[int, int]]:
    """(frame, player) for every ball touch, worked out from the recording:
    the ball's velocity changes sharply while a car is touching it.
    (Replays only record which *team* touched last, not who.)"""
    bpos = A["ball_pos"].astype(np.float64)
    bvel = A["ball_vel"].astype(np.float64)
    cpos = A["car_pos"].astype(np.float64)
    present = (A["car_state"] & X.PRESENT) > 0
    t = A["t"]
    out, last = [], {}
    if len(bpos) < 2 or cpos.ndim != 3:
        return out
    dv = np.linalg.norm(np.diff(bvel, axis=0), axis=1)
    jump = np.linalg.norm(np.diff(bpos, axis=0), axis=1)
    for f in np.where((dv > 250) & (jump < 400))[0] + 1:  # skip kickoff resets (ball teleports)
        d = np.linalg.norm(cpos[f] - bpos[f], axis=1)
        d[~present[f]] = 1e9
        i = int(np.argmin(d))
        if d[i] > TOUCH_DIST:
            continue
        if i in last and t[f] - t[last[i]] < 0.15:  # same contact
            continue
        last[i] = f
        out.append((int(f), i))
    return out


TOUCH_DIST = 230  # car centre to ball centre (ball radius ~93 + car half-length ~60-80, plus a frame of motion)


def compute(track_path: str) -> dict:
    """{"version", "seconds", "players": [{name, id, team, <stat>: value, ...}]}"""
    meta, A, events = X.load_track(track_path)
    t = A["t"].astype(np.float64)
    dt = np.diff(t, append=t[-1] if len(t) else 0)
    dt = np.clip(dt, 0, 0.5)  # ignore long gaps (pauses)
    dt = dt * live_mask(A, events)  # only time with the ball in play counts
    state = A["car_state"]
    pos = A["car_pos"].astype(np.float64)
    vel = A["car_vel"].astype(np.float64)
    boost = A["car_boost"].astype(np.float64)
    P = state.shape[1] if state.ndim == 2 else 0
    out = []
    pickups = [e for e in events if e["type"] == "pickup"]
    pads = pad_map(A, pickups)
    pickups = dedupe_pickups(pickups, pads, t)
    touch_list = touches(A)
    demos = [e for e in events if e["type"] == "demo"]
    for i in range(P):
        pm = meta["players"][i]
        present = (state[:, i] & X.PRESENT) > 0
        w = dt * present
        T = float(w.sum())
        if T < 5:  # barely played (joined at the very end)
            out.append({"name": pm["name"], "id": pm.get("id"), "team": pm.get("team"), "seconds": round(T, 1)})
            continue
        speed = np.linalg.norm(vel[:, i], axis=1)
        z = pos[:, i, 2]

        def frac(mask):
            return round(100 * float((w * mask).sum()) / T, 1)

        team = pm.get("team")
        # pickups: which pad (by location, see pad_map); without a pad layout,
        # big if the gain is large or it tops the tank up from below 88
        big = small = stolen = 0
        overfill = 0.0
        collected = 0.0
        pad_counts = [0] * len(PADS)
        for e in pickups:
            if e["player"] != i:
                continue
            f = e["frame"]
            before = boost[max(f - 1, 0), i]
            after = boost[min(f + 2, len(boost) - 1), i]
            j = pads.get(e["pad"]) if pads else None
            if j is not None:
                pad_counts[j] += 1
                is_big = bool(PADS[j][2])
            else:
                is_big = (after - before) >= 20 or (after >= 99 and before < 88)
            if is_big:
                big += 1
                collected += 100 - before
                overfill += before
            else:
                small += 1
                gain = min(12.0, 100 - before)
                collected += gain
                overfill += 12 - gain
            y = PADS[j][1] if j is not None else pos[f, i, 1]
            if (team == 0 and y > 0) or (team == 1 and y < 0):
                stolen += 1
        minutes = T / 60
        out.append({
            "name": pm["name"], "id": pm.get("id"), "team": team, "seconds": round(T, 1),
            "bpm": round(collected / minutes, 1), "avg_boost": round(float((w * boost[:, i]).sum()) / T, 1),
            "pct_zero": frac(boost[:, i] <= 0.5), "pct_full": frac(boost[:, i] >= 99.5),
            "big_pads": big, "small_pads": small, "stolen": stolen, "overfill": round(overfill),
            "avg_speed": round(float((w * speed).sum()) / T), "pct_supersonic": frac(speed >= SUPERSONIC),
            "pct_boost_speed": frac((speed >= BOOST_SPEED) & (speed < SUPERSONIC)), "pct_slow": frac(speed < BOOST_SPEED),
            "pct_ground": frac(z <= GROUND_Z), "pct_low_air": frac((z > GROUND_Z) & (z < HIGH_AIR_Z)), "pct_high_air": frac(z >= HIGH_AIR_Z),
            "jumps": _rising(state[:, i], X.JUMP), "double_jumps": _rising(state[:, i], X.DOUBLE_JUMP), "dodges": _rising(state[:, i], X.DODGE),
            "powerslide_s": round(float((w * (((state[:, i] & X.HANDBRAKE) > 0) & (z <= GROUND_Z))).sum()), 1),
            "powerslides": _count_starts(((state[:, i] & X.HANDBRAKE) > 0) & (z <= GROUND_Z) & (w > 0), t),
            "distance_km": round(float(np.linalg.norm(np.diff(pos[:, i], axis=0), axis=1)[present[1:] & present[:-1]].clip(0, 500).sum()) / 100000, 2),
            "demos": sum(1 for e in demos if e["attacker"] == i), "demoed": sum(1 for e in demos if e["victim"] == i),
            "ball_hits": sum(1 for _f, pi in touch_list if pi == i),
            "pads": pad_counts if pads else None,
        })
    pos_players, rotation, heats = RP.compute(meta, A, dt)
    ball_players, ball_team, shots = RB.compute(meta, A, events, dt, touch_list)
    for tm, d in ball_team.items():
        rotation.setdefault(tm, {}).update(d)
    for i, r in enumerate(out):
        if i < len(ball_players) and "bpm" in r:
            r.update(ball_players[i])
        if i < len(pos_players) and "bpm" in r:
            r.update(pos_players[i])
            if heats[i]:
                r["heat"] = heats[i]
    return {"version": STATS_VERSION, "seconds": round(float(t[-1] - t[0]), 1) if len(t) else 0, "players": out,
            "rotation": rotation, "shots": shots, "timeline": timeline(meta, A, events, dt, touch_list)}


def attach(match: dict, track_path: str) -> dict:
    """Store computed stats on each matching player (by account id, then name)."""
    rs = compute(track_path)
    by_id = {p.get("id"): p for p in rs["players"] if p.get("id")}
    by_name = {p["name"]: p for p in rs["players"]}
    for t in match.get("teams", []):
        for p in t.get("players", []):
            r = by_id.get(p.get("platform_id")) or by_name.get(p.get("replay_name") or p.get("name"))
            if r:
                p["rstats"] = {k: v for k, v in r.items() if k not in ("name", "id", "team", "heat", "pads")}
                if r.get("heat"):
                    p["rheat"] = r["heat"]
                if r.get("pads"):
                    p["rpads"] = r["pads"]
                else:
                    p.pop("rpads", None)
                for k in ("ball_hits", "demos"):  # these only ever come from replays
                    if k in r:
                        p.setdefault("stats", {})[k] = r[k]
    for k, label in (("ball_hits", "BALL HITS"), ("demos", "DEMOS")):
        if not any(c.get("key") == k for c in match.get("columns", [])):
            match.setdefault("columns", []).append({"key": k, "label": label})
    for t in match.get("teams", []):
        rot = rs.get("rotation", {}).get(0 if t.get("color") == "blue" else 1)
        if rot:
            t["rotation"] = rot
        else:
            t.pop("rotation", None)
    match["rshots"] = rs.get("shots") or []
    tl = rs.get("timeline") or {}
    match["rgoals"] = tl.get("goals") or []
    match["overtime"] = bool(tl.get("overtime"))
    match["rmomentum"] = tl.get("momentum") or []
    match["rstats_version"] = STATS_VERSION
    return match
