"""Team play from a track file: why each goal went in, how it was scored, and the habits behind
them (double commits, giveaways, recovery time, touch share, first/second/third man, boost when
the ball comes into your third).

All positions are team-relative here (own goal at -y), like replay_position. These are estimates
worked out from car and ball positions; a replay doesn't record intentions.
"""
from __future__ import annotations

import numpy as np

from . import replay_extract as X
from .replay_ball import THIRD, _kickoffs

GROUP = [("double_commits", "Double commits"), ("rotation_cuts", "Rotation cuts"), ("pct_held_back", "% held back"),
         ("giveaways", "Giveaways"), ("recovery_s", "Recovery time (s)"),
         ("touch_share", "% of team touches"), ("boost_def", "Boost when ball enters your third"),
         ("pct_first_man", "% first man"), ("pct_second_man", "% second man"), ("pct_third_man", "% third man")]
LOWER_BETTER = {"double_commits", "rotation_cuts", "giveaways", "recovery_s"}
NEUTRAL = {"touch_share", "pct_first_man", "pct_second_man", "pct_third_man"}
TEAM_SUM = {"double_commits", "rotation_cuts", "giveaways"}

CAUSES = [  # priority order: the first that applies is the goal's main cause
    ("kickoff", "Off the kickoff"), ("demo", "A defender was demoed"), ("double_commit", "Double commit or rotation cut"),
    ("giveaway", "Giveaway in your half"), ("fifty", "Lost a 50/50"), ("rebound", "Rebound off a save or clear"),
    ("nobody_back", "Nobody back"), ("low_boost", "Last defender low on boost"), ("other", "Beaten by a good play")]
KINDS = [("kickoff", "Off the kickoff"), ("rebound", "Rebound / follow-up"), ("counter", "Counter-attack"),
         ("pass", "Passing play"), ("solo", "Solo / individual")]

# A "run" at the ball: within GO_DIST of it and driving at it faster than GO_CLOSING, for GO_MIN_S or
# more (gaps under GO_GAP_S are the same run). Two teammates' runs overlapping while both are close
# (BOTH_NEAR) is a conflict; the one who started later is the one who went in on a teammate.
GO_DIST = 1800
GO_CLOSING = 700     # uu/s straight at the ball
GO_MIN_S = 0.25
GO_GAP_S = 0.3
BOTH_NEAR = 1200
SAME_START_S = 0.2   # runs starting this close together: the one further from the ball was the second
ARRIVE = 320         # both cars this close to the ball = they both got there (a true double commit)
FIFTY_DIST = 300
LOW_BOOST = 20


def _runs(on: np.ndarray, t: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) frame of each stretch where `on` is true, joining gaps under GO_GAP_S, keeping GO_MIN_S+."""
    if not on.any():
        return []
    st = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    en = np.flatnonzero(on & ~np.r_[on[1:], False])
    out = []
    for a, b in zip(st, en):
        if out and t[a] - t[out[-1][1]] < GO_GAP_S:
            out[-1] = (out[-1][0], int(b))
        else:
            out.append((int(a), int(b)))
    return [(a, b) for a, b in out if t[b] - t[a] >= GO_MIN_S]


def commits(A: dict, teams: list, present: np.ndarray, dt: np.ndarray, t: np.ndarray, touches: list, kicks: list):
    """When two teammates go for the same ball. Each player's runs at the ball are found first (close to it and
    driving straight at it); two teammates' runs overlapping while both are near the ball is a conflict, and
    it's put on the player who started their run later (the teammate was already going). Outcome:
      double: both got to the ball (both cars at it, or both touched it within a second);
      cut:    the later one took the ball off the teammate who was already going for it;
      held:   the later one pulled out and let the teammate have it (good rotation).
    → (double commits and cuts as [(frame, team, [players])] for goal causes, every conflict)"""
    F, P = len(t), len(teams)
    bpos = A["ball_pos"].astype(np.float64)
    cpos = A["car_pos"].astype(np.float64)
    cvel = A["car_vel"].astype(np.float64)
    rel = bpos[:, None, :] - cpos                       # car → ball
    dist = np.linalg.norm(rel, axis=2)
    closing = np.einsum("fpk,fpk->fp", cvel, rel) / np.maximum(dist, 1.0)
    kick = np.zeros(F, bool)
    for k in kicks:
        kick[k:int(np.searchsorted(t, t[k] + 3.0))] = True   # kickoff challenges are their own thing
    live = (dt > 0) & ~kick
    runs = {i: _runs(present[:, i] & live & (dist[:, i] < GO_DIST) & (closing[:, i] > GO_CLOSING), t)
            for i in range(P) if teams[i] is not None}
    touch_by = {}
    for f, i in touches:
        touch_by.setdefault(i, []).append(f)
    out, ev = [], []
    for i in runs:
        for j in runs:
            if j <= i or teams[j] != teams[i]:
                continue
            for a0, a1 in runs[i]:
                for b0, b1 in runs[j]:
                    lo, hi = max(a0, b0), min(a1, b1)
                    if lo > hi:
                        continue
                    near = (dist[lo:hi + 1, i] < BOTH_NEAR) & (dist[lo:hi + 1, j] < BOTH_NEAR)
                    if not near.any():
                        continue
                    f0 = lo + int(np.argmax(near))
                    if abs(t[a0] - t[b0]) < SAME_START_S:
                        first, second = (i, j) if dist[f0, i] <= dist[f0, j] else (j, i)
                    else:
                        first, second = (i, j) if a0 < b0 else (j, i)
                    end = min(F - 1, int(np.searchsorted(t, t[max(a1, b1)] + 0.4)))
                    tf = [f for f in touch_by.get(first, []) if f0 <= f <= end]
                    ts = [f for f in touch_by.get(second, []) if f0 <= f <= end]
                    both_at = ((dist[f0:end + 1, first] < ARRIVE) & (dist[f0:end + 1, second] < ARRIVE)).any()
                    if both_at or (tf and ts and abs(t[tf[0]] - t[ts[0]]) < 1.0):
                        kind = "double"
                    elif ts and (not tf or ts[0] < tf[0]):
                        kind = "cut"
                    elif tf:
                        kind = "held"
                    else:
                        continue  # nobody got there (the other team did): no call
                    tm = teams[first]
                    if any(e["team"] == tm and abs(t[e["frame"]] - t[f0]) < 1.0 for e in ev):
                        continue  # same moment already counted
                    ev.append({"frame": f0, "team": tm, "first": first, "second": second, "kind": kind})
                    if kind != "held":
                        out.append((f0, tm, [first, second]))
    ev.sort(key=lambda e: e["frame"])
    out.sort()
    return out, ev, runs


def compute(meta: dict, A: dict, events: list, dt: np.ndarray, touch_list: list) -> tuple[list[dict], list[dict], dict]:
    """→ (per-player stats incl. "give_xy", per-goal records, working data for replay_more)"""
    state = A["car_state"]
    P = state.shape[1] if state.ndim == 2 else 0
    t = A["t"].astype(np.float64)
    F = len(t)
    if not P or F < 2:
        return [], [], {"touches": [], "kicks": [], "gives": [], "conflicts": [], "runs": {}}
    teams = [p.get("team") for p in meta["players"]]
    names = [p.get("name") for p in meta["players"]]
    sgn = np.array([(-1.0 if tm == 1 else 1.0) for tm in teams])
    present = (state & X.PRESENT) > 0
    bpos = A["ball_pos"].astype(np.float64)
    bvel = A["ball_vel"].astype(np.float64)
    cpos = A["car_pos"].astype(np.float64)
    boost = A["car_boost"].astype(np.float64)
    yr = cpos[:, :, 1] * sgn[None, :]                  # each car's y, own goal at -y
    byr = bpos[:, 1][:, None] * sgn[None, :]           # ball y from each player's side
    touches = sorted((f, i) for f, i in touch_list if f < F and i < P and teams[i] is not None)
    kicks = _kickoffs(bpos, bvel, t)
    goals = sorted((e for e in events if e["type"] == "goal"), key=lambda e: e["frame"])
    demos = [e for e in events if e["type"] == "demo"]
    st = [{"double_commits": 0, "rotation_cuts": 0, "held_back": 0, "giveaways": 0, "give_xy": [], "touches": 0, "rec": [], "boost_in": []} for _ in range(P)]

    for f, i in touches:
        st[i]["touches"] += 1

    # ---- double commits and rotation cuts, from each player's runs at the ball
    dcs, conflicts, runs = commits(A, teams, present, dt, t, touches, kicks)
    for c in conflicts:
        st[c["second"]][{"double": "double_commits", "cut": "rotation_cuts", "held": "held_back"}[c["kind"]]] += 1

    # ---- 50/50s (lost ones are used for goal causes)
    fifties = []  # (frame, team that lost it)
    last_c = -9.0
    for f, i in touches:
        b = bpos[f]
        if (abs(b[0]) < 300 and abs(b[1]) < 300 and b[2] < 200) or t[f] - last_c < 0.5:
            continue
        d = np.linalg.norm(cpos[f] - b, axis=1)
        riv = [j for j in range(P) if teams[j] is not None and teams[j] != teams[i] and present[f, j] and d[j] < FIFTY_DIST]
        if not riv:
            continue
        last_c = t[f]
        g = min(int(np.searchsorted(t, t[f] + 1.0)), F - 1)
        dy = bpos[g, 1] - b[1]
        if abs(dy) > 300:
            win_team = 0 if dy > 0 else 1
            fifties.append((f, 1 - win_team))

    # ---- giveaways: your touch (not a shot or a clear), then the other team takes it and keeps it
    gives = []  # (frame, player)
    for n, (f, i) in enumerate(touches[:-1]):
        f2, j = touches[n + 1]
        if teams[j] == teams[i] or t[f2] - t[f] > 4.0:
            continue
        y_own = bpos[f, 1] * sgn[i]
        g = min(f + 4, F - 1)
        vy = bvel[g, 1] * sgn[i]
        if y_own < -THIRD and vy > 800:
            continue  # a clear: giving it up is the point
        if y_own > THIRD:
            continue  # attacking touches and shots: losing it there is cheap
        if any(g_["frame"] > f and g_["frame"] <= f2 for g_ in goals):
            continue
        kept = n + 2 < len(touches) and teams[touches[n + 2][1]] == teams[j] and t[touches[n + 2][0]] - t[f2] < 3.0
        if not kept:
            continue
        gives.append((f, i))
        st[i]["giveaways"] += 1
        st[i]["give_xy"].append([int(bpos[f, 0] * sgn[i]), int(y_own)])

    # ---- recovery: stretches with the ball in your half and you not goal-side of it
    live = dt > 0
    for i in range(P):
        if teams[i] is None:
            continue
        caught = present[:, i] & live & (byr[:, i] < 0) & (yr[:, i] > byr[:, i] + 200)
        if not caught.any():
            continue
        starts = np.flatnonzero(caught & ~np.r_[False, caught[:-1]])
        ends = np.flatnonzero(caught & ~np.r_[caught[1:], False])
        for a, b in zip(starts, ends):
            dur = float(t[b] - t[a])
            if dur >= 0.3:
                st[i]["rec"].append(min(dur, 10.0))

    # ---- boost when the ball comes into your third
    for i in range(P):
        if teams[i] is None:
            continue
        inside = byr[:, i] < -THIRD
        entries = np.flatnonzero(inside & ~np.r_[True, inside[:-1]])
        for f in entries:
            if present[f, i] and live[f]:
                st[i]["boost_in"].append(float(boost[f, i]))

    # ---- first / second / third man: rank by distance to the ball within the team
    roles = np.zeros((P, 3))
    for tm in (0, 1):
        idx = [i for i in range(P) if teams[i] == tm]
        if len(idx) < 2:
            continue
        d = np.linalg.norm(cpos[:, idx] - bpos[:, None, :], axis=2)
        d = np.where(present[:, idx], d, np.inf)
        rank = np.argsort(np.argsort(d, axis=1), axis=1)  # 0 = closest
        ok = live & (present[:, idx].sum(axis=1) == len(idx))
        for k, i in enumerate(idx):
            for r in range(min(3, len(idx))):
                roles[i, r] = float((dt * ok * (rank[:, k] == r)).sum())

    # ---- each goal: why it went in (for the team that conceded) and what kind of goal it was
    recs = []
    for g in goals:
        gf, sc_team = g["frame"], g["team"]
        D = 1 - sc_team
        before = [(f, i) for f, i in touches if f <= gf]
        prev_k = max((k for k in kicks if k <= gf), default=0)
        seq = []  # the scoring team's touches since they last won the ball
        for f, i in reversed(before):
            if f < prev_k:
                break
            if teams[i] != sc_team:
                break
            seq.append((f, i))
        seq.reverse()
        f_shot = seq[-1][0] if seq else max(prev_k, gf - 60)
        f_start = seq[0][0] if seq else f_shot
        win = lambda s: int(np.searchsorted(t, t[gf] - s))
        causes, involved = [], []
        if any(k <= gf for k in kicks) and t[gf] - t[prev_k] <= 10:
            causes.append("kickoff")
        dem = [e for e in demos if e.get("victim") is not None and e["victim"] < P and teams[e["victim"]] == D
               and win(6) <= e["frame"] <= f_shot]
        if dem:
            causes.append("demo")
            involved += [names[e["victim"]] for e in dem]
        dc = [x for x in dcs if x[1] == D and win(7) <= x[0] <= f_shot]
        if dc:
            causes.append("double_commit")
            involved += [names[k] for k in dc[-1][2]]
        gv = [x for x in gives if teams[x[1]] == D and win(7) <= x[0] <= f_shot]
        if gv:
            causes.append("giveaway")
            involved.append(names[gv[-1][1]])
        if any(lt == D and win(6) <= f <= f_shot for f, lt in fifties):
            causes.append("fifty")
        d_touch = [(f, i) for f, i in before if teams[i] == D and win(4) <= f < f_shot]
        if d_touch and seq and len(seq) <= 2:
            causes.append("rebound")
            involved.append(names[d_touch[-1][1]])
        defenders = [i for i in range(P) if teams[i] == D and present[f_shot, i]]
        back = [i for i in defenders if yr[f_shot, i] < byr[f_shot, i]]
        if defenders and not back:
            causes.append("nobody_back")
        elif back:
            last = min(back, key=lambda i: yr[f_shot, i])
            if boost[f_shot, last] < LOW_BOOST:
                causes.append("low_boost")
                involved.append(names[last])
        if not causes:
            causes.append("other")
        # the scoring team's side
        if "kickoff" in causes:
            kind = "kickoff"
        else:
            shots_before = [f for f, i in before if teams[i] == D and win(5) <= f < f_shot]
            if shots_before and len(seq) <= 2 and bpos[f_shot, 1] * (1 if sc_team == 0 else -1) > THIRD:
                kind = "rebound"
            elif seq and bpos[f_start, 1] * (1 if sc_team == 0 else -1) < 0 and t[gf] - t[f_start] <= 8:
                kind = "counter"
            elif len({i for _f, i in seq[-3:]}) >= 2:
                kind = "pass"
            else:
                kind = "solo"
        recs.append({"team": sc_team, "clock": int(A["clock"][min(gf, F - 1)]) if "clock" in A else None,
                     "cause": causes[0], "causes": causes, "involved": sorted(set(involved)), "kind": kind,
                     "def_boost": {names[i]: int(boost[f_shot, i]) for i in defenders}})

    out = []
    for i in range(P):
        s = st[i]
        if teams[i] is None:
            out.append({})
            continue
        team_t = sum(st[j]["touches"] for j in range(P) if teams[j] == teams[i]) or 1
        rt = roles[i].sum()
        n_team = sum(1 for j in range(P) if teams[j] == teams[i])
        n_conf = s["double_commits"] + s["rotation_cuts"] + s["held_back"]
        out.append({
            "double_commits": s["double_commits"], "rotation_cuts": s["rotation_cuts"],
            "pct_held_back": round(100 * s["held_back"] / n_conf, 1) if n_conf else None, "giveaways": s["giveaways"],
            "recovery_s": round(sum(s["rec"]) / len(s["rec"]), 2) if s["rec"] else None,
            "touch_share": round(100 * s["touches"] / team_t, 1),
            "boost_def": round(sum(s["boost_in"]) / len(s["boost_in"]), 1) if s["boost_in"] else None,
            "pct_first_man": round(100 * roles[i, 0] / rt, 1) if rt > 5 else None,
            "pct_second_man": round(100 * roles[i, 1] / rt, 1) if rt > 5 else None,
            "pct_third_man": round(100 * roles[i, 2] / rt, 1) if rt > 5 and n_team >= 3 else None,
            "give_xy": s["give_xy"],
        })
    return out, recs, {"touches": touches, "kicks": kicks, "gives": gives, "conflicts": conflicts, "runs": runs}
