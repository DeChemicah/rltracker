"""More team play from a track file: who cuts in on whom, how play changes with the score, spacing,
whiffs, and kickoffs by spawn position. Uses what replay_plays already worked out (runs at the
ball, giveaways). Estimates from car and ball positions."""
from __future__ import annotations

import numpy as np

from . import replay_extract as X
from .replay_ball import THIRD

GROUP = [("whiffs", "Whiffs"), ("air_whiffs", "Whiffs in the air"), ("pct_whiff", "% whiffed"),
         ("lastman_challenges", "Last-man challenges"), ("lowboost_challenges", "Low-boost challenges"),
         ("first_def", "First-defender challenges"), ("pct_first_def_won", "% first-defender challenges won")]
LOWER_BETTER = {"whiffs", "air_whiffs", "pct_whiff", "pct_bunched", "pct_same_side", "lastman_challenges", "lowboost_challenges",
                "pct_unchallenged"}
NEUTRAL: set = {"first_def"}
TEAM_SUM = {"whiffs", "air_whiffs", "lastman_challenges", "lowboost_challenges", "first_def"}
TEAM = [("pct_bunched", "% two of you bunched together"), ("pct_same_side", "% all on one side of the field"),
        ("pct_first_def_won", "% first-defender challenges won"), ("pct_unchallenged", "% attacks into your third nobody challenged")]
LOW_BOOST = 15        # a challenge started with less than this
AIM_DEG = 20          # the ball heading within this angle of a teammate after your touch = a pass to them
REACH = 300           # a run counts as a challenge if the car got this close to the ball (or touched it)

BUNCH_DIST = 600      # two teammates' cars this close = bunched up
SIDE_X = 600          # everyone this far to the same side of the field
WHIFF_NEAR = 600      # a jump/flip started this close to the ball and driving at it = going for it
WHIFF_REACH = 300     # ... and the car got this close to the ball within WHIFF_WIN_S
WHIFF_WIN_S = 0.7
AIR_Z = 300
SOFT_TOUCH = 260      # car centre to ball centre for a soft contact
SOFT_DV = 120         # uu/s change in the ball's velocity between frames
STATES = ("ahead", "tied", "behind")


def _spawn(x: float, sign: float) -> str:
    """Spawn name from the car's x at the countdown. Left/right are from the car's view facing the other
    team's goal: the game's coordinates are left-handed, so +x is to blue's left (and orange's right)."""
    ax = abs(x)
    if ax <= 100:
        return "Back centre"
    side = "left" if x * sign > 0 else "right"
    return f"{'Diagonal' if ax > 1500 else 'Off-centre'} {side}"


def compute(meta: dict, A: dict, events: list, dt: np.ndarray, extra: dict):
    """→ (per-player extras, per-team extras {team: {...}}, match extras {"rconf": [...], "rkick": [...]})"""
    state = A["car_state"]
    P = state.shape[1] if state.ndim == 2 else 0
    t = A["t"].astype(np.float64)
    F = len(t)
    if not P or F < 2:
        return [], {}, {}
    teams = [p.get("team") for p in meta["players"]]
    names = [p.get("name") for p in meta["players"]]
    sgn = np.array([(-1.0 if tm == 1 else 1.0) for tm in teams])
    present = (state & X.PRESENT) > 0
    bpos = A["ball_pos"].astype(np.float64)
    bvel = A["ball_vel"].astype(np.float64)
    cpos = A["car_pos"].astype(np.float64)
    cvel = A["car_vel"].astype(np.float64)
    live = dt > 0
    touches, kicks = extra["touches"], extra["kicks"]
    goals = sorted((e for e in events if e["type"] == "goal"), key=lambda e: e["frame"])

    # score difference for blue at every frame (orange's is the negative)
    diff0 = np.zeros(F, int)
    for g in goals:
        diff0[min(g["frame"] + 1, F):] += 1 if g["team"] == 0 else -1

    def st_of(f: int, tm: int) -> str:
        d = diff0[min(f, F - 1)] * (1 if tm == 0 else -1)
        return "ahead" if d > 0 else "behind" if d < 0 else "tied"

    # ---- who cuts in on whom
    rconf = [{"team": c["team"], "first": names[c["first"]], "second": names[c["second"]], "kind": c["kind"],
              "state": st_of(c["frame"], c["team"])} for c in extra["conflicts"]]

    # ---- by score: time, time last back, and events in each state
    yr = cpos[:, :, 1] * sgn[None, :]
    by_state = [{s: {"s": 0.0, "lb": 0.0, "give": 0, "double": 0, "cut": 0, "held": 0} for s in STATES} for _ in range(P)]
    for tm in (0, 1):
        idx = [i for i in range(P) if teams[i] == tm]
        if not idx:
            continue
        d = diff0 * (1 if tm == 0 else -1)
        lab = np.where(d > 0, 0, np.where(d < 0, 2, 1))
        ym = np.where(present[:, idx], yr[:, idx], np.inf)
        low = ym.min(axis=1)
        for i in idx:
            w = dt * present[:, i]
            lb = w * (yr[:, i] <= low)
            for k, s in enumerate(STATES):
                sel = lab == k
                by_state[i][s]["s"] += float(w[sel].sum())
                by_state[i][s]["lb"] += float(lb[sel].sum())
    for f, i in extra["gives"]:
        by_state[i][st_of(f, teams[i])]["give"] += 1
    for c in extra["conflicts"]:
        by_state[c["second"]][st_of(c["frame"], c["team"])][c["kind"]] += 1

    # ---- spacing
    team_x = {}
    for tm in (0, 1):
        idx = [i for i in range(P) if teams[i] == tm]
        if len(idx) < 2:
            continue
        full = live & present[:, idx].all(axis=1)
        w = dt * full
        T = float(w.sum())
        if T < 5:
            continue
        bunched = np.zeros(F, bool)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                bunched |= np.linalg.norm(cpos[:, idx[a], :2] - cpos[:, idx[b], :2], axis=1) < BUNCH_DIST
        x = cpos[:, idx, 0]
        same = ((x > SIDE_X).all(axis=1) | (x < -SIDE_X).all(axis=1)) if len(idx) >= 3 else np.zeros(F, bool)
        team_x[tm] = {"pct_bunched": round(100 * float((w * bunched).sum()) / T, 1),
                      "pct_same_side": round(100 * float((w * same).sum()) / T, 1) if len(idx) >= 3 else None}

    # ---- whiffs: a jump or flip at the ball that gets close to it but doesn't touch it
    touch_f = {}
    for f, i in touches:
        touch_f.setdefault(i, []).append(f)
    wh = [{"whiffs": 0, "air_whiffs": 0, "attempts": 0} for _ in range(P)]
    on_all = (state & (X.JUMP | X.DODGE)) > 0
    for i in range(P):
        if teams[i] is None:
            continue
        on = on_all[:, i]
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        last = -9.0
        tf = np.array(touch_f.get(i, []), dtype=np.int64)
        for f in starts:
            if not live[f] or not present[f, i] or t[f] - last < 0.5:
                continue
            rel = bpos[f] - cpos[f, i]
            d = float(np.linalg.norm(rel))
            if d > WHIFF_NEAR or float(np.dot(cvel[f, i], rel)) <= 0:
                continue
            g = int(np.searchsorted(t, t[f] + WHIFF_WIN_S))
            reach = float(np.linalg.norm(bpos[f:g + 1] - cpos[f:g + 1, i], axis=1).min(initial=1e9))
            if reach > WHIFF_REACH:
                continue  # not really at the ball (e.g. a flip for speed)
            last = t[f]
            wh[i]["attempts"] += 1
            lo, hi = int(np.searchsorted(t, t[f] - 0.15)), int(np.searchsorted(t, t[f] + WHIFF_WIN_S + 0.2))
            if len(tf) and ((tf >= lo) & (tf <= hi)).any():
                continue
            # softer contacts than the touch detector catches: the ball's velocity changes with this car on it
            hi = min(hi, F - 1)
            if hi > lo + 1:
                dvv = np.linalg.norm(np.diff(bvel[lo:hi + 1], axis=0), axis=1)
                near = np.linalg.norm(bpos[lo + 1:hi + 1] - cpos[lo + 1:hi + 1, i], axis=1) < SOFT_TOUCH
                if ((dvv > SOFT_DV) & near).any():
                    continue
            wh[i]["whiffs"] += 1
            if bpos[f, 2] > AIR_Z:
                wh[i]["air_whiffs"] += 1

    # ---- kickoffs by spawn
    rkick = []
    still = (np.abs(bpos[:, 0]) < 5) & (np.abs(bpos[:, 1]) < 5) & (np.linalg.norm(bvel, axis=1) < 5)
    for k in kicks:
        s0 = k - 1  # back to the start of the countdown, when the cars sit on their spawns
        while s0 > 0 and still[s0 - 1]:
            s0 -= 1
        sp = max(s0, int(np.searchsorted(t, t[k] - 3.2)))  # inside the 3 s countdown, cars on their spawns
        g10 = next((g for g in goals if t[k] <= t[g["frame"]] <= t[k] + 10), None)
        first = [(f, i) for f, i in touches if t[k] <= t[f] <= t[k] + 3.0]
        if not first:
            continue  # not a real kickoff (the ball was reset but nobody played it)
        f0 = first[0][0] if first else None
        later = int(np.searchsorted(t, t[f0] + 2.0)) if f0 is not None else None
        winner = None
        if later is not None and later < F:
            winner = 0 if bpos[later, 1] > 300 else 1 if bpos[later, 1] < -300 else None
        for tm in (0, 1):
            idx = [i for i in range(P) if teams[i] == tm and present[k, i]]
            if not idx:
                continue
            # the kicker: whoever on the team is closest to the ball at the contact (the kickoff frame is the first touch)
            who = min(idx, key=lambda i: float(np.linalg.norm(cpos[k, i] - bpos[k])))
            rkick.append({"team": tm, "who": names[who], "spawn": _spawn(cpos[max(sp, 0), who, 0], sgn[who]),
                          "won": None if winner is None else winner == tm,
                          "goal": None if g10 is None else ("for" if g10["team"] == tm else "against")})

    # ---- challenges: last-man and low-boost, with what happened next
    boost = A["car_boost"].astype(np.float64)
    shot_ev = [(e["frame"], teams[e["player"]]) for e in events if e["type"] == "shot" and e.get("player") is not None and e["player"] < P]
    goal_ev = [(g["frame"], g["team"]) for g in goals]

    def within(f, s, lst, tm):
        hi = t[min(f, F - 1)] + s
        return any(t[min(f, F - 1)] < t[min(x, F - 1)] <= hi and team == tm for x, team in lst)

    def lost_ball(f, tm):
        """The other team had the ball (two touches in a row, or one then a goal) within 3 s after frame f."""
        nxt = [(g, i) for g, i in touches if f < g and t[g] - t[f] <= 3.0]
        for g, i in nxt:
            if teams[i] == tm:
                return False
            after = [x for x in touches if x[0] > g][:1]
            return not after or teams[after[0][1]] != tm
        return False

    chal = [{"lm": 0, "lm_lost": 0, "lm_shot": 0, "lm_goal": 0, "lb": 0, "lb_lost": 0, "lb_shot": 0, "lb_goal": 0} for _ in range(P)]
    for i, rs in (extra.get("runs") or {}).items():
        tm = teams[i]
        mates = [j for j in range(P) if j != i and teams[j] == tm]
        tf = touch_f.get(i, [])
        for a0, a1 in rs:
            end = min(F - 1, int(np.searchsorted(t, t[a1] + 0.3)))
            hit = next((f for f in tf if a0 <= f <= end), None)
            if hit is None:
                dd = np.linalg.norm(bpos[a0:end + 1] - cpos[a0:end + 1, i], axis=1)
                if dd.min(initial=1e9) > REACH:
                    continue
                hit = a0 + int(dd.argmin())
            opp = 1 - tm
            res = (lost_ball(hit, tm), within(hit, 6, shot_ev + goal_ev, opp), within(hit, 8, goal_ev, opp))
            on = [j for j in mates if present[a0, j]]
            ball_y = bpos[a0, 1] * sgn[i]
            if ball_y < -THIRD:
                continue  # in your own third you have to challenge: not a risky one
            last_man = bool(on) and all(yr[a0, i] <= yr[a0, j] for j in on) and ball_y > yr[a0, i]
            if last_man:
                c = chal[i]
                c["lm"] += 1
                c["lm_lost"] += res[0]
                c["lm_shot"] += res[1]
                c["lm_goal"] += res[2]
            if boost[a0, i] < LOW_BOOST:
                c = chal[i]
                c["lb"] += 1
                c["lb_lost"] += res[0]
                c["lb_shot"] += res[1]
                c["lb_goal"] += res[2]

    # ---- first defender: the ball comes into your third off the other team; who challenges first, and how it ends
    fd = [{"n": 0, "won": 0} for _ in range(P)]
    for tm in (0, 1):
        if not any(teams[i] == tm for i in range(P)):
            continue
        sd = 1.0 if tm == 0 else -1.0
        inside = (bpos[:, 1] * sd < -THIRD) & live
        entries = np.flatnonzero(inside & ~np.r_[True, inside[:-1]])
        n_e = won = unch = 0
        last_e = -9.0
        for f in entries:
            if t[f] - last_e < 3.0:
                continue
            prev = [i for g, i in touches if g <= f][-1:]
            if not prev or teams[prev[0]] == tm:
                continue  # we brought it back ourselves: not an attack
            last_e = t[f]
            n_e += 1
            ours = next(((g, i) for g, i in touches if f <= g and t[g] - t[f] <= 4.0 and teams[i] == tm), None)
            if ours is None:
                unch += 1
                continue
            g, i = ours
            fd[i]["n"] += 1
            later = min(F - 1, int(np.searchsorted(t, t[g] + 1.5)))
            ok = not lost_ball(g, tm) and (bpos[later, 1] - bpos[g, 1]) * sd > -300
            if ok:
                fd[i]["won"] += 1
                won += 1
        if n_e:
            team_x.setdefault(tm, {}).update({"pct_first_def_won": round(100 * won / max(1, n_e - unch), 1) if n_e - unch else None,
                                             "pct_unchallenged": round(100 * unch / n_e, 1)})

    # ---- demos: who, where (from the victim's side), and whether a goal followed within 5 s
    rdemos = []
    for e in events:
        if e["type"] != "demo" or e.get("attacker") is None or e.get("victim") is None:
            continue
        a, v, f = e["attacker"], e["victim"], min(e["frame"], F - 1)
        fp = max(0, int(np.searchsorted(t, t[f] - 0.15)))  # the wreck's position is wiped at the demo itself
        if a >= P or v >= P or teams[a] is None or teams[v] is None or teams[a] == teams[v]:
            continue
        yv = cpos[fp, v, 1] * sgn[v]
        rdemos.append({"att": names[a], "vic": names[v], "att_team": teams[a], "x": int(cpos[fp, v, 0] * sgn[v]), "y": int(yv),
                       "third": "def" if yv < -THIRD else "off" if yv > THIRD else "mid",
                       "goal": within(f, 5, goal_ev, teams[a])})

    # ---- passes between teammates, and whether a shot or goal followed
    rpass = []
    for (f1, i), (f2, j) in zip(touches, touches[1:]):
        if i == j or teams[i] is None or teams[i] != teams[j] or t[f2] - t[f1] > 3.0:
            continue
        if np.linalg.norm(bpos[min(f2, F - 1), :2] - bpos[f1, :2]) < 700:
            continue
        tm = teams[i]
        rpass.append({"team": tm, "from": names[i], "to": names[j],
                      "shot": within(f2, 4, shot_ev + goal_ev, tm), "goal": within(f2, 5, goal_ev, tm)})

    # ---- aimed passes: after your touch the ball heads straight at a teammate. Completed if that teammate
    # touches it next within 3 s; failed if the other team gets it first or nobody does. Shots (heading into
    # their goal mouth) and clears from your own third aren't counted as passes.
    raim = []
    for n, (f1, i) in enumerate(touches):
        tm = teams[i]
        if tm is None:
            continue
        g = min(f1 + 3, F - 1)
        v = bvel[g, :2]
        sp = float(np.linalg.norm(v))
        if sp < 500:
            continue
        y_own = bpos[f1, 1] * sgn[i]
        if y_own < -THIRD and bvel[g, 1] * sgn[i] > 800:
            continue  # a clear
        vy_own = bvel[g, 1] * sgn[i]
        if y_own > THIRD and vy_own > 0:
            reach_s = (5120 - y_own) / max(vy_own, 1.0)        # time to their goal line
            if abs(bpos[f1, 0] + bvel[g, 0] * reach_s) < 1100:
                continue  # heading at their goal mouth: a shot
        best = None
        for j in range(P):
            if j == i or teams[j] != tm or not present[g, j]:
                continue
            d = cpos[g, j, :2] - bpos[g, :2]
            dist = float(np.linalg.norm(d))
            if not 800 <= dist <= 5000:
                continue
            ang = float(np.degrees(np.arccos(np.clip(np.dot(v, d) / (sp * dist), -1, 1))))
            if ang < AIM_DEG and (best is None or ang < best[1]):
                best = (j, ang)
        if best is None:
            continue
        j = best[0]
        nxt = touches[n + 1] if n + 1 < len(touches) else None
        if nxt and nxt[1] == i and t[nxt[0]] - t[f1] < 0.4:
            continue  # the same player kept dribbling it: not a pass
        ok = bool(nxt and nxt[1] == j and t[nxt[0]] - t[f1] <= 5.0)
        raim.append({"team": tm, "from": names[i], "to": names[j], "ok": ok})

    players = []
    for i in range(P):
        w = wh[i]
        players.append({"lastman_challenges": chal[i]["lm"], "lowboost_challenges": chal[i]["lb"], "chal": chal[i],
                        "first_def": fd[i]["n"], "pct_first_def_won": round(100 * fd[i]["won"] / fd[i]["n"], 1) if fd[i]["n"] >= 2 else None,
                        "whiffs": w["whiffs"], "air_whiffs": w["air_whiffs"],
                        "pct_whiff": round(100 * w["whiffs"] / w["attempts"], 1) if w["attempts"] >= 3 else None,
                        "by_state": {s: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in d.items()} for s, d in by_state[i].items()}})
    return players, team_x, {"rconf": rconf, "rkick": rkick, "rdemos": rdemos, "rpass": rpass, "raim": raim}
