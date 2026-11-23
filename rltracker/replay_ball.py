"""Phase 4: ball play from a track file — aerial touches, possession, shots
(speed and location), clears, centres and kickoffs.

Touches come from replay_stats.touches() (the ball's velocity jumps while a car
is touching it). Everything is worked out from the recording, so it's an
estimate: replays don't say who touched the ball, or what the game counted as a shot.
"""
from __future__ import annotations

import numpy as np

from . import replay_extract as X

FIELD_X, FIELD_Y = 4096, 5120
THIRD = FIELD_Y / 3
GOAL_HALF_W, GOAL_H = 893, 642
BALL_R = 93
GRAVITY = 650          # uu/s²
KMH = 0.036            # uu/s → km/h
MAX_BALL = 6000        # uu/s, the game's ball speed cap

GROUP = [("aerial_touches", "Aerial touches"), ("pct_possession", "% possession"), ("max_hit_speed", "Fastest hit (km/h)"), ("crossbars", "Crossbar hits"),
         ("avg_shot_speed", "Avg shot speed (km/h)"), ("max_shot_speed", "Hardest shot (km/h)"), ("max_goal_speed", "Hardest goal (km/h)"),
         ("clears", "Clears"), ("centres", "Centers"), ("kickoffs", "Kickoffs taken"), ("kickoff_wins", "Kickoffs won")]
TEAM = [("pct_possession", "% possession"), ("pct_kickoffs_won", "% kickoffs won"), ("kickoff_goals", "Goals within 5 s of kickoff")]


def _kickoffs(bpos, bvel, t):
    """Frame index where each kickoff is released (ball resting on the centre spot)."""
    still = (np.abs(bpos[:, 0]) < 5) & (np.abs(bpos[:, 1]) < 5) & (np.linalg.norm(bvel, axis=1) < 5)
    out = []
    for f in range(1, len(still)):
        if still[f - 1] and not still[f]:
            out.append(f)
    return out


def _contacts(A, i, bpos, bvel, t, dist=280.0, dvmin=60.0):
    """Frames where player i touched the ball: their car right at the ball while its velocity changes.
    Looser than the ball-hit detector, so soft touches and dribbles count too (one frame per contact)."""
    pres = (A["car_state"][:, i] & X.PRESENT) > 0
    d = np.linalg.norm(A["car_pos"][:, i].astype(np.float64) - bpos, axis=1)
    d[~pres] = 1e9
    dv = np.r_[0.0, np.linalg.norm(np.diff(bvel, axis=0), axis=1)]
    jump = np.r_[0.0, np.linalg.norm(np.diff(bpos, axis=0), axis=1)]
    groups: list[list[int]] = []
    for f in np.flatnonzero((d < dist) & (dv > dvmin) & (jump < 400)):
        if groups and t[f] - t[groups[-1][-1]] < 0.25:
            groups[-1].append(int(f))
        else:
            groups.append([int(f)])
    return [max(g, key=lambda f: dv[f]) for g in groups]


def _flight(p, v, s, horizon=5.0, h=1 / 60):
    """Follow the ball from (p, v) towards the opponents' goal (team-relative sign s), bouncing off the
    floor, ceiling and side walls → (miss distance from the goal mouth, seconds) or None if it turns back."""
    x, y, z = p[0] * s, p[1] * s, p[2]
    vx, vy, vz = v[0] * s, v[1] * s, v[2]
    if vy <= 0:
        return None
    tt = 0.0
    while tt < horizon:
        x += vx * h; y += vy * h; z += vz * h
        vz -= GRAVITY * h
        tt += h
        if z < BALL_R:
            z, vz = BALL_R, abs(vz) * 0.6
        elif z > 2044 - BALL_R:
            z, vz = 2044 - BALL_R, -abs(vz) * 0.6
        if abs(x) > 4096 - BALL_R:
            x, vx = np.sign(x) * (4096 - BALL_R), -vx * 0.6
        if y >= FIELD_Y - BALL_R:
            return max(0.0, abs(x) - (GOAL_HALF_W + BALL_R * 0.3)) + max(0.0, z - (GOAL_H + BALL_R * 0.3)), tt
    return None


def ot_mask(clock: np.ndarray) -> np.ndarray:
    """Frames in overtime: the game clock reaches 0:00 and then counts up again."""
    ot = np.zeros(len(clock), bool)
    zero = np.flatnonzero(clock <= 0)
    if len(zero):
        z0 = int(zero[0])
        later = np.flatnonzero(clock[z0:] > 0)
        if len(later):
            ot[z0 + int(later[0]) - 1:] = True
    return ot


def compute(meta: dict, A: dict, events: list, dt: np.ndarray, touch_list: list[tuple[int, int]]):
    """→ (per-player stats, per-team stats {team: {...}}, shots [...])"""
    P = A["car_state"].shape[1] if A["car_state"].ndim == 2 else 0
    if not P or len(dt) < 2:
        return [], {}, []
    t = A["t"].astype(np.float64)
    bpos = A["ball_pos"].astype(np.float64)
    bvel = A["ball_vel"].astype(np.float64)
    F = len(t)
    clock = A["clock"].astype(np.int64) if "clock" in A else None
    ot = ot_mask(clock) if clock is not None else None
    teams = [meta["players"][i].get("team") for i in range(P)]
    names = [meta["players"][i].get("name") for i in range(P)]
    sgn = [(-1.0 if tm == 1 else 1.0) for tm in teams]  # team-relative: own goal at -y
    goals = sorted((e for e in events if e["type"] == "goal"), key=lambda e: e["frame"])
    kicks = _kickoffs(bpos, bvel, t)
    st = [{"aerial_touches": 0, "possession_s": 0.0, "shots": [], "goal_speeds": [], "clears": 0, "centres": 0,
           "kickoffs": 0, "kickoff_wins": 0, "max_hit": 0.0, "crossbars": 0} for _ in range(P)]

    def after(f, k=3):  # ball state just after a contact
        g = min(f + k, F - 1)
        return bpos[g], bvel[g]

    # ---- possession: time from each touch to the next touch, goal or kickoff reset
    stops = sorted([e["frame"] for e in goals] + kicks)
    touches = sorted(touch_list)
    for n, (f, i) in enumerate(touches):
        end = touches[n + 1][0] if n + 1 < len(touches) else F - 1
        nxt = next((s for s in stops if s > f), None)
        if nxt is not None and nxt < end:
            end = nxt
        st[i]["possession_s"] += float(dt[f:end].sum())

    # ---- per touch: aerial, clear, centre, how fast it sent the ball
    for n, (f, i) in enumerate(touches):
        s = sgn[i]
        sp = float(np.linalg.norm(bvel[f + 1:min(f + 6, F)], axis=1).max(initial=0.0))
        if sp <= MAX_BALL:  # anything faster is a reset/teleport, not a hit
            st[i]["max_hit"] = max(st[i]["max_hit"], sp)
        if bpos[f, 2] > 300:
            st[i]["aerial_touches"] += 1
        p, v = after(f)
        x, y = p[0] * s, p[1] * s
        vx, vy = v[0] * s, v[1] * s
        if y < -THIRD and vy > 800:
            st[i]["clears"] += 1
        if y > THIRD and abs(x) > 1500 and vx * np.sign(x) < -500:
            st[i]["centres"] += 1

    # ---- crossbar hits: the ball bounces sharply right at the bar (front edge of the goal, 642 uu up,
    # between the posts) with no car near it. Credited to whoever touched it last, if it was the
    # other team's crossbar (a shot off the bar). Worked out from the ball's path, so it's an estimate.
    dvb = np.r_[0.0, np.linalg.norm(np.diff(bvel, axis=0), axis=1)]
    jmp = np.r_[0.0, np.linalg.norm(np.diff(bpos, axis=0), axis=1)]
    ay = np.abs(bpos[:, 1])
    d_bar = np.hypot(ay - FIELD_Y, bpos[:, 2] - GOAL_H)
    cand = (d_bar < BALL_R + 45) & (ay < FIELD_Y + 30) & (np.abs(bpos[:, 0]) < GOAL_HALF_W + 40) & (dvb > 300) & (jmp < 400)
    pres_all = (A["car_state"] & X.PRESENT) > 0
    last_bar = -9.0
    tf = [f for f, _i in touches]
    for f in np.flatnonzero(cand):
        dc = np.linalg.norm(A["car_pos"][f].astype(np.float64) - bpos[f], axis=1)
        dc[~pres_all[f]] = 1e9
        if dc.min() < 300 or t[f] - last_bar < 0.4:
            continue
        last_bar = t[f]
        k = int(np.searchsorted(tf, f)) - 1
        if k < 0 or t[f] - t[touches[k][0]] > 5:
            continue
        i = touches[k][1]
        if np.sign(bpos[f, 1]) == sgn[i]:  # the bar of the goal they attack
            st[i]["crossbars"] += 1

    # ---- shots. Newer replays record the moment the game counted each shot ("shot" events): the shot
    # is the shooter's last contact with the ball before that. Older track files don't have them, so
    # each player's contacts are ranked by how surely the ball was heading in, and the best ones are
    # taken up to the shots the scoreboard gave them.
    shots = []
    cont = [_contacts(A, i, bpos, bvel, t) for i in range(P)]
    shot_ev = [e for e in events if e["type"] == "shot" and e.get("player") is not None and 0 <= e["player"] < P]
    exact = bool(shot_ev)

    def add(f, i):
        p, v = after(f)
        sh = {"t": round(float(t[f] - t[0]), 1), "frame": int(f), "player": i, "name": names[i], "team": teams[i],
              "x": int(bpos[f, 0]), "y": int(bpos[f, 1]), "speed": round(float(np.linalg.norm(v)) * KMH), "goal": False}
        if clock is not None:
            sh["clock"] = int(clock[f])
            sh["ot"] = bool(ot[f])
        if not exact:
            sh["est"] = True  # location estimated (older recording)
        st[i]["shots"].append(sh)
        shots.append(sh)
        return sh

    def heading_in(f, i):
        """(miss distance from the goal mouth, seconds to the goal line), or None if not heading at goal."""
        p, v = after(f)
        return _flight(p, v, sgn[i])

    def last_contact(i, lo, hi):
        """The shooter's contact in frames lo..hi that best fits a shot (latest one heading in, else the latest)."""
        cs = [f for f in cont[i] if lo <= f <= hi] or [f for f, j in touches if j == i and lo <= f <= hi]
        if not cs:
            return None
        good = [f for f in cs if (h := heading_in(f, i)) and h[0] <= 600]
        return (good or cs)[-1]

    if shot_ev:
        for e in shot_ev:
            fe, i = e["frame"], e["player"]
            lo = int(np.searchsorted(t, t[fe] - 4.0))
            prev_k = max((k for k in kicks if k <= fe), default=0)
            f = last_contact(i, max(lo, prev_k), min(fe + 2, F - 1))
            if f is None:  # contact not seen: put it where the ball was a moment before the game counted it
                f = max(int(np.searchsorted(t, t[fe] - 0.9)), prev_k)
                sh = add(f, i)
                sh["est"] = True
            else:
                add(f, i)

    # ---- goals: mark the scoring shot (add one if none was found)
    for g in goals:
        gf, team = g["frame"], g["team"]
        scorer = g.get("player")
        gspeed = round(float(np.linalg.norm(bvel[max(gf - 1, 0)])) * KMH)
        if scorer is None or not (0 <= scorer < P):
            continue
        st[scorer]["goal_speeds"].append(gspeed)
        prev_k = max((k for k in kicks if k <= gf), default=0)
        sh = next((x for x in reversed(st[scorer]["shots"]) if prev_k <= x["frame"] <= gf and not x["goal"]), None)
        if sh is None:
            f = last_contact(scorer, prev_k, gf)
            if f is None or teams[scorer] != team:  # the scorer's touch wasn't seen: no shot location for this goal
                continue
            sh = add(f, scorer)
        sh["goal"] = True
        sh["goal_speed"] = gspeed

    if not shot_ev:  # older recording: fill up to each player's scoreboard shots
        for i in range(P):
            hs = meta["players"][i].get("header_stats") or {}
            want = hs.get("shots")
            cand = []
            for f in cont[i]:
                h = heading_in(f, i)
                if not h:
                    continue
                miss, tg = h
                tier = 0 if miss <= 50 and tg <= 3.0 else 1 if miss <= 150 and tg <= 3.5 else 2 if miss <= 600 and tg <= 5.0 else None
                if tier is not None and (want is not None or tier == 0):
                    cand.append((tier, tg, f))
            cand.sort()
            n_have = len(st[i]["shots"])
            for _tier, _tg, f in cand:
                if want is not None and n_have >= want:
                    break
                if all(abs(t[f] - t[x["frame"]]) > 1.0 for x in st[i]["shots"]):
                    add(f, i)
                    n_have += 1
    shots.sort(key=lambda s: s["frame"])

    # ---- kickoffs: who went (first touch for each team), who won (where the ball is 2 s later)
    team_k = {0: {"kickoffs": 0, "won": 0, "goals": 0}, 1: {"kickoffs": 0, "won": 0, "goals": 0}}
    for k in kicks:
        first = [(f, i) for f, i in touches if f >= k and t[f] - t[k] <= 5.0]
        g5 = next((g for g in goals if t[k] <= t[g["frame"]] <= t[k] + 5), None)
        if not first:
            if g5:  # a straight-in kickoff goal whose touch wasn't detected
                for tm in (0, 1):
                    team_k[tm]["kickoffs"] += 1
                    team_k[tm]["won"] += g5["team"] == tm
                team_k[g5["team"]]["goals"] += 1
            continue
        f0 = first[0][0]
        went = {}
        for f, i in first:
            if t[f] - t[f0] > 1.5:  # only the opening challenge
                break
            went.setdefault(teams[i], i)
        later = np.searchsorted(t, t[f0] + 2.0)
        if later >= F:
            continue
        by = bpos[later, 1]
        winner = 0 if by > 300 else 1 if by < -300 else None
        for tm, i in went.items():
            st[i]["kickoffs"] += 1
            if winner == tm:
                st[i]["kickoff_wins"] += 1
        for tm in (0, 1):
            team_k[tm]["kickoffs"] += 1
            team_k[tm]["won"] += winner == tm
        if g5:
            team_k[g5["team"]]["goals"] += 1

    total_pos = sum(x["possession_s"] for x in st) or 1.0
    players = []
    for i in range(P):
        x = st[i]
        sp = [s["speed"] for s in x["shots"]]
        players.append({
            "aerial_touches": x["aerial_touches"], "pct_possession": round(100 * x["possession_s"] / total_pos, 1),
            "tracked_shots": len(x["shots"]), "avg_shot_speed": round(sum(sp) / len(sp)) if sp else None,
            "max_shot_speed": max(sp) if sp else None, "max_goal_speed": max(x["goal_speeds"]) if x["goal_speeds"] else None,
            "clears": x["clears"], "centres": x["centres"], "kickoffs": x["kickoffs"], "kickoff_wins": x["kickoff_wins"],
            "crossbars": x["crossbars"],
            "max_hit_speed": max([round(x["max_hit"] * KMH)] * bool(x["max_hit"]) + sp + x["goal_speeds"], default=None),
        })
    team = {}
    for tm in (0, 1):
        pos = sum(st[i]["possession_s"] for i in range(P) if teams[i] == tm)
        k = team_k[tm]
        team[tm] = {"pct_possession": round(100 * pos / total_pos, 1),
                    "pct_kickoffs_won": round(100 * k["won"] / k["kickoffs"], 1) if k["kickoffs"] else None,
                    "kickoff_goals": k["goals"]}
    # ---- each shot's flight: from the shot until a goal, the next touch, a kickoff reset or 4 s.
    # Stored with the shot (absolute field coordinates), so it stays after the replay is gone.
    goal_frames = [(g["frame"], g["team"]) for g in goals]
    cpos = A["car_pos"].astype(np.float64)
    cpresent = (A["car_state"] & X.PRESENT) > 0

    def deflection(f0, f1):
        """First sudden change of the ball's direction along the ground plane between f0 and f1
        (a touch the touch detector missed, or a wall/post) → (frame, player or None)."""
        for j in range(f0 + 4, min(f1, F - 1) + 1):
            if np.linalg.norm(bvel[j, :2] - bvel[j - 1, :2]) < 450:
                continue
            on = np.flatnonzero(cpresent[j])
            if len(on):
                d = np.linalg.norm(cpos[j, on] - bpos[j], axis=1)
                k = int(d.argmin())
                if d[k] < 320:
                    return j, int(on[k])
            return j, None
        return None

    for s in shots:
        f = s["frame"]
        nt = next(((tf, ti) for tf, ti in touches if tf > f + 3), None)
        lim = min(F - 1, int(np.searchsorted(t, t[f] + 4.0)), nt[0] if nt else F - 1)
        ng0 = next((gf for gf, _tm in goal_frames if gf >= f), None)
        if not s.get("goal"):
            dfl = deflection(f, min(lim, ng0 if ng0 is not None else lim))
            if dfl and dfl[1] is not None and (nt is None or dfl[0] < nt[0]):
                nt = dfl
            elif dfl and dfl[1] is None and (nt is None or dfl[0] < nt[0]):
                nt = None
                lim = dfl[0]  # hit a wall / post: the shot ends there
        ng = next(((gf, tm) for gf, tm in goal_frames if gf >= f), None)
        nk = next((k for k in kicks if k > f), None)
        end = min(F - 1, int(np.searchsorted(t, t[f] + 4.0)), lim)
        how, by = "missed", None
        if nk is not None and nk < end:
            end = nk - 1
        if nt is not None and nt[0] <= end and (ng is None or nt[0] < ng[0]):
            end = nt[0]
            if teams[nt[1]] != s["team"]:
                how, by = "saved", names[nt[1]]
            elif names[nt[1]] == s["name"]:
                how = "followed_up"  # the shooter touched it again (e.g. a dribble shot)
            else:
                how, by = "teammate", names[nt[1]]
        elif ng is not None and ng[0] <= end + 1:
            end = min(ng[0], F - 1)
            if ng[1] != s["team"]:
                how = "missed"
            elif s.get("goal"):
                how = "goal"
            else:  # went in, but the goal is another shot's (a touch on the way the detector missed)
                scorer = next((g.get("player") for g in goals if g["frame"] == ng[0]), None)
                how, by = ("followed_up", None) if scorer == s["player"] or scorer is None else ("teammate", names[scorer])
        else:  # stop where it reaches the back wall
            ys = np.abs(bpos[f:end + 1, 1])
            wall = np.flatnonzero(ys >= FIELD_Y - BALL_R - 20)
            if len(wall):
                end = f + int(wall[0])
        if s.get("goal") and ng0 is not None:  # goals always run to the net, however long or deflected
            end, how, by = min(ng0, F - 1), "goal", None
            if t[end] - t[f] > 5.0:
                # the scorer's final touch was missed: find it (last sudden direction change with one of
                # the scoring team right at the ball) and start the shot there; else show the last 5 s
                f5 = int(np.searchsorted(t, t[end] - 5.0))
                real = None
                for j in range(end - 1, max(f + 4, f5), -1):
                    if np.linalg.norm(bvel[j, :2] - bvel[j - 1, :2]) < 450:
                        continue
                    on = [k for k in np.flatnonzero(cpresent[j]) if teams[k] == s["team"]]
                    if on and min(np.linalg.norm(cpos[j, k] - bpos[j]) for k in on) < 320:
                        real = j
                        break
                if real is not None:
                    f = real
                    s.update(x=int(bpos[f, 0]), y=int(bpos[f, 1]), z=int(bpos[f, 2]),
                             speed=round(float(np.linalg.norm(bvel[min(f + 3, F - 1)])) * KMH),
                             t=round(float(t[f] - t[0]), 1))
                    if clock is not None:
                        s.update(clock=int(clock[f]), ot=bool(ot[f]))
                else:
                    f = f5
                    s["partial"] = True
            via = next((names[ti] for tf, ti in touches if f + 3 < tf < ng0 and teams[ti] != s["team"]), None)
            if via:
                s["via"] = via  # went in off an opponent
        idx = list(range(f, end + 1, 3))
        if idx[-1] != end:
            idx.append(end)
        s["path"] = [[int(bpos[j, 0]), int(bpos[j, 1]), int(bpos[j, 2])] for j in idx]
        s["outcome"] = how
        if by:
            s["by"] = by
        s["z"] = int(bpos[f, 2])
        s["flight_s"] = round(float(t[end] - t[f]), 1)
    for s in shots:  # the other team's shape at the moment of the shot
        f, D = s["frame"], 1 - s["team"]
        sd = 1.0 if D == 0 else -1.0  # the defending team's own goal at -y
        ds = [k for k in range(P) if teams[k] == D and cpresent[min(f, F - 1), k]]
        back = [k for k in ds if cpos[f, k, 1] * sd < bpos[f, 1] * sd]
        s["def_back"] = len(back)
        s["def_n"] = len(ds)
        if back:
            last = min(back, key=lambda k: cpos[f, k, 1] * sd)
            s["def_goal_dist"] = int(max(0.0, cpos[f, last, 1] * sd + FIELD_Y))
            s["def_last"] = names[last]
    for s in shots:
        s.pop("frame", None)
        s.pop("player", None)
    shots.sort(key=lambda s: s["t"])
    return players, team, shots
