"""When and how games are won or lost: fatigue and tilt over a session, the score situation,
when goals happen, and time of day. Everything comes from data stored per match at import
(final scores, the goal timeline, the scoreboard), so it works for every stored game."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from . import insights as I

SESSION_GAP_MIN = 45  # same as the Sessions page


def _rec(ms: list[dict]) -> dict:
    w = sum(m["result"] == "W" for m in ms)
    gd = [(I.us(m).get("score") or 0) - (I.them(m).get("score") or 0) for m in ms]
    return {"games": len(ms), "wins": w, "losses": len(ms) - w, "win_rate": I._rate(w, len(ms)),
            "gd": round(sum(gd) / len(gd), 2) if gd else None}


def _score_rel(games: list[dict]) -> dict[int, float]:
    """Each game's score against the player's usual score in that mode (+12 = 12% above)."""
    by = defaultdict(list)
    for m in games:
        v = m["my"]["stats"].get("score")
        if v is not None:
            by[m["playlist"]].append(v)
    avg = {pl: sum(v) / len(v) for pl, v in by.items() if len(v) >= 3 and sum(v)}
    out = {}
    for m in games:
        v, a = m["my"]["stats"].get("score"), avg.get(m["playlist"])
        if v is not None and a:
            out[m["id"]] = 100 * (v / a - 1)
    return out


def _sessions(games: list[dict]) -> list[list[dict]]:
    groups, last = [], None
    for m in games:
        t = datetime.fromisoformat(m["played_at"])
        if last is None or (t - last).total_seconds() > SESSION_GAP_MIN * 60:
            groups.append([])
        groups[-1].append(m)
        last = t
    return groups


def _row(label: str, ms: list[dict], rel: dict) -> dict:
    r = _rec(ms)
    xs = [rel[m["id"]] for m in ms if m["id"] in rel]
    r.update(label=label, score_vs_usual=round(sum(xs) / len(xs), 1) if xs else None)
    return r


# ---------------------------------------------------------------- 1. fatigue and tilt
GAME_NO = [("Games 1–3", 1, 3), ("Games 4–6", 4, 6), ("Games 7–9", 7, 9), ("Games 10–14", 10, 14), ("Game 15+", 15, 999)]
MIN_ROW = 4  # a row needs this many games before it's used in a warning


def fatigue(games: list[dict]) -> dict:
    """Win rate, goal difference and score vs usual by game number in the session, by how
    long the session has run, and right after wins / losses / losing streaks."""
    rel = _score_rel(games)
    by_no, by_len = defaultdict(list), defaultdict(list)
    after = {"after_w": [], "after_l": [], "after_2l": [], "after_3l": [], "after_2w": []}
    nos = []  # (game number in its session, game)
    for g in _sessions(games):
        t0 = datetime.fromisoformat(g[0]["played_at"])
        lose_run = win_run = 0
        for i, m in enumerate(g, 1):
            nos.append((i, m))
            for lab, a, b in GAME_NO:
                if a <= i <= b:
                    by_no[lab].append(m)
            h = (datetime.fromisoformat(m["played_at"]) - t0).total_seconds() / 3600
            by_len["First hour" if h < 1 else "Second hour" if h < 2 else "Third hour" if h < 3 else "Hour 4+"].append(m)
            if i > 1:
                after["after_w" if g[i - 2]["result"] == "W" else "after_l"].append(m)
                if lose_run >= 2:
                    after["after_2l"].append(m)
                if lose_run >= 3:
                    after["after_3l"].append(m)
                if win_run >= 2:
                    after["after_2w"].append(m)
            if m["result"] == "L":
                lose_run, win_run = lose_run + 1, 0
            else:
                lose_run, win_run = 0, win_run + 1
    labels = {"after_w": "After a win", "after_l": "After a loss", "after_2w": "After 2+ wins in a row",
              "after_2l": "After 2+ losses in a row", "after_3l": "After 3+ losses in a row"}
    no_rows = [_row(lab, by_no[lab], rel) for lab, _a, _b in GAME_NO if by_no[lab]]
    len_rows = [_row(lab, by_len[lab], rel) for lab in ("First hour", "Second hour", "Third hour", "Hour 4+") if by_len[lab]]
    streak_rows = [_row(labels[k], after[k], rel) for k in ("after_w", "after_2w", "after_l", "after_2l", "after_3l") if after[k]]
    overall = _row("All games", games, rel)
    return {"games": len(games), "sessions": len(_sessions(games)), "overall": overall, "by_game_no": no_rows,
            "by_length": len_rows, "after": streak_rows, "warnings": _warnings(overall, nos, streak_rows)}


MIN_SPLIT = 8  # games needed on each side of a fatigue split


def _warnings(overall: dict, nos: list[tuple[int, dict]], streak_rows: list[dict]) -> list[dict]:
    """Plain-language patterns worth acting on (needs enough games in each row)."""
    out = []
    base = overall["win_rate"]
    if base is None:
        return out
    # fatigue: the split point where every game after it does clearly worse than every game before it
    best = None
    for k in (3, 5, 7, 9, 12, 15):
        early, late = [m for i, m in nos if i <= k], [m for i, m in nos if i > k]
        if len(early) < MIN_SPLIT or len(late) < MIN_SPLIT:
            continue
        a, b = _rec(early), _rec(late)
        drop = (a["win_rate"] or 0) - (b["win_rate"] or 0)
        if drop >= 15 and (best is None or drop > best[0]):
            best = (drop, k, a, b)
    if best:
        _d, k, a, b = best
        out.append({"kind": "fatigue", "after_game": k,
                    "text": f"From game {k + 1} of a session on you win {b['win_rate']}% ({b['games']} games), against {a['win_rate']}% in games 1–{k}.",
                    "tip": f"Consider a 10-minute break after game {k}, or ending the session there."})
    by = {r["label"]: r for r in streak_rows}
    for lab in ("After 3+ losses in a row", "After 2+ losses in a row"):
        r = by.get(lab)
        if r and r["games"] >= MIN_ROW and base - (r["win_rate"] or 0) >= 10:
            n = 3 if "3+" in lab else 2
            out.append({"kind": "tilt", "after_losses": n,
                        "text": f"{lab} you win {r['win_rate']}% over {r['games']} games (usually {base}%).",
                        "tip": f"After {n} losses in a row, take five minutes off before queueing again."})
            break
    return out


def live_warning(games: list[dict], session: list[dict]) -> str | None:
    """A one-line nudge for the live view when the current session hits a pattern from history."""
    if not session:
        return None
    w = fatigue(games)["warnings"]
    run = 0
    for m in reversed(session):
        if m["result"] != "L":
            break
        run += 1
    for x in w:
        if x["kind"] == "tilt" and run >= x["after_losses"]:
            return f"{run} losses in a row. {x['text']} A short break usually helps."
        if x["kind"] == "fatigue" and len(session) >= x["after_game"]:
            return f"Game {len(session)} of this session. {x['text']}"
    return None


# ---------------------------------------------------------------- 2. the score situation
def situations(games: list[dict]) -> dict:
    """Win rate from each score situation the game passed through (from the goal timeline),
    late leads lost and late comebacks, and the record once the game reached overtime."""
    tl = [m for m in games if m.get("rgoals") is not None and m.get("replay")]
    states = {k: [] for k in ("up2", "up1", "down1", "down2")}
    late_lead_lost = late_saved = 0
    late_lead_games = []
    for m in tl:
        ci = 0 if I.us(m).get("color") == "blue" else 1
        a = b = 0
        seen = set()
        led_late = False
        lead_at_1 = 0
        for g in m["rgoals"]:
            if g["team"] == ci:
                a += 1
            else:
                b += 1
            d = a - b
            late = not g["ot"] and (g["clock"] or 0) <= 60
            if d >= 2:
                seen.add("up2")
            elif d == 1:
                seen.add("up1")
            elif d == -1:
                seen.add("down1")
            elif d <= -2:
                seen.add("down2")
            if late and d > 0:
                led_late = True
            if not late and not g["ot"]:
                lead_at_1 = d  # the score going into the last minute
            if late and d == 0 and g["team"] != ci:  # they equalised in the last minute
                late_lead_lost += 1
            if late and d == 0 and g["team"] == ci:
                late_saved += 1  # we equalised in the last minute
        if led_late or lead_at_1 > 0:
            late_lead_games.append(m)
        for k in seen:
            states[k].append(m)
    labels = {"up2": "Ahead by 2+", "up1": "Ahead by 1", "down1": "Behind by 1", "down2": "Behind by 2+"}
    rows = [dict(_rec(states[k]), label=labels[k], key=k) for k in ("up2", "up1", "down1", "down2") if states[k]]
    ot = [m for m in tl if m.get("overtime")]
    return {"games": len(tl), "rows": rows, "late_equalisers_against": late_lead_lost, "late_equalisers_for": late_saved,
            "led_last_minute": _rec(late_lead_games), "overtime": _rec(ot)}


# ---------------------------------------------------------------- 3. when goals happen
GOAL_WINDOWS = [("kickoff", "Within 10 s of a kickoff"), ("after_for", "Within 30 s after we score"),
                ("after_against", "Within 30 s after they score"), ("last30", "Last 30 s of regulation"),
                ("ot", "Overtime")]
SINCE_BUCKETS = [("0–10 s", 0, 10), ("10–30 s", 10, 30), ("30–60 s", 30, 60), ("1–2 min", 60, 120), ("2 min+", 120, 9999)]


def goal_timing(games: list[dict]) -> dict:
    """Goals for and against by moment: right off a kickoff, just after a goal, the last 30 s,
    overtime, and by time since the last kickoff. Uses the game clock, so goal replays and
    countdowns don't count."""
    tl = [m for m in games if m.get("rgoals") is not None and m.get("replay")]
    win = {k: {"for": 0, "against": 0} for k, _l in GOAL_WINDOWS}
    since = {lab: {"for": 0, "against": 0} for lab, _a, _b in SINCE_BUCKETS}
    total = {"for": 0, "against": 0}
    for m in tl:
        ci = 0 if I.us(m).get("color") == "blue" else 1
        prev_clock, prev_ot, prev_team = 300, False, None
        for g in m["rgoals"]:
            side = "for" if g["team"] == ci else "against"
            clock = int(g["clock"] or 0)
            if g["ot"]:
                s = clock - (prev_clock if prev_ot else 0)
            else:
                s = prev_clock - clock
            s = max(0, s)
            total[side] += 1
            if s <= 10:
                win["kickoff"][side] += 1
            if prev_team is not None and s <= 30:
                win["after_for" if prev_team == ci else "after_against"][side] += 1
            if not g["ot"] and clock <= 30:
                win["last30"][side] += 1
            if g["ot"]:
                win["ot"][side] += 1
            for lab, a, b in SINCE_BUCKETS:
                if a <= s < b:
                    since[lab][side] += 1
                    break
            prev_clock, prev_ot, prev_team = clock, g["ot"], g["team"]
    n = len(tl)
    return {"games": n, "total": total,
            "windows": [{"key": k, "label": lab, **win[k], "diff": win[k]["for"] - win[k]["against"]} for k, lab in GOAL_WINDOWS],
            "since_kickoff": [{"label": lab, **since[lab]} for lab, _a, _b in SINCE_BUCKETS]}


# ---------------------------------------------------------------- 6. time of day, in more detail
def time_detail(games: list[dict]) -> dict:
    """Win rate, goal difference and score vs usual by time of day and day of week, plus a
    day × time-of-day grid."""
    rel = _score_rel(games)

    def hour_b(m):
        h = datetime.fromisoformat(m["played_at"]).hour
        for name, a, b in I.HOUR_BUCKETS:
            if a <= h < b or a <= h + 24 < b:
                return name
        return None

    def day_b(m):
        return I.DAYS[datetime.fromisoformat(m["played_at"]).weekday()]

    hb, db, grid = defaultdict(list), defaultdict(list), defaultdict(list)
    for m in games:
        h, d = hour_b(m), day_b(m)
        if h:
            hb[h].append(m)
            grid[(d, h)].append(m)
        db[d].append(m)
    hours = [b[0] for b in I.HOUR_BUCKETS]
    return {"hours": [_row(h, hb[h], rel) for h in hours if hb[h]],
            "days": [_row(d, db[d], rel) for d in I.DAYS if db[d]],
            "grid": {"cols": [h for h in hours if hb[h]], "rows": [d for d in I.DAYS if db[d]],
                     "cells": {f"{d}|{h}": {"games": len(v), "win_rate": _rec(v)["win_rate"]} for (d, h), v in grid.items()}}}


# ---------------------------------------------------------------- 7. weekly recap
def week_bounds(day: str = "") -> tuple[str, str, str, str]:
    """(start, end, previous start, previous end) of the Monday–Sunday week containing day (default: today)."""
    d = datetime.fromisoformat(day) if day else datetime.now()
    mon = (d - timedelta(days=d.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    nxt, prv = mon + timedelta(days=7), mon - timedelta(days=7)
    f = lambda x: x.strftime("%Y-%m-%dT%H:%M:%S")
    return f(mon), f(nxt - timedelta(seconds=1)), f(prv), f(mon - timedelta(seconds=1))


WEEK_KEYS = ["score", "goals", "assists", "saves", "shots", "shooting_pct", "goal_part", "bpm", "pct_possession",
             "pct_supersonic", "pct_behind_ball", "fifty_wins", "passes", "demos"]


def week_report(day: str = "") -> dict:
    from . import debrief
    s, e, ps, pe = week_bounds(day)
    rep = debrief.session_report(s, e)
    rep["kind"] = "week"
    nxt = (datetime.fromisoformat(e) + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
    rep["week"] = {"start": s, "end": e, "prev_start": ps, "prev_end": pe, "next_start": nxt,
                   "is_current": nxt > datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}
    prev, *_ = I.load(start=ps, end=pe)
    cur, *_ = I.load(start=s, end=e)
    rep["prev"] = dict(_rec(prev), goals_for=sum(I.us(m).get("score") or 0 for m in prev),
                       goals_against=sum(I.them(m).get("score") or 0 for m in prev)) if prev else None
    rep["sessions"] = len(_sessions(cur))
    # stat by stat: this week vs last week, for you
    vs = []
    if prev and cur:
        a = {r["key"]: r for r in I.stat_table(start=s, end=e)["rows"]}
        b = {r["key"]: r for r in I.stat_table(start=ps, end=pe)["rows"]}
        for k in WEEK_KEYS:
            x, y = a.get(k), b.get(k)
            if not x or not y or x.get("you") is None or y.get("you") is None:
                continue
            vs.append({"key": k, "label": x["label"], "this": x["you"], "last": y["you"], "pct_unit": x["pct_unit"],
                       "better": None if x["neutral"] or x["you"] == y["you"] else ((x["you"] > y["you"]) != bool(x["lower_better"]))})
    rep["vs_last_week"] = vs
    # best and worst day
    days = defaultdict(list)
    for m in cur:
        days[m["played_at"][:10]].append(m)
    rep["days"] = [dict(_rec(v), date=d) for d, v in sorted(days.items())]
    return rep
