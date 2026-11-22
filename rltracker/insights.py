"""Deeper analysis built on the stored matches.

Everything here works from *your* finished games (hidden and duplicate
matches excluded). Goals for a team always use the final score, so own
goals are counted.
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime

from . import stats, store
from .tracked import TRACKED_STATS

HOUR_BUCKETS = [  # (label, start hour inclusive, end hour exclusive), wraps past midnight
    ("Morning", 6, 12), ("Afternoon", 12, 17), ("Evening", 17, 22),
    ("Late night", 22, 26), ("After 2 AM", 2, 6),
]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------- loading
def load(size: str = "all", ranked: str = "all", include_unfinished: bool = False, who: str = "me", start: str = "", end: str = "",
         tag: str = ""):
    cfg = store.load_config()
    ms = store.all_matches()
    me = stats.detect_me([m for m in ms if not m["hidden"]], cfg)
    stats.mark_duplicates(ms)
    aliases = cfg.get("aliases") or []
    out = []
    for m in ms:
        if m["hidden"] or m.get("duplicate_of"):
            continue
        stats.enrich(m, me)
        stats.apply_aliases(m, aliases, me)
        stats.reanchor(m, who)
        if m["my"] is None or len(m["teams"]) != 2:
            continue
        if not include_unfinished and m["result"] not in ("W", "L"):
            continue
        if size != "all" and m["playlist"] != size:
            continue
        if ranked == "ranked" and not m.get("ranked"):
            continue
        if ranked == "casual" and m.get("ranked"):
            continue
        if start and m["played_at"] < start or end and m["played_at"] > end:
            continue
        if tag and tag not in (m.get("tags") or []):
            continue
        out.append(m)
    out.sort(key=lambda m: m["played_at"])
    return out, me, cfg, aliases


def us(m):
    return m["teams"][m["my_team"]]


def them(m):
    return m["teams"][1 - m["my_team"]]


def total(team: dict, k: str):
    if k == "goals" and team.get("score") is not None:
        return team["score"]
    vals = [p["stats"].get(k) for p in team["players"] if p["stats"].get(k) is not None]
    return sum(vals) if vals else None


def label_of(p: dict, me_display: str | None) -> str:
    if p.get("key") == "me":
        return me_display or p["name"]
    return p.get("alias") or p["name"]


def _rate(w, n):
    return round(100 * w / n, 1) if n else None


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def game_row(m: dict) -> dict:
    """Compact per-game record used by several views."""
    mine = m["my"]
    return {
        "id": m["id"], "t": m["played_at"], "result": m["result"], "playlist": m["playlist"],
        "ranked": bool(m.get("ranked")), "for": us(m).get("score"), "against": them(m).get("score"),
        "my": {k: mine["stats"].get(k) for k in TRACKED_STATS},
        "us": {k: total(us(m), k) for k in TRACKED_STATS},
        "them": {k: total(them(m), k) for k in TRACKED_STATS},
        "mates": [p.get("alias") or p["name"] for p in us(m)["players"] if p is not mine],
        "mvp_me": m.get("mvp") == mine["name"],
        "size": len(us(m)["players"]) or 1,
    }


# ---------------------------------------------------------------- grading
# Grades compare against a baseline (your usual, or the lobby average). 60 = the
# baseline; +50% on a stat is 100, -50% is 20. Win rate: 50% = 60, 100% = 100.
GRADE_W = {"score": 2, "goals": 1.5, "saves": 1, "shots": 1, "ball_hits": 0.5, "demos": 0.5}
_FLOOR = {"score": 50, "ball_hits": 2}
LETTERS = [(90, "A+", "Outstanding"), (80, "A", "Great"), (68, "B", "Above usual"),
           (52, "C", "About usual"), (40, "D", "Below usual"), (0, "F", "Rough")]


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def _rel(v, base, k):
    if v is None or base is None:
        return None
    return (v - base) / max(abs(base), _FLOOR.get(k, 0.5))


def _stat_pts(rel):
    return None if rel is None else round(_clamp(60 + 80 * rel), 1)


def _win_pts(wr):
    return None if wr is None else round(_clamp(60 + 0.8 * (wr - 50)), 1)


def _letter(pts):
    if pts is None:
        return None
    for cut, l, word in LETTERS:
        if pts >= cut:
            return {"letter": l, "word": word, "pts": round(pts)}


def _stats_pts(avg: dict, base: dict) -> float | None:
    """Weighted grade points of a set of per-game stats against a baseline."""
    num = den = 0.0
    for k, w in GRADE_W.items():
        p = _stat_pts(_rel(avg.get(k), base.get(k), k))
        if p is not None:
            num, den = num + w * p, den + w
    return num / den if den else None


# ---- single-game grades. One game is noisy: a tight midfield battle gives everyone few goals, shots
# and saves, so raw "vs your usual" made a 1-0 win look like an F. A game's grade blends:
#   45% your stats vs your usual, scaled to how busy this game was (everyone else's numbers in it),
#   35% your stats vs the other players in this same game,
#   20% the result (win 80, loss 40).
# Each stat's swing is capped (-60%..+80%) and scored on a gentler scale than the session grade.
GAME_MIX = {"usual": 0.45, "in_game": 0.35, "result": 0.20}
GAME_RESULT_PTS = {"W": 80, "L": 40}


def _soft_stat_pts(rel):
    return None if rel is None else round(_clamp(60 + 60 * max(-0.6, min(0.8, rel))), 1)


def _soft_pts(vals: dict, base: dict) -> float | None:
    num = den = 0.0
    for k, w in GRADE_W.items():
        p_ = _soft_stat_pts(_rel(vals.get(k), base.get(k), k))
        if p_ is not None:
            num, den = num + w * p_, den + w
    return num / den if den else None


def _mode_lobby(pl: str) -> dict:
    """The average player's per-game stats in this mode, over every stored game."""
    model = difficulty_model()
    return model["modes"].get(pl) or {k: (v * (model["factor"].get(pl, {}).get(k) or 1)) if v is not None else None
                                      for k, v in model["base"].items()}


def _paced(usual: dict, players: list[dict], pl: str) -> dict:
    """usual × how busy this game was (these players' per-player numbers vs the mode's average player)."""
    lob = _mode_lobby(pl)
    out = {}
    for k in TRACKED_STATS:
        g = _avg([p["stats"].get(k) for p in players])
        f = max(0.4, min(2.0, g / lob[k])) if g is not None and lob.get(k) else None
        out[k] = usual.get(k) * f if usual.get(k) is not None and f else usual.get(k)
    return out


def _game_grade(vals: dict, paced: dict, in_game: dict, result: str | None) -> tuple[float | None, dict]:
    parts = {"usual": _soft_pts(vals, paced), "in_game": _soft_pts(vals, in_game), "result": GAME_RESULT_PTS.get(result)}
    have = {k: v for k, v in parts.items() if v is not None}
    if not have:
        return None, parts
    w = sum(GAME_MIX[k] for k in have)
    return sum(GAME_MIX[k] * v for k, v in have.items()) / w, {k: (round(v) if v is not None else None) for k, v in parts.items()}


# ---- per-mode baselines: 2s hand out more goals/saves/shots per player than 3s, so every game is
# graded against the usual for its own mode. A mode with too few games borrows the overall
# baseline, scaled by how that mode compares across every stored game (all players).
MIN_MODE_GAMES = 3


def _mode_factors() -> dict:
    """{playlist: {stat: that mode's average per player ÷ the overall average}} over every stored game."""
    m_ = difficulty_model()
    return m_["factor"]


def _bases_by_mode(items: list[tuple[str, dict]], min_n: int = MIN_MODE_GAMES) -> dict:
    """items = (playlist, per-game stats) → {"_all": base, playlist: base or None (too few games)}."""
    out = {"_all": {k: _avg([x.get(k) for _pl, x in items]) for k in TRACKED_STATS}}
    by = defaultdict(list)
    for pl, x in items:
        by[pl].append(x)
    for pl, xs in by.items():
        out[pl] = {k: _avg([x.get(k) for x in xs]) for k in TRACKED_STATS} if len(xs) >= min_n else None
    return out


def _base_for(bases: dict, pl: str, k: str):
    b = bases.get(pl)
    if b and b.get(k) is not None:
        return b[k]
    a = bases["_all"].get(k)
    f = _mode_factors().get(pl, {}).get(k)
    return None if a is None else a * f if f else a


def _expected(bases: dict, playlists: list[str]) -> dict:
    """The baseline for a set of games: each game's own-mode baseline, averaged."""
    return {k: _avg([_base_for(bases, pl, k) for pl in playlists]) for k in TRACKED_STATS}


# ---- team grades: your team's totals against your teams' usual in the same mode. Baselines are kept
# per player (team total ÷ team size) so the per-mode fallback works, then scaled back up.
TEAM_ROT_NOTES = ["pct_possession", "pct_kickoffs_won", "pct_double_up", "pct_no_last_back"]


def _team_pp(totals: dict, size: int) -> dict:
    return {k: (v / size if v is not None else None) for k, v in totals.items()}


def _team_expected(rows_rest: list[dict], targets: list[tuple[str, int]]) -> dict:
    """rows_rest: game_rows to learn from; targets: (playlist, team size) of the games being graded."""
    bases = _bases_by_mode([(r["playlist"], _team_pp(r["us"], r.get("size") or 1)) for r in rows_rest])
    out = {k: _avg([(b * n) if (b := _base_for(bases, pl, k)) is not None else None for pl, n in targets]) for k in TRACKED_STATS}
    return out, bases


def _grade_block(vals: dict, usual: dict) -> tuple[dict, float | None]:
    st = {}
    for k in TRACKED_STATS:
        rel = _rel(vals.get(k), usual.get(k), k)
        st[k] = {"v": vals.get(k), "usual": round(usual[k], 2) if usual.get(k) is not None else None,
                 "change": round(100 * rel) if rel is not None else None, "grade": _letter(_stat_pts(rel))}
    return st, _stats_pts(vals, usual)


def _swing_notes(st: dict, cut: int, prefix: str = "") -> list[dict]:
    ch = sorted(((k, v["change"]) for k, v in st.items() if v["change"] is not None), key=lambda x: -x[1])
    out = []
    if ch and ch[0][1] >= cut:
        out.append({"kind": "up", "text": f"{prefix}{_SL.get(ch[0][0], ch[0][0])} +{ch[0][1]}% vs usual"})
    if ch and ch[-1][1] <= -cut:
        out.append({"kind": "down", "text": f"{prefix}{_SL.get(ch[-1][0], ch[-1][0])} {ch[-1][1]}% vs usual"})
    return out


def team_game_report(m: dict, others: list[dict]) -> dict:
    """Your team in one game: its totals against your teams' usual in the same mode, plus team play."""
    from . import replay_stats as RS
    pl, n = m["playlist"], len(us(m)["players"]) or 1
    vals = {k: total(us(m), k) for k in TRACKED_STATS}
    usual, bases = _team_expected([game_row(o) for o in others], [(pl, n)])
    st, _sp = _grade_block(vals, usual)
    notes = _swing_notes(st, 25)
    opp = {k: total(them(m), k) for k in TRACKED_STATS}
    # same blend as your game grade: vs usual at this game's pace (both teams), vs the other team, the result
    paced = _paced(usual, [p for t in m["teams"] for p in t["players"]], pl)
    sp, parts = _game_grade(vals, paced, opp, m["result"])
    for k in TRACKED_STATS:
        st[k]["grade"] = _letter(_soft_stat_pts(_rel(vals.get(k), paced.get(k), k)))
    words = {"shots": ("Out-shot them", "Out-shot by them"), "saves": ("More saves than them", "Fewer saves than them"),
             "ball_hits": ("More ball hits than them", "Fewer ball hits than them")}
    for k, (more, fewer) in words.items():
        a, b = vals.get(k), opp.get(k)
        if a is not None and b is not None and a != b and max(a, b) >= 3:
            notes.append({"kind": "up" if a > b else "down", "text": f"{more if a > b else fewer} {a}–{b}"})
    rot = us(m).get("rotation") or {}
    if rot:
        same = [us(o).get("rotation") for o in others if us(o).get("rotation") and o["playlist"] == pl]
        pool = same if len(same) >= MIN_MODE_GAMES else [us(o).get("rotation") for o in others if us(o).get("rotation")]
        sw = []
        for k in TEAM_ROT_NOTES:
            v, u = rot.get(k), _avg([r.get(k) for r in pool])
            if v is None or u is None:
                continue
            d = v - u  # these are percentages: compare in points
            good = -d if k in RS.LOWER_BETTER else d
            if abs(d) >= 10:
                sw.append((abs(d), k, d, good))
        for _a, k, d, good in sorted(sw, reverse=True)[:2]:
            notes.append({"kind": "up" if good > 0 else "down",
                          "text": f"{RS.LABELS.get(k) or dict(RS.ROTATION).get(k, k)} {v_fmt(rot.get(k))} ({'+' if d > 0 else ''}{round(d)} pts vs usual)"})
    return {"grade": _letter(sp), "grade_parts": parts, "stats": st, "notes": notes, "estimated": not bases.get(pl), "size": n,
            "players": ["You"] + [label_of(p, None) for p in us(m)["players"] if p is not m["my"]]}


def v_fmt(x):
    return "–" if x is None else f"{x:.0f}%"


def _streaks(results):
    best = {"W": 0, "L": 0}
    kind, n = None, 0
    for r in results:
        n = n + 1 if r == kind else 1
        kind = r
        if r in best:
            best[r] = max(best[r], n)
    return best


def _report(s: dict, rows: list[dict], base: dict, base_wr: float | None) -> dict:
    """Report card for one session: grades vs your usual plus plain-language notes."""
    stats_ = {}
    for k in TRACKED_STATS:
        rel = _rel(s["my_avg"][k], base.get(k), k)
        stats_[k] = {"avg": s["my_avg"][k], "usual": base.get(k),
                     "change": round(100 * rel) if rel is not None else None, "grade": _letter(_stat_pts(rel))}
    sp, wp = _stats_pts(s["my_avg"], base), _win_pts(s["win_rate"])
    overall = None if sp is None and wp is None else (sp if wp is None else wp if sp is None else 0.5 * sp + 0.5 * wp)
    notes = []
    ch = sorted(((k, v["change"]) for k, v in stats_.items() if v["change"] is not None), key=lambda x: -x[1])
    if ch and ch[0][1] >= 15:
        notes.append({"kind": "up", "stat": ch[0][0], "change": ch[0][1]})
    if ch and ch[-1][1] <= -15:
        notes.append({"kind": "down", "stat": ch[-1][0], "change": ch[-1][1]})
    if base_wr is not None and s["win_rate"] is not None:
        d = s["win_rate"] - base_wr
        if abs(d) >= 10:
            notes.append({"kind": "up" if d > 0 else "down",
                          "text": f"Won {s['win_rate']:.0f}% of games, {'up' if d > 0 else 'down'} from your usual {base_wr:.0f}%"})
    st = _streaks([r["result"] for r in rows])
    if st["W"] >= 3:
        notes.append({"kind": "up", "text": f"Best win streak: {st['W']} in a row"})
    if st["L"] >= 3:
        notes.append({"kind": "down", "text": f"Worst losing streak: {st['L']} in a row"})
    early_late = None
    if len(rows) >= 6:
        h = len(rows) // 2
        a, b = rows[:h], rows[h:]

        def wr(xs):
            return _rate(sum(r["result"] == "W" for r in xs), len(xs))

        def sc(xs):
            return _avg([r["my"]["score"] for r in xs])
        early_late = {"early": {"games": len(a), "win_rate": wr(a), "score": sc(a)},
                      "late": {"games": len(b), "win_rate": wr(b), "score": sc(b)}}
        d = (wr(b) or 0) - (wr(a) or 0)
        if d <= -20:
            notes.append({"kind": "down", "text": f"Faded late: {wr(a):.0f}% wins in the first {len(a)} games, {wr(b):.0f}% in the last {len(b)}"})
        elif d >= 20:
            notes.append({"kind": "up", "text": f"Got better as it went: {wr(a):.0f}% wins early, {wr(b):.0f}% late"})
    worst = min(rows, key=lambda r: r["my"]["score"] if r["my"]["score"] is not None else 1e9)
    mates = defaultdict(lambda: [0, 0])
    for r in rows:
        for n in r["mates"]:
            mates[n][0] += 1
            mates[n][1] += r["result"] == "W"
    mate_rows = sorted(({"name": n, "games": g, "wins": w, "win_rate": _rate(w, g)} for n, (g, w) in mates.items()),
                       key=lambda e: -e["games"])
    return {"grade": _letter(overall), "stats_grade": _letter(sp), "win_grade": _letter(wp),
            "stats": stats_, "notes": notes, "early_late": early_late, "streaks": st,
            "worst": {"id": worst["id"], "score": worst["my"]["score"], "t": worst["t"], "result": worst["result"]},
            "mates": mate_rows, "usual_win_rate": base_wr}


# ---------------------------------------------------------------- sessions
def sessions(cutoff_min: int = 45) -> dict:
    games, me, cfg, aliases = load()
    disp = stats.my_display(me, aliases)
    groups: list[list] = []
    last = None
    for m in games:
        t = datetime.fromisoformat(m["played_at"])
        if last is None or (t - last).total_seconds() > cutoff_min * 60:
            groups.append([])
        groups[-1].append(m)
        last = t
    out = []
    for i, g in enumerate(groups):
        rows = [game_row(m) for m in g]
        wins = sum(r["result"] == "W" for r in rows)
        start, end = rows[0]["t"], rows[-1]["t"]
        mins = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
        my_avg = {k: _avg([r["my"][k] for r in rows]) for k in TRACKED_STATS}
        us_avg = {k: _avg([r["us"][k] for r in rows]) for k in TRACKED_STATS}
        them_avg = {k: _avg([r["them"][k] for r in rows]) for k in TRACKED_STATS}
        best = max(rows, key=lambda r: r["my"]["score"] or 0)
        mates = defaultdict(int)
        for r in rows:
            for n in r["mates"]:
                mates[n] += 1
        streak, kind = 0, None
        for r in rows:
            if r["result"] == kind:
                streak += 1
            else:
                kind, streak = r["result"], 1
        s = {
            "n": i + 1, "start": start, "end": end, "minutes": round(mins),
            "games": len(rows), "wins": wins, "losses": len(rows) - wins, "win_rate": _rate(wins, len(rows)),
            "my_avg": my_avg, "us_avg": us_avg, "them_avg": them_avg,
            "diff": {k: (round(us_avg[k] - them_avg[k], 2) if us_avg[k] is not None and them_avg[k] is not None else None) for k in TRACKED_STATS},
            "goals_for": sum(r["for"] or 0 for r in rows), "goals_against": sum(r["against"] or 0 for r in rows),
            "best": {"id": best["id"], "score": best["my"]["score"], "t": best["t"], "result": best["result"]},
            "mvps": sum(r["mvp_me"] for r in rows),
            "mates": sorted(mates, key=lambda n: -mates[n]),
            "end_streak": {"kind": kind, "n": streak},
            "rows": rows, "_games": g,
        }
        out.append(s)
    # report cards: each session against your games outside it
    every = [r for s in out for r in s["rows"]]
    for s in out:
        ids = {r["id"] for r in s["rows"]}
        rest = [r for r in every if r["id"] not in ids]
        if len(rest) < 3:
            rest = every
        bases = _bases_by_mode([(r["playlist"], r["my"]) for r in rest])
        pls = [r["playlist"] for r in s["rows"]]
        base = _expected(bases, pls)
        wr_by = defaultdict(list)
        for r in rest:
            wr_by[r["playlist"]].append(r["result"] == "W")
        wr_all = _rate(sum(r["result"] == "W" for r in rest), len(rest))
        base_wr = _avg([_rate(sum(wr_by[pl]), len(wr_by[pl])) if len(wr_by[pl]) >= MIN_MODE_GAMES else wr_all for pl in pls])
        s["report"] = _report(s, s["rows"], base, base_wr)
        t_usual, t_bases = _team_expected(rest, [(r["playlist"], r.get("size") or 1) for r in s["rows"]])
        t_st, t_sp = _grade_block(s["us_avg"], t_usual)
        t_wp = _win_pts(s["win_rate"])
        t_all = None if t_sp is None and t_wp is None else (t_sp if t_wp is None else t_wp if t_sp is None else 0.5 * t_sp + 0.5 * t_wp)
        s["report"]["team"] = {"grade": _letter(t_all), "stats_grade": _letter(t_sp), "win_grade": _letter(t_wp), "stats": t_st,
                               "notes": _swing_notes(t_st, 15, "Team "),
                               "diff": s["diff"], "estimated": any(not t_bases.get(r["playlist"]) for r in s["rows"])}
        s["report"]["modes"] = {pl: pls.count(pl) for pl in sorted(set(pls))}
        s["report"]["estimated_modes"] = sorted({pl for pl in pls if not bases.get(pl)})  # too few earlier games in that mode
        cg = clutch(s.get("_games", []))
        s["report"]["by_minute"] = cg["by_minute"] if cg["timeline_games"] else None
        s["report"]["minute_games"] = cg["timeline_games"]
        dm = difficulty_model()
        sv = [opp_strength(m, dm) for m in s.pop("_games", [])]
        sv = [v for v in sv if v is not None]
        if sv:
            avg_ = sum(sv) / len(sv)
            s["report"]["difficulty"] = {"strength": round(avg_, 1), "tier": tier_of(avg_, dm)}
        s["summary"] = _session_text(s, disp)
    out.reverse()
    return {"cutoff": cutoff_min, "me": disp, "sessions": out}


def _session_text(s: dict, disp: str | None) -> str:
    """Plain text you can paste into Discord."""
    d = datetime.fromisoformat(s["start"])
    lines = [f"**{disp or 'Session'} — {d:%a %b} {d.day}, {d:%I:%M %p}**".replace(" 0", " "),
             f"{s['wins']}W – {s['losses']}L ({s['win_rate']}%) · {s['games']} games · {s['minutes']} min"]
    a = s["my_avg"]
    lines.append(f"Per game: {a['score']} pts · {a['goals']} G · {a['saves']} Sv · {a['shots']} Sh · {a['ball_hits']} BH · {a['demos']} Demos")
    lines.append(f"Goals {s['goals_for']}–{s['goals_against']} · best game {s['best']['score']} pts · MVPs {s['mvps']}")
    g = (s.get("report") or {}).get("grade")
    if g:
        lines.append(f"Report card: **{g['letter']}** ({g['word'].lower()})")
    if s["mates"]:
        lines.append("With " + ", ".join(s["mates"][:4]))
    return "\n".join(lines)


def live(cutoff_min: int = 45) -> dict:
    """The session in progress (if the last game was within the cutoff)."""
    data = sessions(cutoff_min)
    cur = data["sessions"][0] if data["sessions"] else None
    active = False
    if cur:
        active = (datetime.now() - datetime.fromisoformat(cur["end"])).total_seconds() < cutoff_min * 60
    games, *_ = load()
    return {"me": data["me"], "active": active, "session": cur, "cutoff": cutoff_min,
            "last_game": game_report(games[-1], games[:-1]) if games else None,
            "last_team": team_game_report(games[-1], games[:-1]) if games else None}


# replay stats worth a line on a single game's report card (lower is better for the RS.LOWER_BETTER ones)
SMALL_COUNT_NOTES = {"centres"}  # stats that are usually 0-3 a game: only note a gap of 2+
GAME_REPLAY_NOTES = ["bpm", "pct_supersonic", "aerial_touches", "centres", "pct_possession", "stolen", "overfill", "pct_zero", "demoed"]


def game_report(m: dict, others: list[dict]) -> dict:
    """Report card for one game: your stats against your usual in the same mode (your other games)."""
    from . import replay_stats as RS
    pl = m["playlist"]
    mine = m["my"]["stats"]
    bases = _bases_by_mode([(o["playlist"], o["my"]["stats"]) for o in others])
    usual = _expected(bases, [pl])
    st = {}
    for k in TRACKED_STATS:
        rel = _rel(mine.get(k), usual.get(k), k)
        st[k] = {"v": mine.get(k), "usual": round(usual[k], 2) if usual.get(k) is not None else None,
                 "change": round(100 * rel) if rel is not None else None, "grade": _letter(_stat_pts(rel))}
    rest = [p for t in m["teams"] for p in t["players"] if p is not m["my"]]
    paced = _paced(usual, rest, pl)
    in_game = {k: _avg([p["stats"].get(k) for p in rest]) for k in TRACKED_STATS}
    sp, parts = _game_grade(mine, paced, in_game, m["result"])
    for k in TRACKED_STATS:  # each stat's grade: against your usual at this game's pace
        st[k]["grade"] = _letter(_soft_stat_pts(_rel(mine.get(k), paced.get(k), k)))
    notes = []
    ch = sorted(((k, v["change"]) for k, v in st.items() if v["change"] is not None), key=lambda x: -x[1])
    if ch and ch[0][1] >= 25:
        notes.append({"kind": "up", "text": f"{_SL.get(ch[0][0], ch[0][0])} +{ch[0][1]}% vs your usual"})
    if ch and ch[-1][1] <= -25:
        notes.append({"kind": "down", "text": f"{_SL.get(ch[-1][0], ch[-1][0])} {ch[-1][1]}% vs your usual"})
    # replay stats: the biggest swings against your usual in this mode
    if RS.has(m["my"]):
        same = [o["my"] for o in others if RS.has(o["my"]) and o["playlist"] == pl]
        pool = same if len(same) >= MIN_MODE_GAMES else [o["my"] for o in others if RS.has(o["my"])]
        swings = []
        for k in GAME_REPLAY_NOTES:
            v, u = m["my"]["rstats"].get(k), _avg([p["rstats"].get(k) for p in pool])
            if v is None or u is None or not pool:
                continue
            if k in SMALL_COUNT_NOTES and abs(v - u) < 2:  # 0 vs a usual of 1 isn't a story
                continue
            rel = _rel(v, u, k)
            good = -rel if k in RS.LOWER_BETTER else rel
            if abs(rel) >= 0.3:
                swings.append((abs(rel), k, rel, good))
        for _a, k, rel, good in sorted(swings, reverse=True)[:2]:
            notes.append({"kind": "up" if good > 0 else "down",
                          "text": f"{RS.LABELS.get(k, k)} {'+' if rel > 0 else ''}{round(100 * rel)}% vs your usual"})
    team = us(m)["players"]
    team_score = sum(p["stats"].get("score") or 0 for p in team)
    top = max(team, key=lambda p: p["stats"].get("score") or 0)
    gp = stats.goal_participation([(m["my"], us(m))])
    if m.get("mvp") == m["my"]["name"]:
        notes.insert(0, {"kind": "up", "text": "MVP"})
    elif top is m["my"] and len(team) > 1:
        notes.insert(0, {"kind": "up", "text": "Top scorer on your team"})
    return {
        "id": m["id"], "t": m["played_at"], "result": m["result"], "playlist": pl,
        "for": us(m).get("score"), "against": them(m).get("score"), "overtime": bool(m.get("overtime")),
        "grade": _letter(sp), "grade_parts": parts, "stats": st, "notes": notes, "estimated": not bases.get(pl),
        "score_share": round(100 * (mine.get("score") or 0) / team_score) if team_score else None,
        "goal_part": gp["pct"] if gp else None, "difficulty": game_difficulty(m),
        "mates": [label_of(p, None) for p in team if p is not m["my"]],
    }


_SL = {"score": "Score", "goals": "Goals", "assists": "Assists", "ball_hits": "Ball hits", "saves": "Saves", "shots": "Shots", "demos": "Demos"}


# ---------------------------------------------------------------- dashboard extras
def dashboard(size="all", ranked="all", last=0, who="me", start="", end="", tag="") -> dict:
    games, me, cfg, aliases = load(size, ranked, who=who, start=start, end=end, tag=tag)
    if last:
        games = games[-last:]
    rows = [game_row(m) for m in games]
    n = len(rows)

    def bucket_table(title, fn, order):
        b = defaultdict(lambda: [0, 0])
        for r in rows:
            k = fn(r)
            if k is None:
                continue
            b[k][0] += 1
            b[k][1] += r["result"] == "W"
        return {"title": title, "rows": [{"label": k, "games": b[k][0], "wins": b[k][1], "win_rate": _rate(b[k][1], b[k][0])}
                                         for k in order if b[k][0]]}

    def score_b(r):
        v = r["my"]["score"]
        return None if v is None else ("Under 300" if v < 300 else "300–449" if v < 450 else "450+")

    def saves_b(r):
        v = r["my"]["saves"]
        return None if v is None else ("0" if v == 0 else "1" if v == 1 else "2" if v == 2 else "3+")

    def goals_b(r):
        v = r["my"]["goals"]
        return None if v is None else ("0" if v == 0 else "1" if v == 1 else "2+")

    def cmp_b(k):
        def f(r):
            a, b = r["us"][k], r["them"][k]
            if a is None or b is None:
                return None
            return "We had more" if a > b else "Even" if a == b else "They had more"
        return f

    factors = [
        bucket_table("Your score", score_b, ["Under 300", "300–449", "450+"]),
        bucket_table("Your goals", goals_b, ["0", "1", "2+"]),
        bucket_table("Your saves", saves_b, ["0", "1", "2", "3+"]),
        bucket_table("Team shots vs theirs", cmp_b("shots"), ["We had more", "Even", "They had more"]),
        bucket_table("Team saves vs theirs", cmp_b("saves"), ["We had more", "Even", "They had more"]),
        bucket_table("Team ball hits vs theirs", cmp_b("ball_hits"), ["We had more", "Even", "They had more"]),
        bucket_table("Team demos vs theirs", cmp_b("demos"), ["We had more", "Even", "They had more"]),
    ]
    # kickoffs: each team's share of kickoffs won (from replays), ours against theirs
    kick = {r["id"]: ((us(m).get("rotation") or {}).get("pct_kickoffs_won"), (them(m).get("rotation") or {}).get("pct_kickoffs_won"))
            for r, m in zip(rows, games)}

    def kick_b(r):
        a, b = kick.get(r["id"], (None, None))
        if a is None or b is None:
            return None
        return "We won more" if a > b else "Even" if a == b else "They won more"
    factors.append(bucket_table("Team kickoffs won vs theirs", kick_b, ["We won more", "Even", "They won more"]))

    def hour_b(r):
        h = datetime.fromisoformat(r["t"]).hour
        for name, a, b in HOUR_BUCKETS:
            if a <= h < b or a <= h + 24 < b:
                return name
        return None

    time = {
        "hours": bucket_table("Time of day", hour_b, [b[0] for b in HOUR_BUCKETS]),
        "days": bucket_table("Day of week", lambda r: DAYS[datetime.fromisoformat(r["t"]).weekday()], DAYS),
    }

    def ssum(key, side):
        return sum((r[side][key] or 0) for r in rows)
    my_g, my_sh = sum(r["my"]["goals"] or 0 for r in rows), sum(r["my"]["shots"] or 0 for r in rows)
    credited = sum(sum(p["stats"].get("goals") or 0 for p in us(m)["players"]) for m in games)
    opp_credited = sum(sum(p["stats"].get("goals") or 0 for p in them(m)["players"]) for m in games)
    us_sh, them_sh = ssum("shots", "us"), ssum("shots", "them")
    us_sv, them_sv = ssum("saves", "us"), ssum("saves", "them")
    ga, gf = sum(r["against"] or 0 for r in rows), sum(r["for"] or 0 for r in rows)
    my_sv = sum(r["my"]["saves"] or 0 for r in rows)
    efficiency = {
        "my_shooting": _rate(my_g, my_sh), "my_goals": my_g, "my_shots": my_sh,
        "team_shooting": _rate(credited, us_sh), "opp_shooting": _rate(opp_credited, them_sh),
        "team_save_rate": _rate(us_sv, us_sv + ga), "opp_save_rate": _rate(them_sv, them_sv + gf),
        "my_save_share": _rate(my_sv, us_sv), "team_saves": us_sv, "goals_against": ga,
    }

    calendar = defaultdict(lambda: {"games": 0, "wins": 0})
    for r in rows:
        d = r["t"][:10]
        calendar[d]["games"] += 1
        calendar[d]["wins"] += r["result"] == "W"

    # Rivals: opponents you've faced 2+ times, worst record first.
    opp = {}
    for m in games:
        for p in them(m)["players"]:
            e = opp.setdefault(p["key"], {"key": p["key"], "label": label_of(p, None), "games": 0, "wins": 0})
            e["games"] += 1
            e["wins"] += m["result"] == "W"
    rivals = sorted((e for e in opp.values() if e["games"] >= 2),
                    key=lambda e: (-(e["games"] - e["wins"]), e["wins"] / e["games"], -e["games"]))
    for e in rivals:
        e["losses"] = e["games"] - e["wins"]
        e["win_rate"] = _rate(e["wins"], e["games"])

    # Boost & movement: you vs everyone else in your replay games, and in wins vs losses.
    from . import replay_stats as RS
    from . import replay_position as RP
    rg = [m for m in games if RS.has(m["my"])]
    others = [p for m in rg for t in m["teams"] for p in t["players"] if p is not m["my"]]
    mates = [p for m in rg for p in us(m)["players"] if p is not m["my"] and RS.has(p)]
    replay = {"games": len(rg), "meta": RS.meta(), "me": RS.avg_of([m["my"] for m in rg]), "lobby": RS.avg_of(others),
              "team": RS.avg_of(mates) if mates else None,
              "wins": RS.avg_of([m["my"] for m in rg if m["result"] == "W"]),
              "losses": RS.avg_of([m["my"] for m in rg if m["result"] == "L"]),
              "heat": RP.avg_heat([m["my"].get("rheat") for m in rg]),
              "mates_heat": RP.avg_heat([p.get("rheat") for m in rg for p in us(m)["players"] if p is not m["my"]])}
    # opponents' grids are relative to their own goal: turn them 180° so they sit on your field
    opp = RP.avg_heat([p.get("rheat") for m in rg for p in them(m)["players"]])
    replay["opp_heat"] = RP.encode_heat(RP.decode_heat(opp)[::-1, ::-1]) if opp else None

    replay["pads"] = pad_totals(rg)
    return {"games": n, "series": rows, "factors": factors, "time": time, "efficiency": efficiency, "replay": replay,
            "calendar": [{"date": d, **v} for d, v in sorted(calendar.items())], "rivals": rivals[:12],
            "win_conditions": win_conditions(rg), "clutch": clutch(games), "percentiles": percentiles(games),
            "difficulty": difficulty_summary(games), "consistency": consistency(games),
            "tags": tag_summary(games), "style": style_for(who)}


# ---------------------------------------------------------------- lobby difficulty
_DIFF_CACHE: dict = {}


def difficulty_model() -> dict:
    """Baseline = the average player in every stored game; each game's opponents are graded
    against it (60 = an average lobby). Tiers are the thirds of all your games."""
    # this runs once per graded stat (hundreds of times per page), so re-check the cheap DB signature
    # at most every 2 s and never load the matches just to check
    now = time.monotonic()
    if _DIFF_CACHE.get("model") and now - _DIFF_CACHE.get("checked", 0) < 2:
        return _DIFF_CACHE["model"]
    sig = store.matches_signature()
    _DIFF_CACHE["checked"] = now
    if _DIFF_CACHE.get("sig") == sig:
        return _DIFF_CACHE["model"]
    ms = store.all_matches(include_hidden=False)
    items = [(stats.playlist_of(m), p["stats"]) for m in ms if not m.get("duplicate_of") for t in m["teams"] for p in t["players"]]
    base = {k: _avg([x.get(k) for _pl, x in items]) for k in TRACKED_STATS}
    by = defaultdict(list)
    for pl, x in items:
        by[pl].append(x)
    modes, factor = {}, {}
    for pl, xs in by.items():
        mb = {k: _avg([x.get(k) for x in xs]) for k in TRACKED_STATS}
        factor[pl] = {k: (mb[k] / base[k]) if mb[k] is not None and base[k] else None for k in TRACKED_STATS}
        if len(xs) >= 2 * MIN_MODE_GAMES:  # players, so a couple of games of that mode
            modes[pl] = mb
    model = {"base": base, "modes": modes, "factor": factor, "lo": 55, "hi": 65}
    _DIFF_CACHE.update(sig=sig, model=model)  # opp_strength below needs the per-mode baselines
    games, *_ = load(include_unfinished=True)
    vals = sorted(v for v in (opp_strength(m, model) for m in games) if v is not None)
    if len(vals) >= 6:
        model["lo"], model["hi"] = vals[len(vals) // 3], vals[(2 * len(vals)) // 3]
    _DIFF_CACHE.update(sig=sig, model=model)
    return model


def opp_strength(m: dict, model: dict) -> float | None:
    """The opponents' stats graded against the average player in the same mode (60 = average)."""
    pl = m.get("playlist") or stats.playlist_of(m)
    base = model.get("modes", {}).get(pl) or {k: (v * (model.get("factor", {}).get(pl, {}).get(k) or 1)) if v is not None else None
                                              for k, v in model["base"].items()}
    if m.get("my_team") is None or len(m["teams"]) != 2:
        return None
    opp = them(m)["players"]
    if not opp:
        return None
    avg = {k: _avg([p["stats"].get(k) for p in opp]) for k in TRACKED_STATS}
    pts = _stats_pts(avg, base)
    return round(pts, 1) if pts is not None else None


def tier_of(v, model) -> str | None:
    if v is None:
        return None
    return "Tough" if v > model["hi"] else "Easy" if v < model["lo"] else "Average"


def game_difficulty(m: dict) -> dict | None:
    model = difficulty_model()
    v = opp_strength(m, model)
    return None if v is None else {"strength": v, "tier": tier_of(v, model)}


def difficulty_summary(games: list[dict]) -> dict:
    model = difficulty_model()
    rows = []
    for m in games:
        v = opp_strength(m, model)
        if v is not None and m["result"] in ("W", "L"):
            rows.append((m, v, tier_of(v, model)))
    tiers = []
    for t in ("Easy", "Average", "Tough"):
        sel = [m for m, v, tt in rows if tt == t]
        w = sum(m["result"] == "W" for m in sel)
        tiers.append({"tier": t, "games": len(sel), "wins": w, "losses": len(sel) - w, "win_rate": _rate(w, len(sel))})
    tough_wins = sorted(((m, v) for m, v, tt in rows if m["result"] == "W"), key=lambda x: -x[1])[:5]
    return {"tiers": tiers, "avg": _avg([v for _m, v, _t in rows]), "lo": model["lo"], "hi": model["hi"],
            "best_wins": [{"id": m["id"], "t": m["played_at"], "strength": v, "for": us(m).get("score"), "against": them(m).get("score")}
                          for m, v in tough_wins],
            "series": [{"id": m["id"], "t": m["played_at"], "strength": v, "result": m["result"]} for m, v, _t in rows]}


# ---------------------------------------------------------------- consistency
def _pctl(xs: list[float], q: float):
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (i - lo), 1)


def _steady(xs: list[float]) -> dict | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 5:
        return None
    mean = sum(xs) / len(xs)
    sd = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
    cv = sd / mean if mean else None
    within = sum(1 for x in xs if mean and abs(x - mean) <= 0.25 * mean)
    return {"games": len(xs), "mean": round(mean, 1), "sd": round(sd, 1),
            "steadiness": round(max(0.0, min(100.0, 100 * (1 - cv)))) if cv is not None else None,
            "floor": _pctl(xs, 0.1), "ceiling": _pctl(xs, 0.9), "within": round(100 * within / len(xs))}


def consistency(games: list[dict]) -> dict:
    """How steady the viewed player's score is from game to game, vs their regular teammates."""
    me_ = _steady([m["my"]["stats"].get("score") for m in games])
    series = [{"id": m["id"], "t": m["played_at"], "score": m["my"]["stats"].get("score"), "result": m["result"]}
              for m in games if m["my"]["stats"].get("score") is not None]
    mates: dict[str, dict] = {}
    for m in games:
        for p in us(m)["players"]:
            if p is m["my"] or not p.get("key"):
                continue
            e = mates.setdefault(p["key"], {"key": p["key"], "label": p.get("alias") or p["name"], "xs": []})
            e["xs"].append(p["stats"].get("score"))
    others = []
    for e in mates.values():
        st = _steady(e["xs"])
        if st:
            others.append({"key": e["key"], "label": e["label"], **st})
    others.sort(key=lambda e: -(e["steadiness"] or 0))
    return {"me": me_, "series": series, "mates": others}


# ---------------------------------------------------------------- tags
def tag_summary(games: list[dict]) -> dict:
    cnt: dict[str, list] = defaultdict(list)
    untagged = []
    for m in games:
        if m["result"] not in ("W", "L"):
            continue
        ts = m.get("tags") or []
        if not ts:
            untagged.append(m)
        for t in ts:
            cnt[t].append(m)

    def rec(ms):
        w = sum(m["result"] == "W" for m in ms)
        return {"games": len(ms), "wins": w, "losses": len(ms) - w, "win_rate": _rate(w, len(ms)),
                "score": _avg([m["my"]["stats"].get("score") for m in ms])}
    rows = sorted(({"tag": t, **rec(ms)} for t, ms in cnt.items()), key=lambda e: -e["games"])
    return {"rows": rows, "untagged": rec(untagged), "all": rec([m for m in games if m["result"] in ("W", "L")])}


# ---------------------------------------------------------------- play style radar + look-alikes
STYLE_AXES = [("scoring", "Scoring", "main", "goals"), ("shooting", "Shooting", "main", "shots"), ("saving", "Saving", "main", "saves"),
              ("touches", "Ball touches", "main", "ball_hits"), ("boost", "Boost", "r", "bpm"), ("speed", "Speed", "r", "avg_speed"),
              ("aerial", "Aerial", "r", "aerial_touches"), ("goalside", "Goal-side", "r", "pct_behind_ball")]
STYLE_ABOUT = {"scoring": "Goals per game", "shooting": "Shots per game", "saving": "Saves per game",
               "touches": "Ball hits per game", "boost": "Boost collected per minute",
               "speed": "Average car speed (km/h)", "aerial": "Ball touches above ~3 m (in the air) per game",
               "goalside": "% of time between the ball and their own goal"}
_STYLE_CACHE: dict = {}


def style_profiles(min_games: int = 3) -> dict:
    """Every player seen in min_games+ games: their per-game average on each axis as a percentile
    of all single-game values by everyone in your games (0-100)."""
    ms = store.all_matches(include_hidden=False)
    sig = (len(ms), max((m["id"] for m in ms), default=0), sum(int(m.get("rstats_version") or 0) for m in ms))
    if _STYLE_CACHE.get("sig") == sig:
        return _STYLE_CACHE["out"]
    from . import replay_stats as RS
    games, me, cfg, aliases = load(include_unfinished=True)
    disp = stats.my_display(me, aliases)

    def val(p, src, k):
        if src == "main":
            return p["stats"].get(k)
        return (p.get("rstats") or {}).get(k) if RS.has(p) else None
    pool = {ax: [] for ax, *_ in STYLE_AXES}
    per: dict[str, dict] = {}
    for m in games:
        for t in m["teams"]:
            for p in t["players"]:
                if not p.get("key"):
                    continue
                keys = [p["key"]] + ([stats.ACCOUNT + p["platform_id"]] if p["key"] == "me" and p.get("platform_id") else [])
                es = [per.setdefault(k_, {"key": k_, "label": label_of(p, disp), "games": 0, "v": defaultdict(list)}) for k_ in keys]
                for e in es:
                    e["games"] += 1
                for ax, _l, src, k in STYLE_AXES:
                    x = val(p, src, k)
                    if x is not None:
                        pool[ax].append(x)  # each performance counts once in the pool
                        for e in es:
                            e["v"][ax].append(x)
    for ax in pool:
        pool[ax].sort()
    import bisect
    out = {}
    for key, e in per.items():
        if e["games"] < min_games:
            continue
        axes, raw = {}, {}
        for ax, *_ in STYLE_AXES:
            xs = e["v"][ax]
            raw[ax] = round(sum(xs) / len(xs), 2) if xs else None  # the actual per-game average
            if len(xs) < 2 or len(pool[ax]) < 10:
                axes[ax] = None
                continue
            a = sum(xs) / len(xs)
            lo, hi = bisect.bisect_left(pool[ax], a), bisect.bisect_right(pool[ax], a)
            axes[ax] = round(100 * (lo + hi) / 2 / len(pool[ax]))
        out[key] = {"key": key, "label": e["label"], "games": e["games"], "axes": axes, "raw": raw}
    _STYLE_CACHE.update(sig=sig, out=out)
    return out


def style_for(key: str) -> dict | None:
    prof = style_profiles()
    me_ = prof.get(key)
    if not me_:
        return None
    sims = []
    for k, o in prof.items():
        if k == key or (stats.is_me_key(k) and stats.is_me_key(key)):
            continue
        common = [ax for ax, *_ in STYLE_AXES if me_["axes"].get(ax) is not None and o["axes"].get(ax) is not None]
        if len(common) < 3:
            continue
        d = (sum((me_["axes"][ax] - o["axes"][ax]) ** 2 for ax in common) / len(common)) ** 0.5
        sims.append({"key": k, "label": o["label"], "games": o["games"], "similarity": round(max(0, 100 - d * 1.25)),
                     "axes": o["axes"], "common": len(common)})
    sims.sort(key=lambda e: -e["similarity"])
    return {**me_, "labels": [[ax, lab] for ax, lab, *_ in STYLE_AXES], "alike": sims[:3],
            "about": {ax: {"text": STYLE_ABOUT[ax], "stat": k, "replay": src == "r"} for ax, _l, src, k in STYLE_AXES},
            "different": sorted(sims, key=lambda e: e["similarity"])[:2] if len(sims) > 3 else []}


# ---------------------------------------------------------------- win conditions
def win_conditions(rg: list[dict], top: int = 8) -> dict:
    """For every replay stat (the player's own, and their team's), the split that best
    separates wins from losses: "you win 78% when possession is above 52%"."""
    from . import replay_stats as RS
    labels = dict(RS.LABELS)
    labels.update(dict(RS.ROTATION))
    n = len(rg)
    out = {"games": n, "rows": []}
    if n < 6:
        return out
    cand = [("me", k) for k in RS.KEYS] + [("team", k) for k, _l in RS.ROTATION]
    min_side = max(3, round(n * 0.2))
    for src, k in cand:
        vals = []
        for m in rg:
            v = (m["my"].get("rstats") or {}).get(k) if src == "me" else (us(m).get("rotation") or {}).get(k)
            if v is not None:
                vals.append((float(v), m["result"] == "W", m["id"]))
        if len(vals) < max(6, n * 0.6):
            continue
        vals.sort()
        best = None
        for c in range(min_side, len(vals) - min_side + 1):
            if vals[c - 1][0] == vals[c][0]:
                continue  # split only between different values
            lo, hi = vals[:c], vals[c:]
            wl, wh = sum(x[1] for x in lo), sum(x[1] for x in hi)
            rl, rh = 100 * wl / len(lo), 100 * wh / len(hi)
            if best is None or abs(rh - rl) > abs(best["diff"]):
                cut = (vals[c - 1][0] + vals[c][0]) / 2
                best = {"src": src, "key": k, "label": labels.get(k, k), "cut": round(cut, 2), "diff": round(rh - rl, 1),
                        "above": {"games": len(hi), "wins": wh, "win_rate": round(rh, 1)},
                        "below": {"games": len(lo), "wins": wl, "win_rate": round(rl, 1)},
                        "_set": frozenset(x[2] for x in (lo if len(lo) <= len(hi) else hi))}
        if best and abs(best["diff"]) >= 15:
            out["rows"].append(best)
    out["rows"].sort(key=lambda r: -abs(r["diff"]))
    # stats that split out exactly the same games say the same thing: fold them into one row
    merged: list[dict] = []
    for r in out["rows"]:
        same = next((x for x in merged if x["_set"] == r["_set"]), None)
        if same:
            same.setdefault("also", []).append({"src": r["src"], "key": r["key"], "label": r["label"], "cut": r["cut"],
                                                "higher_wins": r["diff"] > 0})
        else:
            merged.append(r)
    for r in merged:
        r.pop("_set", None)
    out["rows"] = merged[:top]
    out["losses"] = sum(m["result"] == "L" for m in rg)
    return out


# ---------------------------------------------------------------- boost pickups across games
def _pad_mirror() -> list[int]:
    """Index of the pad opposite each pad (field turned 180°)."""
    from .replay_stats import PADS
    return [min(range(len(PADS)), key=lambda j: (PADS[j][0] + x) ** 2 + (PADS[j][1] + y) ** 2) for x, y, _b in PADS]


def pad_totals(rg: list[dict]) -> dict | None:
    """Pickups per pad summed over games, turned so the viewed player's goal is always at the
    bottom (blue's end): you, your teammates and the opponents."""
    from .replay_stats import PADS
    mir = _pad_mirror()
    out = {k: [0] * len(PADS) for k in ("me", "mates", "opp")}
    seen = {"me": 0, "mates": 0, "opp": 0}  # player-games behind each total, for per-player averages
    n = 0
    for m in rg:
        if not m["my"].get("rpads"):
            continue
        n += 1
        flip = us(m).get("color") != "blue"
        for ti, t in enumerate(m["teams"]):
            for p in t["players"]:
                if not p.get("rpads"):
                    continue
                k = "me" if p is m["my"] else "mates" if ti == m["my_team"] else "opp"
                seen[k] += 1
                for j, c in enumerate(p["rpads"]):
                    out[k][mir[j] if flip else j] += c or 0
    return dict(out, games=n, seen=seen, layout=PADS) if n else None


# ---------------------------------------------------------------- close games and clutch
def clutch(games: list[dict]) -> dict:
    """Close games, overtime, last-minute goals, comebacks and blown leads, from the
    viewed player's side. Final scores work for every game; timelines need a replay."""
    def rec(ms):
        w = sum(m["result"] == "W" for m in ms)
        return {"games": len(ms), "wins": w, "losses": len(ms) - w, "win_rate": _rate(w, len(ms))}
    finished = [m for m in games if us(m).get("score") is not None and them(m).get("score") is not None]
    close = [m for m in finished if abs(us(m)["score"] - them(m)["score"]) == 1]
    big = [m for m in finished if abs(us(m)["score"] - them(m)["score"]) >= 3]
    tl = [m for m in games if m.get("rgoals") is not None and m.get("replay")]
    ot = [m for m in tl if m.get("overtime")]
    first_for, first_against, comeback, blown, trailed, led = [], [], [], [], [], []
    late_for = late_against = my_clutch = ot_goals_me = 0
    clutch_list = []
    by_min = [{"label": f"{5 - i}:00–{4 - i}:00", "minute": i + 1, "for": 0, "against": 0} for i in range(5)] + \
             [{"label": "Overtime", "minute": "OT", "for": 0, "against": 0}]
    for m in tl:
        ci = 0 if us(m).get("color") == "blue" else 1
        me_name = m["my"].get("replay_name") or m["my"]["name"]
        a = b = 0
        was_behind = was_ahead = False
        for i, g in enumerate(m["rgoals"]):
            mine = g["team"] == ci
            # minute of the game: regulation counts down from 5:00 (a goal at 0:00 is still the 5th minute)
            slot = by_min[5] if g["ot"] else by_min[min(4, max(0, (300 - int(g["clock"] or 0)) // 60))]
            slot["for" if mine else "against"] += 1
            if i == 0:
                (first_for if mine else first_against).append(m)
            before = a - b
            if mine:
                a += 1
            else:
                b += 1
            was_behind |= a < b
            was_ahead |= a > b
            late = g["ot"] or g["clock"] <= 60
            if late and not g["ot"]:
                if mine:
                    late_for += 1
                else:
                    late_against += 1
            if mine and g.get("scorer") == me_name and late and before in (-1, 0):
                my_clutch += 1
                ot_goals_me += g["ot"]
                clutch_list.append({"id": m["id"], "t": m["played_at"], "ot": g["ot"], "clock": g["clock"],
                                    "kind": "winner" if g["ot"] else ("go-ahead" if before == 0 else "equaliser"), "result": m["result"]})
        if was_behind:
            trailed.append(m)
            if m["result"] == "W":
                comeback.append(m)
        if was_ahead:
            led.append(m)
            if m["result"] == "L":
                blown.append(m)
    return {
        "games": len(finished), "timeline_games": len(tl),
        "close": rec(close), "big": rec(big), "overtime": rec(ot),
        "scored_first": rec(first_for), "conceded_first": rec(first_against),
        "trailed": rec(trailed), "comebacks": len(comeback), "led": rec(led), "blown": len(blown),
        "late_for": late_for, "late_against": late_against, "my_clutch_goals": my_clutch, "my_ot_goals": ot_goals_me,
        "clutch_goals": sorted(clutch_list, key=lambda x: x["t"], reverse=True)[:8],
        "by_minute": [dict(s, diff=s["for"] - s["against"]) for s in by_min if s["minute"] != "OT" or s["for"] + s["against"]],
    }


# ---------------------------------------------------------------- percentiles vs the lobby
def percentiles(games: list[dict]) -> dict:
    """Where the viewed player's per-game average sits among every single-game performance
    by the other players in the same games. pct = the real rank (0-100); good = how good that is
    (the same, or turned round for stats where lower is better)."""
    from . import replay_stats as RS
    keys = [("main", k, k) for k in TRACKED_STATS]
    keys += [("replay", k, RS.LABELS.get(k, k)) for k in RS.KEYS if k not in RS.NEUTRAL and k not in ("demos",)]
    out = []
    for src, k, lab in keys:
        mine, other = [], []
        for m in games:
            get = (lambda p: (p.get("stats") or {}).get(k)) if src == "main" else (lambda p: (p.get("rstats") or {}).get(k) if RS.has(p) else None)
            v = get(m["my"])
            if v is None:
                continue
            mine.append(v)
            for t in m["teams"]:
                for p in t["players"]:
                    if p is not m["my"]:
                        x = get(p)
                        if x is not None:
                            other.append(x)
        if len(mine) < 3 or len(other) < 10:
            continue
        avg = sum(mine) / len(mine)
        below = sum(1 for x in other if x < avg) + 0.5 * sum(1 for x in other if x == avg)
        pc = 100 * below / len(other)  # the real rank: 80 = more than 80% of them
        low = k in RS.LOWER_BETTER
        out.append({"src": src, "key": k, "label": lab, "avg": round(avg, 2), "pct": round(pc),
                    "good": round(100 - pc if low else pc), "lower_better": low, "games": len(mine), "others": len(other)})
    return {"rows": out}


# ---------------------------------------------------------------- every stat in one table (Stats page)
STAT_HELP = {
    "score": "In-game points from the scoreboard.", "goals": "Goals scored.", "assists": "Passes the game credited as an assist.",
    "ball_hits": "Times you touched the ball (worked out from the replay).", "saves": "Saves the game credited.",
    "shots": "Shots the game counted.", "demos": "Opponents you demolished.",
    "shooting_pct": "Goals ÷ shots.", "goal_part": "Share of your team's goals you scored or assisted.",
    "bpm": "Boost collected per minute.", "avg_boost": "Average boost in the tank.", "pct_zero": "Share of time on 0 boost.",
    "pct_full": "Share of time on 100 boost.", "big_pads": "100-boost pads taken.", "small_pads": "12-boost pads taken.",
    "stolen": "Pads taken in the opponents' half.", "overfill": "Boost wasted by taking pads you didn't need.",
    "avg_speed": "Average car speed (game units).", "pct_supersonic": "Share of time at supersonic speed.",
    "pct_boost_speed": "Share of time at boost speed, below supersonic.", "pct_slow": "Share of time moving slowly.",
    "pct_ground": "Share of time on the ground.", "pct_low_air": "Share of time a little off the ground.",
    "pct_high_air": "Share of time high in the air.", "jumps": "Jumps.", "double_jumps": "Double jumps.", "dodges": "Flips/dodges.",
    "powerslides": "Separate powerslides (re-presses within 0.2 s count as one).", "powerslide_s": "Seconds spent powersliding.",
    "distance_km": "Distance driven.", "demoed": "Times you were demolished.",
    "aerial_touches": "Touches with the ball above ~3 m.", "pct_possession": "Share of the game your touch was the last one.",
    "max_hit_speed": "The fastest you sent the ball.", "crossbars": "Shots that hit the opponents' crossbar.",
    "avg_shot_speed": "Average shot speed.", "max_shot_speed": "Your hardest shot.", "max_goal_speed": "Your hardest goal.",
    "clears": "Touches in your third that sent the ball upfield.", "centres": "Touches near the side wall in their third that sent the ball to the middle.",
    "kickoffs": "Kickoffs you went for.", "kickoff_wins": "Kickoffs you won (ball on their side 2 s later).",
    "tracked_shots": "Shots placed on the shot map.",
    "pct_def_third": "Share of time in your defensive third.", "pct_neutral_third": "Share of time in the middle third.",
    "pct_off_third": "Share of time in your offensive third.", "pct_behind_ball": "Share of time between the ball and your goal.",
    "pct_last_back": "Share of time you were the furthest back on your team.", "pct_first_forward": "Share of time you were the furthest forward.",
    "avg_dist_ball": "Average distance from the ball.", "avg_dist_mate": "Average distance from your teammates.",
}
STAT_GROUP_NAMES = {"main": "Scoreboard", "boost": "Boost", "movement": "Movement", "demos": "Demos", "ball": "Ball", "positioning": "Positioning"}


def _stat_catalogue():
    """[(key, label, group, source)] for every per-player stat; source 'main' = scoreboard, 'r' = replay."""
    from . import replay_stats as RS
    out = [(k, _SL.get(k, k), "main", "main") for k in TRACKED_STATS]
    out += [("shooting_pct", "Shooting %", "main", "ratio"), ("goal_part", "Goal participation %", "main", "ratio")]
    seen = {k for k, *_ in out}
    for g, items in RS.GROUPS.items():
        if g == "touches":
            continue
        for k, lab in items:
            if k not in seen:
                out.append((k, lab, g, "r"))
                seen.add(k)
    return out


def _stat_value(m: dict, p: dict, team: dict, key: str, src: str):
    from . import replay_stats as RS
    if src == "main":
        return p["stats"].get(key)
    if src == "r":
        return p["rstats"].get(key) if RS.has(p) else None
    if key == "shooting_pct":
        sh = p["stats"].get("shots")
        return round(100 * (p["stats"].get("goals") or 0) / sh, 1) if sh else None
    if key == "goal_part":
        gp = stats.goal_participation([(p, team)])
        return gp["pct"] if gp else None
    return None


def _ratio_total(rows: list[tuple[dict, dict]], key: str):
    """Shooting % / goal participation over many games: totals, not an average of per-game ratios."""
    if key == "shooting_pct":
        g = sum(p["stats"].get("goals") or 0 for p, _t in rows)
        sh = sum(p["stats"].get("shots") or 0 for p, _t in rows)
        return round(100 * g / sh, 1) if sh else None
    gp = stats.goal_participation(rows)
    return gp["pct"] if gp else None


def stat_table(size="all", ranked="all", last=0, who="me", start="", end="", tag="") -> dict:
    """Every stat: you, your average teammate, the lobby, your percentile, recent trend, best game, wins vs losses."""
    from . import replay_stats as RS
    games, *_ = load(size, ranked, who=who, start=start, end=end, tag=tag)
    if last:
        games = games[-last:]
    rows = []
    for k, lab, grp, src in _stat_catalogue():
        low, neutral = k in RS.LOWER_BETTER, k in RS.NEUTRAL
        mine = [(m, _stat_value(m, m["my"], us(m), k, src)) for m in games]
        mine = [(m, v) for m, v in mine if v is not None]
        if not mine:
            continue
        mates = [_stat_value(m, p, us(m), k, src) for m in games for p in us(m)["players"] if p is not m["my"]]
        lobby = [_stat_value(m, p, t, k, src) for m in games for t in m["teams"] for p in t["players"] if p is not m["my"]]
        if src == "ratio":
            you = _ratio_total([(m["my"], us(m)) for m, _v in mine], k)
            mate_v = _ratio_total([(p, us(m)) for m in games for p in us(m)["players"] if p is not m["my"]], k)
            lob_v = _ratio_total([(p, t) for m in games for t in m["teams"] for p in t["players"] if p is not m["my"]], k)
        else:
            you, mate_v, lob_v = _avg([v for _m, v in mine]), _avg(mates), _avg(lobby)
        pool = sorted(v for v in lobby if v is not None)
        pct = None
        if len(pool) >= 10 and you is not None:
            lo = sum(1 for v in pool if v < you)
            eq = sum(1 for v in pool if v == you)
            pct = round(100 * (lo + eq / 2) / len(pool))
        recent = _avg([v for _m, v in mine[-10:]]) if len(mine) >= 15 else None
        best_m, best_v = (min if low else max)(mine, key=lambda x: x[1])
        rows.append({
            "key": k, "label": lab, "group": grp, "group_label": STAT_GROUP_NAMES.get(grp, grp), "help": STAT_HELP.get(k, ""),
            "lower_better": low, "neutral": neutral, "pct_unit": src == "ratio" or k.startswith("pct_"),
            "you": you, "mates": mate_v, "lobby": lob_v,
            "pct": pct, "good": None if pct is None or neutral else (100 - pct if low else pct),
            "recent": recent, "games": len(mine),
            "best": {"v": best_v, "id": best_m["id"], "t": best_m["played_at"], "result": best_m["result"]},
            "wins": _avg([v for m, v in mine if m["result"] == "W"]), "losses": _avg([v for m, v in mine if m["result"] == "L"]),
        })
    return {"games": len(games), "rows": rows}


def stat_detail(key: str, size="all", ranked="all", last=0, who="me", start="", end="", tag="") -> dict:
    """One stat over time: every game's value, by mode, best games."""
    from . import replay_stats as RS
    cat = {k: (lab, grp, src) for k, lab, grp, src in _stat_catalogue()}
    if key not in cat:
        return {"key": key, "series": []}
    lab, grp, src = cat[key]
    games, *_ = load(size, ranked, who=who, start=start, end=end, tag=tag)
    if last:
        games = games[-last:]
    series = []
    for m in games:
        v = _stat_value(m, m["my"], us(m), key, src)
        if v is not None:
            series.append({"id": m["id"], "t": m["played_at"], "v": v, "result": m["result"], "playlist": m["playlist"],
                           "for": us(m).get("score"), "against": them(m).get("score")})
    by_mode = defaultdict(list)
    for x in series:
        by_mode[x["playlist"]].append(x["v"])
    low = key in RS.LOWER_BETTER
    best = sorted(series, key=lambda x: x["v"], reverse=not low)[:5]
    return {"key": key, "label": lab, "group": STAT_GROUP_NAMES.get(grp, grp), "help": STAT_HELP.get(key, ""),
            "lower_better": low, "series": series, "best": best,
            "by_mode": [{"mode": k, "avg": _avg(v), "games": len(v)} for k, v in sorted(by_mode.items())],
            "wins": _avg([x["v"] for x in series if x["result"] == "W"]), "losses": _avg([x["v"] for x in series if x["result"] == "L"])}


# ---------------------------------------------------------------- shot map across a session's games
def shot_map(games: list[dict]) -> dict:
    """Every tracked shot in these games, turned so your goal is at the bottom: shots against
    you attack downwards, your team's shots attack upwards."""
    shots = []
    saves_by: dict[str, int] = defaultdict(int)
    for m in games:
        if not m.get("rshots"):
            continue
        ci = 0 if us(m).get("color") == "blue" else 1
        me_name = m["my"].get("replay_name") or m["my"]["name"]
        mates = {p.get("replay_name") or p["name"]: (p.get("alias") or p["name"]) for p in us(m)["players"]}
        f = -1 if ci == 1 else 1
        for s in m["rshots"]:
            side = "for" if s["team"] == ci else "against"
            x = dict(s)
            x["x"], x["y"] = s["x"] * f, s["y"] * f
            if s.get("path"):
                x["path"] = [[px * f, py * f, pz] for px, py, pz in s["path"]]
            x.update(side=side, mine=side == "for" and s["name"] == me_name, id=m["id"], date=m["played_at"], result=m["result"])
            if side == "against" and s.get("outcome") == "saved" and s.get("by") in mates:
                x["by"] = mates[s["by"]]
                saves_by[x["by"]] += 1
            shots.append(x)

    def summ(xs):
        g = sum(1 for x in xs if x.get("outcome") == "goal")
        return {"shots": len(xs), "goals": g, "saved": sum(1 for x in xs if x.get("outcome") == "saved"), "conv": _rate(g, len(xs))}
    return {"shots": shots, "games": len({x["id"] for x in shots}),
            "against": summ([x for x in shots if x["side"] == "against"]), "for": summ([x for x in shots if x["side"] == "for"]),
            "mine": summ([x for x in shots if x["mine"]]),
            "saves_by": sorted(({"name": k, "saves": v} for k, v in saves_by.items()), key=lambda e: -e["saves"])}


def session_shots(start: str, end: str) -> dict:
    """The shot map for one session's games (asked for when a session is opened)."""
    games, *_ = load(start=start, end=end)
    return shot_map(games)


# ---------------------------------------------------------------- team extras
def team_extras(lineup: str = "", size: str = "all", ranked: str = "all") -> dict:
    games, me, cfg, aliases = load(size, ranked)
    disp = stats.my_display(me, aliases)

    def lineup_key(m):
        return ",".join(sorted(p["key"] for p in us(m)["players"] if p["key"] != "me"))

    def lineup_label(m):
        return ", ".join(sorted(label_of(p, disp) for p in us(m)["players"] if p["key"] != "me")) or "solo"

    # Every lineup you've played, with team-vs-team +/- per stat.
    by = defaultdict(list)
    labels = {}
    for m in games:
        k = lineup_key(m)
        by[k].append(m)
        labels[k] = lineup_label(m)
    compare = []
    for k, ms in by.items():
        w = sum(m["result"] == "W" for m in ms)
        diff = {s: _avg([(total(us(m), s) or 0) - (total(them(m), s) or 0) for m in ms
                         if total(us(m), s) is not None and total(them(m), s) is not None]) for s in TRACKED_STATS}
        compare.append({"key": k, "label": labels[k], "games": len(ms), "wins": w, "losses": len(ms) - w,
                        "win_rate": _rate(w, len(ms)), "diff": diff})
    compare.sort(key=lambda e: -e["games"])

    pool = games if lineup in ("", "any") else [m for m in games if lineup_key(m) == lineup]
    if lineup == "" and compare:
        pool = by[compare[0]["key"]]

    # Chemistry: win rate for every pair of players who were on your team.
    names = {}
    pairs = defaultdict(lambda: [0, 0])
    for m in games:
        ps = us(m)["players"]
        for p in ps:
            names[p["key"]] = label_of(p, disp)
        keys = sorted({p["key"] for p in ps})
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                pairs[(a, b)][0] += 1
                pairs[(a, b)][1] += m["result"] == "W"
    seen = defaultdict(int)
    for (a, b), (n_, _w) in pairs.items():
        seen[a] += n_
        seen[b] += n_
    people = sorted([k for k in names if k == "me" or seen[k] >= 2 or k.startswith("@")],
                    key=lambda k: (k != "me", -seen[k]))[:8]
    chemistry = {"players": [{"key": k, "label": names[k]} for k in people],
                 "pairs": [{"a": a, "b": b, "games": v[0], "wins": v[1], "win_rate": _rate(v[1], v[0])}
                           for (a, b), v in pairs.items() if a in people and b in people]}

    # MVP board for the selected lineup.
    board = {}
    for m in pool:
        ps = us(m)["players"]
        top = max(ps, key=lambda p: p["stats"].get("score") or 0)
        for p in ps:
            e = board.setdefault(p["key"], {"key": p["key"], "label": label_of(p, disp), "games": 0,
                                            "mvps": 0, "top_scorer": 0, "carried_losses": 0, "wins": 0})
            e["games"] += 1
            e["wins"] += m["result"] == "W"
            e["mvps"] += m.get("mvp") == p["name"]
            if p is top:
                e["top_scorer"] += 1
                e["carried_losses"] += m["result"] == "L"
    mvp_board = sorted(board.values(), key=lambda e: (-e["mvps"], -e["top_scorer"], -e["games"]))
    for e in mvp_board:
        e["mvp_rate"] = _rate(e["mvps"], e["wins"])
        e["top_rate"] = _rate(e["top_scorer"], e["games"])
    mvp_board = [e for e in mvp_board if e["games"] >= 2 or e["key"] == "me"]
    return {"lineup_compare": compare, "chemistry": chemistry, "mvp_board": mvp_board,
            "teammate_ratings": teammate_ratings(games, disp)}


# ---------------------------------------------------------------- teammate ratings
def teammate_ratings(games: list[dict], disp: str | None, min_games: int = 2) -> dict:
    """A rating for everyone who has been on your team: their stats against the
    average player in your lobbies (60%) and your win rate with them (40%)."""
    lobby = {k: _avg([p["stats"].get(k) for m in games for t in m["teams"] for p in t["players"]]) for k in TRACKED_STATS}
    lobby_modes = _bases_by_mode([(m["playlist"], p["stats"]) for m in games for t in m["teams"] for p in t["players"]],
                                 min_n=2 * MIN_MODE_GAMES)
    by: dict[str, dict] = {}
    for m in games:
        team = us(m)["players"]
        top = max(team, key=lambda p: p["stats"].get("score") or 0)
        team_score = total(us(m), "score") or 0
        for p in team:
            if p is m["my"] or not p.get("key") or p["key"] == "me":
                continue
            e = by.setdefault(p["key"], {"key": p["key"], "label": label_of(p, disp), "rows": [], "ids": set(), "pls": [],
                                         "wins": 0, "mvps": 0, "top": 0, "share": []})
            e["rows"].append(p["stats"])
            e["pls"].append(m["playlist"])
            e["ids"].add(m["id"])
            e["wins"] += m["result"] == "W"
            e["mvps"] += m.get("mvp") == p["name"]
            e["top"] += p is top
            if team_score and p["stats"].get("score") is not None:
                e["share"].append(100 * p["stats"]["score"] / team_score)
    out = []
    for e in by.values():
        n = len(e["rows"])
        if n < min_games and not e["key"].startswith("@"):
            continue
        avg = {k: _avg([r.get(k) for r in e["rows"]]) for k in TRACKED_STATS}
        exp = _expected(lobby_modes, e["pls"])  # the average player in the same modes
        vs = {}
        for k in TRACKED_STATS:
            r = _rel(avg[k], exp[k], k)
            vs[k] = round(100 * r) if r is not None else None
        without = [m for m in games if m["id"] not in e["ids"]]
        wr = _rate(e["wins"], n)
        wr_wo = _rate(sum(m["result"] == "W" for m in without), len(without)) if len(without) >= min_games else None
        perf = _stats_pts(avg, exp)
        winp = _win_pts(wr)
        rating = None if perf is None else (0.6 * perf + 0.4 * winp if winp is not None else perf)
        ranked = sorted(((k, v) for k, v in vs.items() if v is not None), key=lambda x: -x[1])
        out.append({
            "key": e["key"], "label": e["label"], "games": n, "wins": e["wins"], "losses": n - e["wins"],
            "win_rate": wr, "win_rate_without": wr_wo,
            "impact": round(wr - wr_wo, 1) if wr is not None and wr_wo is not None else None,
            "my_score_with": _avg([m["my"]["stats"].get("score") for m in games if m["id"] in e["ids"]]),
            "my_score_without": _avg([m["my"]["stats"].get("score") for m in without]),
            "avg": avg, "vs_lobby": vs, "mvps": e["mvps"], "top_rate": _rate(e["top"], n),
            "share": round(sum(e["share"]) / len(e["share"])) if e["share"] else None,
            "rating": _letter(rating), "performance": _letter(perf),
            "strength": ranked[0] if ranked and ranked[0][1] >= 10 else None,
            "weakness": ranked[-1] if ranked and ranked[-1][1] <= -10 else None,
            "few_games": n < 5,
        })
    out.sort(key=lambda e: (e["few_games"], -(e["rating"] or {}).get("pts", 0), -e["games"]))
    return {"lobby": lobby, "players": out, "stats": TRACKED_STATS}


# ---------------------------------------------------------------- teammate impact
def teammate_impact(key: str, min_games: int = 2) -> dict | None:
    """How one player's stats and win rate change depending on who's on their
    team: e.g. "averages +0.8 saves when Alex plays". Works for anyone."""
    games, me, cfg, aliases = load(include_unfinished=False)
    disp = stats.my_display(me, aliases)
    mine = []  # (match, the player's team, the player)
    for m in games:
        for t in m["teams"]:
            for p in t["players"]:
                if stats.key_matches(p, key):
                    mine.append((m, t, p))
    if not mine:
        return None
    label = label_of(mine[-1][2], disp)

    def won(m, t):
        return t.get("winner")

    def summary(rows):
        n = len(rows)
        w = sum(1 for m, t, p in rows if won(m, t))
        return {"games": n, "win_rate": _rate(w, n),
                "avg": {k: _avg([p["stats"].get(k) for m, t, p in rows]) for k in TRACKED_STATS}}

    base = summary(mine)
    mates: dict[str, dict] = {}
    for m, t, p in mine:
        for q in t["players"]:
            if q is p or not q.get("key"):
                continue
            mates.setdefault(q["key"], {"key": q["key"], "label": label_of(q, disp)})
    out = []
    for mk, info in mates.items():
        with_ = [(m, t, p) for m, t, p in mine if any(q.get("key") == mk for q in t["players"])]
        without = [(m, t, p) for m, t, p in mine if not any(q.get("key") == mk for q in t["players"])]
        if len(with_) < min_games:
            continue
        a, b = summary(with_), summary(without)
        diff = {k: (round(a["avg"][k] - b["avg"][k], 2) if a["avg"][k] is not None and b["avg"][k] is not None else None)
                for k in TRACKED_STATS}
        out.append({**info, "with": a, "without": b, "diff": diff,
                    "win_diff": (round(a["win_rate"] - b["win_rate"], 1) if b["games"] and a["win_rate"] is not None else None)})
    out.sort(key=lambda e: -e["with"]["games"])

    # plain-language highlights: the biggest per-game swings with enough games both ways
    notes = []
    for e in out:
        if e["without"]["games"] < min_games:
            continue
        for k in TRACKED_STATS:
            d, wo = e["diff"][k], e["without"]["avg"][k]
            if d is None or wo is None:
                continue
            rel = d / max(abs(wo), 0.5 if k != "score" else 50)
            if abs(rel) >= 0.2:
                notes.append({"mate": e["label"], "mate_key": e["key"], "stat": k, "diff": d, "rel": round(100 * rel),
                              "with": e["with"]["avg"][k], "without": wo})
    notes.sort(key=lambda n: -abs(n["rel"]))
    return {"key": key, "label": label, "base": base, "mates": out, "notes": notes[:6], "stats": TRACKED_STATS}


# ---------------------------------------------------------------- player profile
def profile(key: str) -> dict | None:
    games, me, cfg, aliases = load(include_unfinished=True)
    disp = stats.my_display(me, aliases)
    apps = []
    for m in games:
        for ti, t in enumerate(m["teams"]):
            for p in t["players"]:
                if stats.key_matches(p, key):
                    apps.append((m, ti, p))
    if not apps:
        return None
    names = defaultdict(int)
    for _m, _ti, p in apps:
        names[p["name"]] += 1
    p0 = apps[-1][2]
    label = label_of(p0, disp)
    if key.startswith(stats.ACCOUNT):  # one of your accounts
        label = f"{label} · {p0.get('replay_name') or p0['name']}"
    alias = None if stats.is_me_key(key) else p0.get("alias")

    def split(rel):
        sel = [(m, ti, p) for m, ti, p in apps
               if (rel == "with" and ti == m["my_team"]) or (rel == "against" and ti != m["my_team"])]
        fin = [(m, ti, p) for m, ti, p in sel if m["result"] in ("W", "L")]
        w = sum(m["result"] == "W" for m, _, _ in fin)
        return sel, {"games": len(fin), "wins": w, "losses": len(fin) - w, "win_rate": _rate(w, len(fin))}

    with_, rec_with = split("with")
    against, rec_against = split("against")
    allp = [p for _m, _ti, p in apps]
    averages = {k: _avg([p["stats"].get(k) for p in allp]) for k in TRACKED_STATS}
    bests = {}
    for k in TRACKED_STATS:
        ranked_ = sorted((x for x in apps if (x[2]["stats"].get(k) or 0) > 0), key=lambda x: -x[2]["stats"][k])[:3]
        bests[k] = [{"value": p["stats"][k], "id": m["id"], "t": m["played_at"], "result": m["result"] if ti == m["my_team"] else
                     {"W": "L", "L": "W"}.get(m["result"])} for m, ti, p in ranked_]
    share = defaultdict(list)
    for m, ti, p in apps:
        team_ = m["teams"][ti]
        for k in ("goals", "saves", "ball_hits", "shots", "score"):
            tot = sum(q["stats"].get(k) or 0 for q in team_["players"])
            if tot:
                share[k].append((p["stats"].get(k) or 0) / tot)
    recent = []
    for m, ti, p in reversed(apps[-15:]):
        mine = ti == m["my_team"]
        res = m["result"] if mine else {"W": "L", "L": "W"}.get(m["result"])
        recent.append({"id": m["id"], "t": m["played_at"], "rel": "with you" if mine else "against you",
                       "result": res, "playlist": m["playlist"], "stats": {k: p["stats"].get(k) for k in TRACKED_STATS},
                       "for": m["teams"][ti].get("score"), "against": m["teams"][1 - ti].get("score")})
    mvps = sum(m.get("mvp") == p["name"] for m, ti, p in apps)
    from . import replay_stats as RS
    rgames = [p["rstats"] for _m, _ti, p in apps if p.get("rstats") and "bpm" in p["rstats"]]
    from . import replay_position as RP
    rplayers = [p for _m, _ti, p in apps if RS.has(p)]
    rstats = RS.avg_of(rplayers)
    rstats["ball_hits"] = _avg([p["stats"].get("ball_hits") for p in rplayers])
    heat = RP.avg_heat([p.get("rheat") for p in rplayers])
    rotation = RP.avg_rotation([m["teams"][ti].get("rotation") for m, ti, p in apps if RS.has(p)])
    return {
        "key": key, "label": label, "alias": alias, "is_me": stats.is_me_key(key),
        "names": sorted(names, key=lambda n: -names[n]),
        "games": len(apps), "with": rec_with, "against": rec_against, "mvps": mvps,
        "averages": averages, "bests": bests,
        "share": {k: round(100 * sum(v) / len(v), 1) for k, v in share.items() if v},
        "recent": recent,
        "rstats": rstats, "rstats_games": len(rgames), "rstats_labels": RS.meta(), "heat": heat, "rotation": rotation,
        "first": apps[0][0]["played_at"], "last": apps[-1][0]["played_at"],
        "style": style_for(key),
    }


# ---------------------------------------------------------------- player vs player
def compare_players(a: str, b: str, last: int = 0, frm: str = "", to: str = "",
                    start: str = "", end: str = "", c: str = "") -> dict:
    """Two (or three, with c) players side by side over the same timeframe.

    Timeframe: everything, each player's last N games (last), a date range
    (frm/to as YYYY-MM-DD), or a session's exact time window (start/end ISO).
    Each player is seen from their own side: their team, their result."""
    from . import replay_stats as RS
    cfg = store.load_config()
    ms = store.all_matches()
    me = stats.detect_me([m for m in ms if not m["hidden"]], cfg)
    stats.mark_duplicates(ms)
    aliases = cfg.get("aliases") or []
    disp = stats.my_display(me, aliases)
    pool = []
    for m in ms:
        if m["hidden"] or m.get("duplicate_of") or len(m["teams"]) != 2:
            continue
        stats.enrich(m, me)
        stats.apply_aliases(m, aliases, me)
        day = m["played_at"][:10]
        if frm and day < frm or to and day > to:
            continue
        if start and m["played_at"] < start or end and m["played_at"] > end:
            continue
        pool.append(m)
    pool.sort(key=lambda m: m["played_at"])
    viewers = stats.viewers(ms, disp)
    label = {v["key"]: v["label"] for v in viewers}

    def apps(key):
        out = []
        for m in pool:
            for ti, t in enumerate(m["teams"]):
                for p in t["players"]:
                    if stats.key_matches(p, key):
                        res = stats._result_for(m, ti)
                        if res in ("W", "L"):
                            out.append((m, ti, p, res))
        return out[-last:] if last else out

    def side(key):
        A = apps(key)
        n = len(A)
        w = sum(r == "W" for *_x, r in A)
        avg = {k: _avg([p["stats"].get(k) for _m, _ti, p, _r in A]) for k in TRACKED_STATS}
        tot = {k: sum(p["stats"].get(k) or 0 for _m, _ti, p, _r in A) for k in TRACKED_STATS}
        diff = {k: _avg([(total(m["teams"][ti], k) - total(m["teams"][1 - ti], k))
                         if total(m["teams"][ti], k) is not None and total(m["teams"][1 - ti], k) is not None else None
                         for m, ti, _p, _r in A]) for k in TRACKED_STATS}
        share = {}
        for k in ("score", "goals", "assists", "ball_hits", "saves", "shots", "demos"):
            xs = [(p["stats"].get(k) or 0) / t for m, ti, p, _r in A
                  if (t := sum(q["stats"].get(k) or 0 for q in m["teams"][ti]["players"]))]
            share[k] = round(100 * sum(xs) / len(xs), 1) if xs else None
        rp = [p for _m, _ti, p, _r in A if RS.has(p)]
        return {
            "key": key, "label": label.get(key, key), "games": n, "wins": w, "losses": n - w, "win_rate": _rate(w, n),
            "mvps": sum(m.get("mvp") == p["name"] for m, _ti, p, _r in A),
            "shooting": _rate(tot["goals"], tot["shots"]) if tot.get("shots") else None,
            "goal_part": (stats.goal_participation([(p, m["teams"][ti]) for m, ti, p, _r in A]) or {}).get("pct"),
            "avg": avg, "totals": tot, "team_diff": diff, "share": share,
            "best": {k: max((p["stats"].get(k) or 0 for _m, _ti, p, _r in A), default=None) for k in TRACKED_STATS},
            "replay_games": len(rp), "replay": RS.avg_of(rp),
            "first": A[0][0]["played_at"] if A else None, "last": A[-1][0]["played_at"] if A else None,
            "_ids": {m["id"]: (ti, p, r) for m, ti, p, r in A},
        }

    SA, SB = side(a), side(b)
    # head to head: games both played in this timeframe
    both = [mid for mid in SA["_ids"] if mid in SB["_ids"]]
    together = [mid for mid in both if SA["_ids"][mid][0] == SB["_ids"][mid][0]]
    against = [mid for mid in both if SA["_ids"][mid][0] != SB["_ids"][mid][0]]
    tw = sum(SA["_ids"][mid][2] == "W" for mid in together)
    aw = sum(SA["_ids"][mid][2] == "W" for mid in against)

    def h2h_avg(ids, S):
        return {k: _avg([S["_ids"][mid][1]["stats"].get(k) for mid in ids]) for k in TRACKED_STATS}

    h2h = {
        "together": {"games": len(together), "wins": tw, "losses": len(together) - tw, "win_rate": _rate(tw, len(together)),
                     "a": h2h_avg(together, SA), "b": h2h_avg(together, SB)},
        "against": {"games": len(against), "a_wins": aw, "b_wins": len(against) - aw,
                    "a": h2h_avg(against, SA), "b": h2h_avg(against, SB)},
    }
    out = {"a": SA, "b": SB, "h2h": h2h, "stats": TRACKED_STATS, "players": viewers, "rs_meta": RS.meta(),
           "style_a": style_for(a), "style_b": style_for(b)}
    if c and c not in (a, b):  # optional third player
        SC = side(c)
        trio = [mid for mid in together if mid in SC["_ids"] and SC["_ids"][mid][0] == SA["_ids"][mid][0]]
        w3 = sum(SA["_ids"][mid][2] == "W" for mid in trio)
        h2h["trio"] = {"games": len(trio), "wins": w3, "losses": len(trio) - w3, "win_rate": _rate(w3, len(trio)),
                       "a": h2h_avg(trio, SA), "b": h2h_avg(trio, SB), "c": h2h_avg(trio, SC)}
        SC.pop("_ids")
        out.update(c=SC, style_c=style_for(c))
    for S in (SA, SB):
        S.pop("_ids")
    return out
