"""Where goals come from and who's involved: goal causes and kinds, expected goals (xG), team-play
habits per player, win rate by role, and focus goals. Built from what's stored per match."""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

import numpy as np

from . import insights as I
from . import replay_plays as PL
from . import stats, store

CAUSE_LABEL = dict(PL.CAUSES)
KIND_LABEL = dict(PL.KINDS)


def _ci(m):
    return 0 if I.us(m).get("color") == "blue" else 1


def _members(games: list[dict], disp: str | None) -> list[tuple[str, str]]:
    """You plus teammates in at least a tenth of these games (3+): [(key, label)]."""
    cnt, lab = Counter(), {}
    for m in games:
        for p in I.us(m)["players"]:
            if p is m["my"]:
                continue
            k = p.get("key")
            if k:
                cnt[k] += 1
                lab[k] = p.get("alias") or p["name"]
    out = [("me", disp or "You")]
    out += [(k, lab[k]) for k, c in cnt.most_common() if c >= max(3, 0.1 * len(games))]
    return out[:5]


def _player(m: dict, key: str):
    if key == "me":
        return m["my"]
    return next((p for p in I.us(m)["players"] if p.get("key") == key), None)


def _rname(p: dict) -> str:
    return p.get("replay_name") or p["name"]


# ---------------------------------------------------------------- expected goals
_XG: dict = {}
POST = 893.0
GOAL_Y = 5120.0


def _feat(s: dict) -> list[float] | None:
    d = 1.0 if s.get("team") == 0 else -1.0
    x, y, z = s.get("x"), s.get("y"), s.get("z") or 0
    if x is None or y is None or s.get("speed") is None:
        return None
    x, dy = x * d, GOAL_Y - y * d
    dy = max(dy, 50.0)
    dist = math.hypot(x, dy)
    angle = abs(math.atan2(POST - x, dy) - math.atan2(-POST - x, dy))  # how much of the goal mouth is visible
    n, b = s.get("def_n") or 0, s.get("def_back")
    nobody = 1.0 if b == 0 else 0.0                       # nobody goal-side
    share = (b / n) if (b is not None and n) else 0.6     # share of the defenders goal-side (typical value if unknown)
    return [1.0, math.log(dist / 1000), angle, s["speed"] / 100, min(z, 1500) / 1000, nobody, share]


def _scored(s: dict) -> int:
    return 1 if s.get("goal") or s.get("outcome") == "goal" else 0


def model() -> dict:
    """Logistic regression fitted on every stored shot (all players, all games). Cached per data change."""
    if _XG and time.time() - _XG.get("checked", 0) < 2:  # checked a moment ago (one page load scores hundreds of shots)
        return _XG
    sig = store.matches_signature()
    if _XG.get("sig") == sig:
        _XG["checked"] = time.time()
        return _XG
    X, y = [], []
    for m in store.all_matches():
        for s in m.get("rshots") or []:
            f = _feat(s)
            if f:
                X.append(f)
                y.append(_scored(s))
    w = np.array([-1.0, -1.0, 2.0, 0.6, -0.3, 1.0, -0.8])  # sensible start if there are few shots
    if len(X) >= 80 and 0 < sum(y) < len(y):
        Xa, ya = np.array(X), np.array(y, dtype=float)
        lam = np.diag([0.0] + [1.0] * (Xa.shape[1] - 1))
        for _ in range(50):  # Newton steps with a ridge penalty
            p = 1 / (1 + np.exp(-Xa @ w))
            g = Xa.T @ (p - ya) + lam @ w
            H = Xa.T @ (Xa * (p * (1 - p))[:, None]) + lam
            step = np.linalg.solve(H, g)
            w = w - step
            if np.abs(step).max() < 1e-6:
                break
    _XG.clear()
    _XG.update(sig=sig, w=w, shots=len(X), goals=int(sum(y)), at=time.time(), checked=time.time())
    return _XG


def xg_of(s: dict) -> float | None:
    f = _feat(s)
    if not f:
        return None
    return float(1 / (1 + math.exp(-float(np.dot(model()["w"], f)))))


def xg_summary(games: list[dict], disp: str | None) -> dict:
    mem = _members(games, disp)
    tm = {"for": [0.0, 0, 0], "against": [0.0, 0, 0]}  # xg, goals, shots
    per = {k: [0.0, 0, 0, 0] for k, _l in mem}           # xg, goals, shots, games
    big_missed = []
    for m in games:
        ci = _ci(m)
        names = {_rname(p): k for k, _l in mem if (p := _player(m, k))}
        for k, _l in mem:
            if _player(m, k):
                per[k][3] += 1
        for s in m.get("rshots") or []:
            x = xg_of(s)
            if x is None:
                continue
            side = "for" if s.get("team") == ci else "against"
            tm[side][0] += x
            tm[side][1] += _scored(s)
            tm[side][2] += 1
            k = names.get(s.get("name")) if side == "for" else None
            if k:
                per[k][0] += x
                per[k][1] += _scored(s)
                per[k][2] += 1
                if x >= 0.45 and not _scored(s):
                    big_missed.append({"id": m["id"], "t": m["played_at"], "who": k, "xg": round(x, 2), "speed": s.get("speed")})
    n = len(games) or 1
    side = lambda v: {"xg": round(v[0], 1), "goals": v[1], "shots": v[2], "xg_pg": round(v[0] / n, 2), "goals_pg": round(v[1] / n, 2),
                      "per_shot": round(v[0] / v[2], 3) if v[2] else None, "diff": round(v[1] - v[0], 1)}
    lab = dict(mem)
    M = model()
    return {"games": len(games), "model_shots": M["shots"], "model_goals": M["goals"],
            "for": side(tm["for"]), "against": side(tm["against"]),
            "players": [{"key": k, "label": lab[k], "xg": round(v[0], 1), "goals": v[1], "shots": v[2], "games": v[3],
                         "xg_pg": round(v[0] / v[3], 2) if v[3] else None, "per_shot": round(v[0] / v[2], 3) if v[2] else None,
                         "diff": round(v[1] - v[0], 1)} for k, v in per.items() if v[3]],
            "big_chances_missed": sorted(big_missed, key=lambda x: -x["xg"])[:8]}


# ---------------------------------------------------------------- goal causes and kinds
def goals_breakdown(games: list[dict], disp: str | None) -> dict:
    mem = _members(games, disp)
    lab = dict(mem)
    against, against_any, kinds_for, kinds_against = Counter(), Counter(), Counter(), Counter()
    inv = defaultdict(Counter)  # member → cause → goals involved in
    recent = []
    low_boost_def = []
    n_against = n_for = 0
    for m in games:
        ci = _ci(m)
        names = {_rname(p): k for k, _l in mem if (p := _player(m, k))}
        for g in m.get("rplays") or []:
            if g["team"] == ci:
                n_for += 1
                kinds_for[g["kind"]] += 1
                continue
            n_against += 1
            kinds_against[g["kind"]] += 1
            against[g["cause"]] += 1
            for c in g["causes"]:
                against_any[c] += 1
            who = [names[x] for x in g.get("involved") or [] if x in names]
            for k in who:
                inv[k][g["cause"]] += 1
            for nm, b in (g.get("def_boost") or {}).items():
                if nm in names:
                    low_boost_def.append(b)
            recent.append({"id": m["id"], "t": m["played_at"], "clock": g.get("clock"), "cause": g["cause"],
                           "label": CAUSE_LABEL.get(g["cause"], g["cause"]), "also": [CAUSE_LABEL[c] for c in g["causes"][1:] if c in CAUSE_LABEL],
                           "who": [lab[k] for k in who], "result": m["result"]})
    pct = lambda c, n: round(100 * c / n, 1) if n else None
    return {
        "games": sum(1 for m in games if m.get("rplays") is not None and m.get("replay")), "against": n_against, "for": n_for,
        "causes": [{"key": k, "label": l, "goals": against[k], "pct": pct(against[k], n_against), "any": against_any[k]}
                   for k, l in PL.CAUSES if against_any[k]],
        "kinds_for": [{"key": k, "label": l, "goals": kinds_for[k], "pct": pct(kinds_for[k], n_for)} for k, l in PL.KINDS],
        "kinds_against": [{"key": k, "label": l, "goals": kinds_against[k], "pct": pct(kinds_against[k], n_against)} for k, l in PL.KINDS],
        "involved": [{"key": k, "label": lab[k], "goals": sum(inv[k].values()),
                      "by": [{"label": CAUSE_LABEL[c], "goals": n} for c, n in inv[k].most_common()]} for k, _l in mem],
        "recent": sorted(recent, key=lambda x: x["t"], reverse=True)[:12],
    }


# ---------------------------------------------------------------- habits, roles
HABIT_KEYS = ["double_commits", "rotation_cuts", "pct_held_back", "giveaways", "recovery_s", "touch_share", "pct_last_back", "boost_def",
              "pct_first_man", "pct_second_man", "pct_third_man"]


def team_play(games: list[dict], disp: str | None) -> dict:
    from . import replay_stats as RS
    mem = _members(games, disp)
    rows, gives = [], {}
    lobby = defaultdict(list)
    for m in games:
        for t in m["teams"]:
            for p in t["players"]:
                if RS.has(p):
                    for k in HABIT_KEYS:
                        if p["rstats"].get(k) is not None:
                            lobby[k].append(p["rstats"][k])
    for key, label in mem:
        ps = [(m, p) for m in games if (p := _player(m, key)) and RS.has(p)]
        if not ps:
            continue
        v = {k: I._avg([p["rstats"].get(k) for _m, p in ps]) for k in HABIT_KEYS}
        # win rate by the role this player mostly played in each game
        role_rec = defaultdict(lambda: [0, 0])
        for m, p in ps:
            r = p["rstats"]
            opts = [("First man", r.get("pct_first_man")), ("Second man", r.get("pct_second_man")), ("Third man", r.get("pct_third_man"))]
            opts = [(n, x) for n, x in opts if x is not None]
            if not opts:
                continue
            main = max(opts, key=lambda o: o[1])[0]
            role_rec[main][0] += 1
            role_rec[main][1] += m["result"] == "W"
        rows.append({"key": key, "label": label, "games": len(ps), **v,
                     "roles": [{"label": n, "games": role_rec[n][0], "wins": role_rec[n][1], "losses": role_rec[n][0] - role_rec[n][1],
                                "win_rate": I._rate(role_rec[n][1], role_rec[n][0])} for n in ("First man", "Second man", "Third man") if role_rec[n][0]]})
        gives[key] = [xy for _m, p in ps for xy in (p.get("rgive") or [])][-400:]
    return {"rows": rows, "lobby": {k: I._avg(v) for k, v in lobby.items()}, "gives": gives,
            "lower_better": sorted(PL.LOWER_BETTER), "labels": dict(PL.GROUP) | {"pct_last_back": "% last back"}}


def diagnose(size="all", ranked="all", last=0, who="me", start="", end="", tag="") -> dict:
    games, me, cfg, aliases = I.load(size, ranked, who=who, start=start, end=end, tag=tag)
    if last:
        games = games[-last:]
    disp = stats.my_display(me, aliases) if who == "me" else None
    rg = [m for m in games if m.get("replay")]
    return {"games": len(games), "goals": goals_breakdown(rg, disp), "xg": xg_summary(rg, disp), "team": team_play(rg, disp),
            "rotation": rotation_more(rg, disp), "mechanics": mechanics(rg, disp), "kickoffs": kickoff_spawns(rg, disp),
            "defending": defending(rg, disp), "opponents": opponents(rg), "changes": changes(games),
            "challenges": challenges(rg, disp), "demos": demos_detail(rg, disp), "passing": passing(rg, disp)}


# ---------------------------------------------------------------- focus goals
def focus_progress() -> dict:
    """Your chosen focus stats, session by session (last 10 sessions), against before you set them."""
    from .patterns import _sessions
    cfg = store.load_config()
    focus = [f for f in cfg.get("focus") or [] if f.get("key")]
    games, *_ = I.load()
    cat = {k: (lab, grp, src) for k, lab, grp, src in I._stat_catalogue()}
    from . import replay_stats as RS
    out = []
    sess = _sessions(games)[-10:]
    for f in focus:
        k = f["key"]
        if k not in cat:
            continue
        lab, _grp, src = cat[k]

        def val(ms):
            if src == "ratio":
                return I._ratio_total([(m["my"], I.us(m)) for m in ms], k)
            return I._avg([I._stat_value(m, m["my"], I.us(m), k, src) for m in ms])
        since = f.get("since") or ""
        before = [m for m in games if m["played_at"] < since]
        after = [m for m in games if m["played_at"] >= since]
        out.append({"key": k, "label": lab, "since": since, "lower_better": k in RS.LOWER_BETTER, "pct_unit": src == "ratio" or k.startswith("pct_"),
                    "before": val(before) if before else None, "after": val(after) if after else None, "after_games": len(after),
                    "sessions": [{"start": g[0]["played_at"], "games": len(g), "v": val(g)} for g in sess]})
    return {"focus": out, "options": [{"key": k, "label": lab, "group": I.STAT_GROUP_NAMES.get(grp, grp)} for k, (lab, grp, _s) in cat.items()]}


# ---------------------------------------------------------------- who cuts whom, behaviour by score, spacing
def rotation_more(games: list[dict], disp: str | None) -> dict:
    mem = _members(games, disp)
    lab = dict(mem)
    keys = [k for k, _l in mem]
    pair = defaultdict(lambda: {"double": 0, "cut": 0, "held": 0})  # (second, first) → counts
    state = {k: {s: {"s": 0.0, "lb": 0.0, "give": 0, "double": 0, "cut": 0, "held": 0} for s in ("ahead", "tied", "behind")} for k in keys}
    bunch, side, bunch_them, side_them, rows = [], [], [], [], []
    for m in games:
        ci = _ci(m)
        names = {_rname(p): k for k in keys if (p := _player(m, k))}
        for c in m.get("rconf") or []:
            if c["team"] != ci:
                continue
            a, b = names.get(c["second"]), names.get(c["first"])
            if a and b:
                pair[(a, b)][c["kind"]] += 1
        for k in keys:
            p = _player(m, k)
            for s, d in ((p or {}).get("rstate") or {}).items():
                for f, v in d.items():
                    state[k][s][f] += v
        ru, rt = I.us(m).get("rotation") or {}, I.them(m).get("rotation") or {}
        if ru.get("pct_bunched") is not None:
            bunch.append(ru["pct_bunched"])
            rows.append((ru["pct_bunched"], m["result"] == "W"))
        if ru.get("pct_same_side") is not None:
            side.append(ru["pct_same_side"])
        if rt.get("pct_bunched") is not None:
            bunch_them.append(rt["pct_bunched"])
        if rt.get("pct_same_side") is not None:
            side_them.append(rt["pct_same_side"])
    matrix = [{"second": a, "second_label": lab[a], "first": b, "first_label": lab[b], **v,
               "went_in": v["double"] + v["cut"], "held_pct": round(100 * v["held"] / (v["held"] + v["double"] + v["cut"]), 1)
               if v["held"] + v["double"] + v["cut"] else None} for (a, b), v in pair.items()]
    by = []
    for k in keys:
        for s in ("ahead", "tied", "behind"):
            d = state[k][s]
            if d["s"] < 60:
                continue
            per5 = 300 / d["s"]
            n = d["held"] + d["double"] + d["cut"]
            by.append({"key": k, "label": lab[k], "state": s, "minutes": round(d["s"] / 60),
                       "pct_last_back": round(100 * d["lb"] / d["s"], 1),
                       "giveaways": round(d["give"] * per5, 2), "went_in": round((d["double"] + d["cut"]) * per5, 2),
                       "held_pct": round(100 * d["held"] / n, 1) if n >= 3 else None})
    split = None
    if len(rows) >= 10:
        med = float(np.median([b for b, _w in rows]))
        hi, lo = [w for b, w in rows if b > med], [w for b, w in rows if b <= med]
        split = {"median": round(med, 1), "tight_wr": I._rate(sum(hi), len(hi)), "tight_n": len(hi),
                 "spread_wr": I._rate(sum(lo), len(lo)), "spread_n": len(lo)}
    return {"matrix": sorted(matrix, key=lambda x: -x["went_in"]), "by_state": by,
            "spacing": {"bunched": I._avg(bunch), "same_side": I._avg(side), "bunched_them": I._avg(bunch_them),
                        "same_side_them": I._avg(side_them), "split": split}}


# ---------------------------------------------------------------- whiffs, shot selection
LOW_XG = 0.10


def mechanics(games: list[dict], disp: str | None) -> dict:
    from . import replay_stats as RS
    mem = _members(games, disp)
    rows = []
    lob_low = [0, 0]
    for m in games:
        for s in m.get("rshots") or []:
            x = xg_of(s)
            if x is not None:
                lob_low[0] += x < LOW_XG
                lob_low[1] += 1
    for key, label in mem:
        ps = [(m, p) for m in games if (p := _player(m, key)) and RS.has(p)]
        if not ps:
            continue
        low = [0, 0, 0]  # low-chance shots, their goals, all shots
        for m, p in ps:
            nm = _rname(p)
            for s in m.get("rshots") or []:
                if s.get("name") != nm or (x := xg_of(s)) is None:
                    continue
                low[2] += 1
                if x < LOW_XG:
                    low[0] += 1
                    low[1] += _scored(s)
        rows.append({"key": key, "label": label, "games": len(ps),
                     "whiffs": I._avg([p["rstats"].get("whiffs") for _m, p in ps]),
                     "air_whiffs": I._avg([p["rstats"].get("air_whiffs") for _m, p in ps]),
                     "pct_whiff": I._avg([p["rstats"].get("pct_whiff") for _m, p in ps]),
                     "low_shots": low[0], "low_goals": low[1], "shots": low[2],
                     "low_share": round(100 * low[0] / low[2], 1) if low[2] else None})
    lobby = {k: I._avg([p["rstats"].get(k) for m in games for t in m["teams"] for p in t["players"] if RS.has(p)])
             for k in ("whiffs", "air_whiffs", "pct_whiff")}
    lobby["low_share"] = round(100 * lob_low[0] / lob_low[1], 1) if lob_low[1] else None
    return {"rows": rows, "lobby": lobby, "low_xg": LOW_XG}


# ---------------------------------------------------------------- kickoffs by spawn
def kickoff_spawns(games: list[dict], disp: str | None) -> dict:
    mem = _members(games, disp)
    lab = dict(mem)
    agg = defaultdict(lambda: {"taken": 0, "won": 0, "lost": 0, "goal_for": 0, "goal_against": 0})
    team = defaultdict(lambda: {"taken": 0, "won": 0, "lost": 0, "goal_for": 0, "goal_against": 0})
    for m in games:
        ci = _ci(m)
        names = {_rname(p): k for k, _l in mem if (p := _player(m, k))}
        for k in m.get("rkick") or []:
            if k["team"] != ci:
                continue
            sp = k["spawn"]
            keys = [sp] + ([sp.rsplit(" ", 1)[0]] if sp.endswith((" left", " right")) else [])  # the side and the whole spawn type
            who = names.get(k["who"])
            for d in [team[x] for x in keys] + ([agg[(who, x)] for x in keys] if who else []):
                d["taken"] += 1
                d["won"] += k["won"] is True
                d["lost"] += k["won"] is False
                d["goal_for"] += k["goal"] == "for"
                d["goal_against"] += k["goal"] == "against"
    fin = lambda d: {**d, "win_pct": round(100 * d["won"] / (d["won"] + d["lost"]), 1) if d["won"] + d["lost"] else None}
    spawns = ["Diagonal", "Diagonal left", "Diagonal right", "Off-centre", "Off-centre left", "Off-centre right", "Back centre"]
    return {"spawns": spawns, "team": {s: fin(team[s]) for s in spawns if team[s]["taken"]},
            "players": [{"key": k, "label": lab[k], "by": {s: fin(agg[(k, s)]) for s in spawns if agg[(k, s)]["taken"]}} for k, _l in mem]}


# ---------------------------------------------------------------- saves vs xG, defensive shape at shots against
def defending(games: list[dict], disp: str | None) -> dict:
    mem = _members(games, disp)
    lab = dict(mem)
    per = {k: {"saves": 0, "xg_saved": 0.0, "big_saves": 0, "last_faced": 0, "last_conceded": 0, "last_xg": 0.0} for k, _l in mem}
    shape = defaultdict(lambda: [0, 0, 0.0])  # defenders goal-side → shots, goals, xg
    dist_goal, dist_saved = [], []
    for m in games:
        ci = _ci(m)
        names = {_rname(p): k for k, _l in mem if (p := _player(m, k))}
        for s in m.get("rshots") or []:
            if s.get("team") == ci:
                continue
            x = xg_of(s)
            if x is None:
                continue
            g = _scored(s)
            if s.get("def_back") is not None:
                b = shape[min(s["def_back"], 3)]
                b[0] += 1
                b[1] += g
                b[2] += x
            if s.get("def_goal_dist") is not None:
                (dist_goal if g else dist_saved).append(s["def_goal_dist"])
            k = names.get(s.get("by")) if s.get("outcome") == "saved" else None
            if k:
                per[k]["saves"] += 1
                per[k]["xg_saved"] += x
                per[k]["big_saves"] += x >= 0.35
            kl = names.get(s.get("def_last"))
            if kl:
                per[kl]["last_faced"] += 1
                per[kl]["last_conceded"] += g
                per[kl]["last_xg"] += x
    rows = []
    for k, v in per.items():
        n = sum(1 for m in games if _player(m, k))
        if not n:
            continue
        rows.append({"key": k, "label": lab[k], "games": n, "saves": v["saves"], "xg_saved": round(v["xg_saved"], 1),
                     "xg_saved_pg": round(v["xg_saved"] / n, 2), "big_saves": v["big_saves"],
                     "last_faced": v["last_faced"], "last_conceded": v["last_conceded"],
                     "last_vs_xg": round(v["last_conceded"] - v["last_xg"], 1)})
    lbl = {0: "Nobody goal-side", 1: "One goal-side", 2: "Two goal-side", 3: "Three goal-side"}
    return {"rows": rows,
            "shape": [{"label": lbl[k], "shots": v[0], "goals": v[1], "goal_pct": round(100 * v[1] / v[0], 1) if v[0] else None,
                       "xg_pct": round(100 * v[2] / v[0], 1) if v[0] else None} for k, v in sorted(shape.items())],
            "last_dist": {"goals": I._avg(dist_goal), "stopped": I._avg(dist_saved)}}


# ---------------------------------------------------------------- what beats you: opponent styles
OPP_STYLES = [("possession", "High-possession teams", lambda m: (I.them(m).get("rotation") or {}).get("pct_possession")),
              ("demos", "Demo-heavy teams", lambda m: I.total(I.them(m), "demos")),
              ("speed", "Fast teams", lambda m: I._avg([(p.get("rstats") or {}).get("avg_speed") for p in I.them(m)["players"]])),
              ("aerial", "Aerial-heavy teams", lambda m: sum((p.get("rstats") or {}).get("aerial_touches") or 0 for p in I.them(m)["players"])),
              ("passes", "Passing teams", lambda m: sum((p.get("rstats") or {}).get("passes") or 0 for p in I.them(m)["players"]))]


def opponents(games: list[dict]) -> dict:
    out = []
    for key, label, fn in OPP_STYLES:
        vals = [(v, m["result"] == "W") for m in games if (v := fn(m)) is not None]
        if len(vals) < 12:
            continue
        vs = sorted(v for v, _w in vals)
        lo_cut, hi_cut = vs[len(vs) // 3], vs[(2 * len(vs)) // 3]
        hi = [w for v, w in vals if v > hi_cut]
        lo = [w for v, w in vals if v < lo_cut]
        if len(hi) < 4 or len(lo) < 4:
            continue
        out.append({"key": key, "label": label, "high_wr": I._rate(sum(hi), len(hi)), "high_n": len(hi),
                    "low_wr": I._rate(sum(lo), len(lo)), "low_n": len(lo),
                    "gap": round(I._rate(sum(hi), len(hi)) - I._rate(sum(lo), len(lo)), 1)})
    return {"games": len(games), "rows": sorted(out, key=lambda r: r["gap"])}


# ---------------------------------------------------------------- what's changed lately
def changes(games: list[dict]) -> dict:
    """Every stat: your recent games against the ones before. Flags changes that are big compared
    with how much the stat varies from game to game."""
    from . import replay_stats as RS
    n = len(games)
    k_recent = max(10, n // 4)
    if n - k_recent < 8:
        return {"games": n, "rows": [], "recent": k_recent}
    new = games[-k_recent:]
    rows = []
    for k, lab, grp, src in I._stat_catalogue():
        if k in RS.NEUTRAL or src == "ratio":
            continue
        vals = [(m, v) for m in games if (v := I._stat_value(m, m["my"], I.us(m), k, src)) is not None]
        by_mode = defaultdict(list)
        for m, v in vals:
            by_mode[m["playlist"]].append(v)
        mode_avg = {pl: float(np.mean(v)) for pl, v in by_mode.items()}
        # each game against the average for its mode, so more 2v2 or 3v3 lately doesn't look like a change
        norm = {m["id"]: v / mode_avg[m["playlist"]] for m, v in vals if mode_avg[m["playlist"]]}
        recent_ids = {m["id"] for m in new}
        a = [x for i, x in norm.items() if i not in recent_ids]
        b = [x for i, x in norm.items() if i in recent_ids]
        if len(a) < 8 or len(b) < 8:
            continue
        se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
        if se <= 0:
            continue
        z = (float(np.mean(b)) - float(np.mean(a))) / se
        ma = float(np.mean([v for m, v in vals if m["id"] not in recent_ids]))
        mb = float(np.mean([v for m, v in vals if m["id"] in recent_ids]))
        good = -z if k in RS.LOWER_BETTER else z
        rows.append({"key": k, "label": lab, "group": I.STAT_GROUP_NAMES.get(grp, grp), "before": round(ma, 2), "recent": round(mb, 2),
                     "change_pct": round(100 * (mb - ma) / abs(ma), 1) if ma else None, "z": round(good, 2),
                     "pct_unit": k.startswith("pct_"), "lower_better": k in RS.LOWER_BETTER})
    rows.sort(key=lambda r: -r["z"])
    return {"games": n, "recent": k_recent, "before": n - k_recent,
            "better": [r for r in rows if r["z"] >= 2][:6], "worse": [r for r in rows if r["z"] <= -2][::-1][:6]}


# ---------------------------------------------------------------- risky challenges and the first defender
def challenges(games: list[dict], disp: str | None) -> dict:
    from . import replay_stats as RS
    mem = _members(games, disp)
    rows = []
    lob = defaultdict(list)
    for m in games:
        for t in m["teams"]:
            for p in t["players"]:
                if RS.has(p):
                    for k in ("lastman_challenges", "lowboost_challenges"):
                        lob[k].append(p["rstats"].get(k) or 0)
    for key, label in mem:
        ps = [(m, p) for m in games if (p := _player(m, key)) and RS.has(p)]
        if not ps:
            continue
        c = Counter()
        for _m, p in ps:
            c.update(p.get("rchal") or {})
        fd_n = sum(p["rstats"].get("first_def") or 0 for _m, p in ps)
        fd_w = sum(round((p["rstats"].get("first_def") or 0) * (p["rstats"].get("pct_first_def_won") or 0) / 100) for _m, p in ps
                   if p["rstats"].get("pct_first_def_won") is not None)
        fd_known = sum(p["rstats"].get("first_def") or 0 for _m, p in ps if p["rstats"].get("pct_first_def_won") is not None)
        n = len(ps)
        pc = lambda a, b: round(100 * a / b, 1) if b else None
        rows.append({"key": key, "label": label, "games": n,
                     "lm_pg": round(c["lm"] / n, 2), "lm": c["lm"], "lm_lost_pct": pc(c["lm_lost"], c["lm"]), "lm_shot": c["lm_shot"], "lm_goal": c["lm_goal"],
                     "lb_pg": round(c["lb"] / n, 2), "lb": c["lb"], "lb_lost_pct": pc(c["lb_lost"], c["lb"]), "lb_shot": c["lb_shot"], "lb_goal": c["lb_goal"],
                     "fd_pg": round(fd_n / n, 2), "fd_won_pct": pc(fd_w, fd_known)})
    tm = lambda side, k: I._avg([(side(m).get("rotation") or {}).get(k) for m in games])
    return {"rows": rows, "lobby": {k: I._avg(v) for k, v in lob.items()},
            "team": {"unchallenged": tm(I.us, "pct_unchallenged"), "unchallenged_them": tm(I.them, "pct_unchallenged"),
                     "fd_won": tm(I.us, "pct_first_def_won"), "fd_won_them": tm(I.them, "pct_first_def_won")}}


# ---------------------------------------------------------------- demos in detail
def demos_detail(games: list[dict], disp: str | None) -> dict:
    mem = _members(games, disp)
    lab = dict(mem)
    per = {k: {"demoed": 0, "demoed_def": 0, "demoed_goal": 0, "demos": 0, "demos_goal": 0, "games": 0} for k, _l in mem}
    by_opp = Counter()
    opp_name = {}
    team = {"for": 0, "for_goal": 0, "against": 0, "against_goal": 0}
    spots = {k: [] for k, _l in mem}
    for m in games:
        ci = _ci(m)
        names = {_rname(p): k for k, _l in mem if (p := _player(m, k))}
        for k in set(names.values()):
            per[k]["games"] += 1
        opp = {_rname(p): p for p in I.them(m)["players"]}
        for d in m.get("rdemos") or []:
            if d["att_team"] == ci:
                team["for"] += 1
                team["for_goal"] += d["goal"]
                k = names.get(d["att"])
                if k:
                    per[k]["demos"] += 1
                    per[k]["demos_goal"] += d["goal"]
            else:
                team["against"] += 1
                team["against_goal"] += d["goal"]
                k = names.get(d["vic"])
                if k:
                    per[k]["demoed"] += 1
                    per[k]["demoed_def"] += d["third"] == "def"
                    per[k]["demoed_goal"] += d["goal"]
                    spots[k].append([d["x"], d["y"]])
                op = opp.get(d["att"])
                if op is not None:
                    ok_ = op.get("key") or d["att"]
                    by_opp[ok_] += 1
                    opp_name[ok_] = op.get("alias") or op["name"]
    rows = [{"key": k, "label": lab[k], **v, "demoed_pg": round(v["demoed"] / v["games"], 2) if v["games"] else None,
             "demos_pg": round(v["demos"] / v["games"], 2) if v["games"] else None,
             "def_pct": round(100 * v["demoed_def"] / v["demoed"], 1) if v["demoed"] else None} for k, v in per.items() if v["games"]]
    return {"rows": rows, "team": team, "spots": spots,
            "top_demoers": [{"key": k, "label": opp_name[k], "demos": n} for k, n in by_opp.most_common(6) if n >= 3]}


# ---------------------------------------------------------------- who passes to whom
def passing(games: list[dict], disp: str | None) -> dict:
    mem = _members(games, disp)
    lab = dict(mem)
    pair = defaultdict(lambda: {"n": 0, "shot": 0, "goal": 0, "failed": 0})
    team = {"us": 0, "them": 0, "us_goal": 0, "them_goal": 0}
    for m in games:
        ci = _ci(m)
        names = {_rname(p): k for k, _l in mem if (p := _player(m, k))}
        for x in m.get("rpass") or []:
            side = "us" if x["team"] == ci else "them"
            team[side] += 1
            team[side + "_goal"] += x["goal"]
            a, b = names.get(x["from"]), names.get(x["to"])
            if side == "us" and a and b:
                d = pair[(a, b)]
                d["n"] += 1
                d["shot"] += x["shot"]
                d["goal"] += x["goal"]
        for x in m.get("raim") or []:  # aimed passes that didn't arrive
            if x["team"] != ci or x["ok"]:
                continue
            a, b = names.get(x["from"]), names.get(x["to"])
            if a and b:
                pair[(a, b)]["failed"] += 1
    n = len(games) or 1
    return {"pairs": sorted([{"from": a, "from_label": lab[a], "to": b, "to_label": lab[b], **v,
                               "shot_pct": round(100 * v["shot"] / v["n"], 1) if v["n"] else None,
                               "completion": round(100 * v["n"] / (v["n"] + v["failed"]), 1) if v["n"] + v["failed"] else None}
                              for (a, b), v in pair.items() if v["n"]],
                             key=lambda x: -x["n"]),
            "team": {"us_pg": round(team["us"] / n, 1), "them_pg": round(team["them"] / n, 1), "us_goal": team["us_goal"], "them_goal": team["them_goal"]}}


# ---------------------------------------------------------------- one game: xG on each shot (for the game window)
def game_xg(m: dict) -> dict:
    """Puts each shot's xG on it and returns each team's total: {"blue": .., "orange": ..}."""
    out = {"blue": 0.0, "orange": 0.0}
    for s in m.get("rshots") or []:
        x = xg_of(s)
        if x is None:
            continue
        s["xg"] = round(x, 3)
        out["blue" if s.get("team") == 0 else "orange"] += x
    return {k: round(v, 2) for k, v in out.items()}
