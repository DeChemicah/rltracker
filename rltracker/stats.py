"""Turn stored matches into dashboard numbers for "me"."""
from __future__ import annotations

import difflib
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime

from . import store

NON_STAT = {"ping"}


def norm_name(name: str) -> str:
    n = re.sub(r"^\s*[\[(][^\])]{1,6}[\])]\s*", "", name or "")  # clan tag
    n = n.lower().replace("0", "o")
    return re.sub(r"[^a-z0-9]", "", n)


def same_player(a: str, b: str) -> bool:
    na, nb = norm_name(a), norm_name(b)
    if not na or not nb:
        return False
    return na == nb or difflib.SequenceMatcher(None, na, nb).ratio() >= 0.82


def alias_for(name: str, aliases: list[dict]) -> str | None:
    """The nickname you gave this player, tolerant of small name differences."""
    best = None
    na = norm_name(name)
    for a in aliases or []:
        nb = norm_name(a.get("name", ""))
        if not na or not nb:
            continue
        r = 1.0 if na == nb else difflib.SequenceMatcher(None, na, nb).ratio()
        if r >= 0.82 and (best is None or r > best[0]):
            best = (r, a.get("alias"))
    return best[1] if best else None


def my_display(me: list[str], aliases: list[dict]) -> str | None:
    """What to call you: the alias you gave any of your own accounts
    (e.g. an alt account's name -> your alias), otherwise your first in-game name."""
    for name in me or []:
        a = alias_for(name, aliases)
        if a and a.strip().lower() != "me":
            return a
    return me[0] if me else None


def apply_aliases(m: dict, aliases: list[dict], me: list[str] | None = None) -> dict:
    idc = _id_config()
    new_links = {}
    for t in m.get("teams", []):
        for p in t["players"]:
            pid = p.get("platform_id")
            a = alias_for(p["name"], aliases)
            if pid and not a and pid in idc["alias_ids"]:
                a = idc["alias_ids"][pid]  # same account, different name
            elif pid and a and idc["alias_ids"].get(pid) != a:
                new_links[pid] = a  # learn: this account is that alias
            p["alias"] = a
    if new_links:
        cfg = store.load_config()
        cfg["alias_ids"] = {**(cfg.get("alias_ids") or {}), **new_links}
        store.save_config(cfg)
    mine = m.get("my")
    if mine is not None and me:
        a = (mine.get("alias") or "").strip()
        if not a or a.lower() == "me":  # you on any account -> your name
            d = my_display(me, aliases)
            mine["alias"] = d if d and d != mine["name"] else None
    # Stable id for profile links: "me", "@alias", or the normalised name.
    for t in m.get("teams", []):
        for p in t["players"]:
            p["key"] = "me" if p is mine else _pkey(p)
    return m


def detect_me(matches: list[dict], cfg: dict) -> list[str]:
    names = [n for n in cfg.get("player_names") or [] if n.strip()]
    names += [a["name"] for a in cfg.get("aliases") or [] if (a.get("alias") or "").strip().lower() == "me"]
    if names:
        return names
    # Best signal: the profile card in the corner of post-game screens.
    cards: Counter = Counter()
    card_display: dict[str, str] = {}
    for m in matches:
        lp = m.get("local_player")
        if lp and norm_name(lp):
            cards[norm_name(lp)] += 1
            card_display.setdefault(norm_name(lp), re.sub(r"^\s*[\[(][^\])]{1,6}[\])]\s*", "", lp))
    if cards:
        return [card_display[k] for k, _ in cards.most_common() if _ >= 2 or len(cards) == 1][:3] or [card_display[cards.most_common(1)[0][0]]]
    # Otherwise whoever shows up in the most scoreboards is probably you.
    counts: Counter = Counter()
    display: dict[str, str] = {}
    for m in matches:
        seen = set()
        for t in m["teams"]:
            for p in t["players"]:
                k = norm_name(p["name"])
                if k and k not in seen:
                    seen.add(k)
                    counts[k] += 1
                    display.setdefault(k, re.sub(r"^\s*[\[(][^\])]{1,6}[\])]\s*", "", p["name"]))
    if not counts:
        return []
    top, n = counts.most_common(1)[0]
    if n < 2 and len(matches) > 1:
        return []
    return [display[top]]


_ids_cache = {"mtime": None, "ids": set(), "alias_ids": {}}


def _id_config():
    """Your account ids and alias<->account links from config (cached by file time)."""
    try:
        mt = os.path.getmtime(store.CONFIG_PATH)
    except OSError:
        mt = None
    if _ids_cache["mtime"] != mt:
        cfg = store.load_config()
        _ids_cache.update(mtime=mt, ids=set(cfg.get("my_ids") or []), alias_ids=dict(cfg.get("alias_ids") or {}))
    return _ids_cache


def find_me(m: dict, me: list[str]):
    """Return (team_index, player) for me in this match, or (None, None)."""
    ids = _id_config()["ids"]
    if ids and not m.get("me"):  # replay account id: exact, survives name changes
        for ti, t in enumerate(m["teams"]):
            for p in t["players"]:
                if p.get("platform_id") in ids:
                    return ti, p
    if m.get("me"):  # manual override stored on the match
        ti, pi = m["me"]
        try:
            return ti, m["teams"][ti]["players"][pi]
        except (IndexError, KeyError, TypeError):
            pass
    best = None
    for ti, t in enumerate(m["teams"]):
        for p in t["players"]:
            for name in me:
                r = difflib.SequenceMatcher(None, norm_name(p["name"]), norm_name(name)).ratio()
                if r >= 0.82 and (best is None or r > best[0]):
                    best = (r, ti, p)
    return (best[1], best[2]) if best else (None, None)


def mark_duplicates(matches: list[dict]) -> None:
    """The same match stored twice a few minutes apart (older data): keep the last."""
    ms = sorted(matches, key=lambda m: m["played_at"])
    for a, b in zip(ms, ms[1:]):
        if a.get("hidden"):
            continue
        try:
            dt = (datetime.fromisoformat(b["played_at"]) - datetime.fromisoformat(a["played_at"])).total_seconds()
        except ValueError:
            continue
        if dt > 12 * 60:
            continue
        na = sorted(norm_name(p["name"]) for t in a["teams"] for p in t["players"])
        nb = sorted(norm_name(p["name"]) for t in b["teams"] for p in t["players"])
        if len(na) != len(nb):
            continue
        same = sum(1 for x in na if any(difflib.SequenceMatcher(None, x, y).ratio() > 0.8 for y in nb))
        scores_a = sorted(p["stats"].get("score", 0) for t in a["teams"] for p in t["players"])
        scores_b = sorted(p["stats"].get("score", 0) for t in b["teams"] for p in t["players"])
        if same >= 0.75 * len(na) and (scores_a == scores_b or not a.get("is_final")):
            a["duplicate_of"] = b["id"]


def playlist_of(m: dict) -> str:
    n = m.get("team_size") or max((len(t["players"]) for t in m["teams"]), default=0)
    if not n:
        return "?"
    return f"{n}v{n}" if n <= 4 else "custom"  # >4 rows = private match / subs


def enrich(m: dict, me: list[str]) -> dict:
    ti, p = find_me(m, me)
    m["playlist"] = playlist_of(m)
    m["my_team"] = ti
    m["my"] = p
    m["result"] = _result_for(m, ti)
    # MVP = top score on the winning team. The winner flag is only stored for some games (replays only
    # mark it when orange wins), so fall back to the final score like the result does.
    m["mvp"] = None
    win = next((t for t in m["teams"] if t.get("winner")), None)
    if win is None and len(m["teams"]) == 2:
        a, b = (t.get("score") for t in m["teams"])
        if a is not None and b is not None and a != b:
            win = m["teams"][0 if a > b else 1]
    if win and win["players"]:
        m["mvp"] = max(win["players"], key=lambda q: q["stats"].get("score", 0))["name"]
    return m


def _result_for(m: dict, ti) -> str | None:
    if ti is None or len(m["teams"]) != 2:
        return None
    mine, other = m["teams"][ti], m["teams"][1 - ti]
    if mine.get("winner"):
        return "W"
    if other.get("winner"):
        return "L"
    if mine.get("score") is not None and other.get("score") is not None and m.get("is_final"):
        return "W" if mine["score"] > other["score"] else "L" if mine["score"] < other["score"] else None
    return None


def goal_participation(pairs) -> dict | None:
    """Share of the team's goals the player scored or assisted, over (player, their team) pairs."""
    involved = team_goals = 0
    for p, t in pairs:
        tg = t.get("score")
        if tg is None:
            tg = sum((q.get("stats") or {}).get("goals") or 0 for q in t["players"])
        if not tg:
            continue
        s = p.get("stats") or {}
        involved += min(tg, (s.get("goals") or 0) + (s.get("assists") or 0))
        team_goals += tg
    return {"pct": round(100 * involved / team_goals, 1), "involved": involved, "team_goals": team_goals} if team_goals else None


ACCOUNT = "me:"  # who = "me:<platform id>": you, on one account only


def is_me_key(key: str | None) -> bool:
    return key == "me" or bool(key and key.startswith(ACCOUNT))


def key_matches(p: dict, key: str) -> bool:
    """Does this player entry belong to `key`? 'me:<id>' = you on that account."""
    if key.startswith(ACCOUNT):
        return p.get("key") == "me" and p.get("platform_id") == key[len(ACCOUNT):]
    return p.get("key") == key


def reanchor(m: dict, who: str | None) -> dict:
    """View a match from another player's side (after enrich + apply_aliases):
    'my' becomes that player, 'my_team' their team, 'result' their result.
    who = a player key ('me', 'me:<account id>', '@alias', or a normalised name). Not in it → my=None."""
    if not who or who == "me":
        return m
    for ti, t in enumerate(m["teams"]):
        for p in t["players"]:
            if key_matches(p, who):
                m["my"], m["my_team"], m["result"] = p, ti, _result_for(m, ti)
                return m
    m["my"], m["my_team"], m["result"] = None, None, None
    return m


def viewers(matches: list[dict], me_label: str | None) -> list[dict]:
    """Players the dashboard can be viewed as: you, anyone with an alias, or seen in 3+ games."""
    seen: dict[str, dict] = {}
    for m in matches:
        if m["hidden"] or m.get("duplicate_of"):
            continue
        for t in m["teams"]:
            for p in t["players"]:
                k = p.get("key")
                if not k:
                    continue
                e = seen.setdefault(k, {"key": k, "label": (me_label or p["name"]) if k == "me" else (p.get("alias") or p["name"]), "games": 0})
                e["games"] += 1
    out = [e for e in seen.values() if e["key"] == "me" or e["key"].startswith("@") or e["games"] >= 3]
    # your accounts, one by one, when you've played on more than one (e.g. an alt at a lower rank)
    accts: dict[str, dict] = {}
    for m in matches:
        if m["hidden"] or m.get("duplicate_of"):
            continue
        for t in m["teams"]:
            for p in t["players"]:
                if p.get("key") == "me" and p.get("platform_id"):
                    e = accts.setdefault(p["platform_id"], {"key": ACCOUNT + p["platform_id"], "games": 0, "names": Counter(),
                                                            "platform": p["platform_id"].split(":")[0]})
                    e["games"] += 1
                    e["names"][p.get("replay_name") or p["name"]] += 1
    if len(accts) > 1:
        for e in out:
            if e["key"] == "me":
                e["label"] = f"{e['label']} · all accounts"
        for e in sorted(accts.values(), key=lambda e: -e["games"]):
            out.append({"key": e["key"], "label": f"{e['names'].most_common(1)[0][0]} ({e['platform']})", "games": e["games"],
                        "account": True})
    return sorted(out, key=lambda e: (not is_me_key(e["key"]), e["key"] != "me", -e["games"], e["label"].lower()))


def summary(size: str = "all", ranked: str = "all", last: int = 0, who: str = "me", start: str = "", end: str = "",
            tag: str = "") -> dict:
    cfg = store.load_config()
    matches = store.all_matches()
    me = detect_me([m for m in matches if not m["hidden"]], cfg)
    mark_duplicates(matches)
    aliases = cfg.get("aliases") or []
    for m in matches:
        enrich(m, me)
        apply_aliases(m, aliases, me)
    me_label = my_display(me, aliases)
    view_list = viewers(matches, me_label)
    who = who if any(v["key"] == who for v in view_list) else "me"
    for m in matches:
        reanchor(m, who)

    def keep(m):
        if m["hidden"] or m.get("duplicate_of"):
            return False
        if size != "all" and m["playlist"] != size:
            return False
        if ranked == "ranked" and not m.get("ranked"):
            return False
        if ranked == "casual" and m.get("ranked"):
            return False
        if start and m["played_at"] < start or end and m["played_at"] > end:  # timeframe (a session or a date range)
            return False
        if tag and tag not in (m.get("tags") or []):
            return False
        return True

    pool = [m for m in matches if keep(m)]
    if last:
        pool = pool[-last:]
    mine = [m for m in pool if m["my"] is not None]

    wins = sum(1 for m in mine if m["result"] == "W")
    losses = sum(1 for m in mine if m["result"] == "L")
    streak, streak_kind = 0, None
    for m in reversed(mine):
        if m["result"] not in ("W", "L"):
            continue
        if streak_kind is None:
            streak_kind = m["result"]
        if m["result"] != streak_kind:
            break
        streak += 1

    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    bests: dict[str, dict] = {}
    for m in mine:
        for k, v in m["my"]["stats"].items():
            if k in NON_STAT or v is None:
                continue
            totals[k] += v
            counts[k] += 1
            if k not in bests or v > bests[k]["value"]:
                bests[k] = {"value": v, "match_id": m["id"], "played_at": m["played_at"]}
    averages = {k: round(totals[k] / counts[k], 2) for k in totals}
    mvps = sum(1 for m in mine if m["mvp"] and m["my"] and m["mvp"] == m["my"]["name"])
    shooting = round(100 * totals["goals"] / totals["shots"], 1) if totals.get("shots") else None
    gp = goal_participation([(m["my"], m["teams"][m["my_team"]]) for m in mine])

    # Compare against everyone else in the same matches.
    lobby_tot: dict[str, float] = defaultdict(float)
    lobby_cnt: dict[str, int] = defaultdict(int)
    for m in mine:
        for t in m["teams"]:
            for p in t["players"]:
                if p is m["my"]:
                    continue
                for k, v in p["stats"].items():
                    if k not in NON_STAT and v is not None:
                        lobby_tot[k] += v
                        lobby_cnt[k] += 1
    lobby_avg = {k: round(lobby_tot[k] / lobby_cnt[k], 2) for k in lobby_tot}


    mates: dict[str, dict] = {}
    opps: dict[str, dict] = {}
    for m in mine:
        for ti, t in enumerate(m["teams"]):
            for p in t["players"]:
                if p is m["my"]:
                    continue
                k = ("@" + p["alias"].lower()) if p.get("alias") else norm_name(p["name"])
                if not k:
                    continue
                book = mates if ti == m["my_team"] else opps
                e = book.setdefault(k, {"name": p["name"], "alias": p.get("alias"), "key": p.get("key") or k, "games": 0, "wins": 0,
                                        "stats": defaultdict(float)})
                e["games"] += 1
                e["wins"] += m["result"] == "W"
                for sk, sv in p["stats"].items():
                    if sv is not None and sk not in NON_STAT:
                        e["stats"][sk] += sv

    def finish(book):
        out = []
        for e in book.values():
            if e["games"] < 2 and not e["alias"]:
                continue
            e["averages"] = {k: round(v / e["games"], 2) for k, v in e.pop("stats").items()}
            out.append(e)
        return sorted(out, key=lambda e: (not e["alias"], -e["games"]))[:12]

    top_mates = finish(mates)
    top_opps = finish(opps)

    playlists = sorted({m["playlist"] for m in matches if not m["hidden"]})
    return {
        "me": me,
        "me_display": me_label if who == "me" else next(v["label"] for v in view_list if v["key"] == who),
        "who": who, "viewers": view_list,
        "playlists": playlists,
        "games": len(mine),
        "unmatched": len(pool) - len(mine),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100 * wins / (wins + losses), 1) if wins + losses else None,
        "streak": {"kind": streak_kind, "n": streak},
        "mvps": mvps,
        "shooting_pct": shooting,
        "goal_participation": gp,
        "averages": averages,
        "lobby_averages": lobby_avg,
        "totals": {k: int(v) for k, v in totals.items()},
        "bests": bests,
        "teammates": top_mates,
        "opponents": top_opps,
        "recent": [match_row(m) for m in reversed(mine)][:20],
        "columns": _columns(mine),
    }


def _columns(ms):
    seen = []
    for m in ms:
        for c in m.get("columns", []):
            if c["key"] not in [x["key"] for x in seen]:
                seen.append(c)
    return seen


def match_row(m: dict) -> dict:
    teams = m["teams"]
    my_t = m.get("my_team")
    return {
        "id": m["id"],
        "played_at": m["played_at"],
        "playlist": m.get("playlist"),
        "ranked": m.get("ranked"),
        "result": m.get("result"),
        "is_final": m.get("is_final"),
        "hidden": m.get("hidden"),
        "edited": m.get("edited"),
        "duplicate_of": m.get("duplicate_of"),
        "score_line": [t.get("score") for t in teams],
        "colors": [t.get("color") for t in teams],
        "my_team": my_t,
        "my": m.get("my"),
        "mvp": m.get("mvp"),
    }


def match_list() -> list[dict]:
    cfg = store.load_config()
    matches = store.all_matches()
    me = detect_me([m for m in matches if not m["hidden"]], cfg)
    mark_duplicates(matches)
    aliases = cfg.get("aliases") or []
    return [match_row(apply_aliases(enrich(m, me), aliases, me)) for m in reversed(matches)]


def _pkey(p: dict) -> str:
    a = (p.get("alias") or "").strip()
    return "@" + a.lower() if a else norm_name(p["name"])


STYLE_NAMES = {"striker": "Striker", "keeper": "Keeper", "playmaker": "Playmaker"}


def play_style(rel: dict) -> dict | None:
    """Play style from a player's share of their team's stats, where 1.0 = a fair
    share for the team size (so 2v2 and 3v3 compare fairly).
    striker = goals + shots, keeper = saves, playmaker = ball hits (score if no replays).
    Differences are cubed so a clear lean actually moves the dot off the middle."""
    def mean(*ks):
        vs = [rel[k] for k in ks if rel.get(k) is not None]
        return sum(vs) / len(vs) if vs else None
    axes = {"striker": mean("goals", "shots"), "keeper": mean("saves"),
            "playmaker": mean("ball_hits") if rel.get("ball_hits") is not None else mean("score")}
    if any(v is None for v in axes.values()):
        return None
    w = {k: max(v, 0.05) ** 3 for k, v in axes.items()}
    tot = sum(w.values())
    w = {k: round(v / tot, 3) for k, v in w.items()}
    top = max(w, key=w.get)
    second = sorted(w.values())[-2]
    if w[top] < 0.42:
        style = "All-rounder"
    elif second >= 0.36:
        a, b = sorted([k for k in w if w[k] >= 0.36], key=lambda k: -w[k])[:2]
        style = f"{STYLE_NAMES[a]} / {STYLE_NAMES[b]}"
    else:
        style = STYLE_NAMES[top]
    return {"weights": w, "ratio": {k: round(v, 2) for k, v in axes.items()}, "style": style,
            "playmaker_from": "ball_hits" if rel.get("ball_hits") is not None else "score"}


def team(lineup: str = "", size: str = "all", ranked: str = "all") -> dict:
    """Your team's tendencies and +/- against opponents, per stat.

    lineup: comma-separated teammate keys ("" = the most-played lineup,
    "any" = every game you played).
    """
    cfg = store.load_config()
    matches = store.all_matches()
    me = detect_me([m for m in matches if not m["hidden"]], cfg)
    mark_duplicates(matches)
    aliases = cfg.get("aliases") or []
    games = []
    for m in matches:
        if m["hidden"] or m.get("duplicate_of") or len(m["teams"]) != 2:
            continue
        enrich(m, me)
        apply_aliases(m, aliases, me)
        if m["my"] is None or m["result"] not in ("W", "L"):
            continue
        if size != "all" and m["playlist"] != size:
            continue
        if ranked == "ranked" and not m.get("ranked"):
            continue
        if ranked == "casual" and m.get("ranked"):
            continue
        mates = [p for p in m["teams"][m["my_team"]]["players"] if p is not m["my"]]
        m["_lineup"] = ",".join(sorted(_pkey(p) for p in mates))
        m["_labels"] = {_pkey(p): (p.get("alias") or p["name"]) for p in mates}
        games.append(m)

    # Lineups you've played, most common first.
    lineups: dict[str, dict] = {}
    for m in games:
        e = lineups.setdefault(m["_lineup"], {"key": m["_lineup"], "games": 0, "wins": 0,
                                              "label": ", ".join(sorted(m["_labels"].values())) or "solo"})
        e["games"] += 1
        e["wins"] += m["result"] == "W"
    lineup_list = sorted(lineups.values(), key=lambda e: (-e["games"], e["label"]))
    if not lineup:
        lineup = lineup_list[0]["key"] if lineup_list else "any"
    pool = games if lineup == "any" else [m for m in games if m["_lineup"] == lineup]

    cols = []
    for m in pool:
        for c in m.get("columns", []):
            if c["key"] not in NON_STAT and c["key"] not in [x["key"] for x in cols]:
                cols.append({"key": c["key"], "label": c["label"]})
    keys = [c["key"] for c in cols]

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    def team_total(players, k):
        vals = [p["stats"].get(k) for p in players]
        return sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None

    def side_total(team_, k):
        # Goals = the team's actual score (includes own goals the scoreboard
        # doesn't credit to anyone), so it matches the final result.
        if k == "goals" and team_.get("score") is not None:
            return team_["score"]
        return team_total(team_["players"], k)

    per_game = []
    players: dict[str, dict] = {}
    for m in pool:
        mine = m["teams"][m["my_team"]]
        other = m["teams"][1 - m["my_team"]]
        us = {k: side_total(mine, k) for k in keys}
        them = {k: side_total(other, k) for k in keys}
        per_game.append({
            "id": m["id"], "played_at": m["played_at"], "result": m["result"], "playlist": m["playlist"],
            "ranked": m.get("ranked"), "us": us, "them": them,
            "diff": {k: (us[k] - them[k] if us[k] is not None and them[k] is not None else None) for k in keys},
        })
        for p in mine["players"]:
            pk = "me" if p is m["my"] else _pkey(p)
            e = players.setdefault(pk, {
                "key": pk, "label": (my_display(me, aliases) or p["name"]) if pk == "me" else (p.get("alias") or p["name"]),
                "name": p["name"], "is_me": pk == "me", "games": 0, "wins": 0, "mvps": 0,
                "_v": {k: [] for k in keys}, "_share": {k: [] for k in keys}, "_rel": {k: [] for k in keys},
            })
            e["games"] += 1
            e["wins"] += m["result"] == "W"
            e["mvps"] += m["mvp"] == p["name"]
            for k in keys:
                v = p["stats"].get(k)
                e["_v"][k].append(v)
                tot = team_total(mine["players"], k)
                if v is not None and tot:
                    e["_share"][k].append(v / tot)
                    e["_rel"][k].append(v / tot * len(mine["players"]))  # 1.0 = a fair share for the team size

    def summarize(rows):
        out = []
        for k in keys:
            u = avg([g["us"][k] for g in rows])
            t = avg([g["them"][k] for g in rows])
            d = avg([g["diff"][k] for g in rows])
            won = sum(1 for g in rows if g["diff"][k] is not None and g["diff"][k] > 0)
            counted = sum(1 for g in rows if g["diff"][k] is not None)
            us_vals = [g["us"][k] for g in rows if g["us"][k] is not None]
            them_vals = [g["them"][k] for g in rows if g["them"][k] is not None]
            out.append({"key": k, "us": u, "them": t, "diff": d,
                        "us_total": sum(us_vals) if us_vals else None,
                        "them_total": sum(them_vals) if them_vals else None,
                        "diff_total": sum(g["diff"][k] for g in rows if g["diff"][k] is not None) if counted else None,
                        "won_pct": round(100 * won / counted) if counted else None})
        return out

    plist = []
    for e in players.values():
        if lineup == "any" and e["games"] < 2 and not e["is_me"]:
            continue
        plist.append({
            "key": e["key"], "label": e["label"], "name": e["name"], "is_me": e["is_me"],
            "games": e["games"], "wins": e["wins"], "mvps": e["mvps"],
            "avg": {k: avg(e["_v"][k]) for k in keys},
            "share": {k: (round(100 * sum(e["_share"][k]) / len(e["_share"][k]), 1) if e["_share"][k] else None) for k in keys},
            "role": play_style({k: (sum(v) / len(v) if v else None) for k, v in e["_rel"].items()}),
        })
    plist.sort(key=lambda e: (not e["is_me"], -e["games"]))

    wins = sum(m["result"] == "W" for m in pool)
    return {
        "replay": _team_replay(pool, plist),
        "me": me,
        "lineups": lineup_list,
        "lineup": lineup,
        "playlists": sorted({m["playlist"] for m in games}),
        "games": len(pool),
        "wins": wins,
        "losses": len(pool) - wins,
        "win_rate": round(100 * wins / len(pool), 1) if pool else None,
        "columns": cols,
        "overall": summarize(per_game),                                  # us vs them, every stat
        "in_wins": summarize([g for g in per_game if g["result"] == "W"]),
        "in_losses": summarize([g for g in per_game if g["result"] == "L"]),
        "per_game": list(reversed(per_game)),
        "players": plist,
    }


def records(player: str = "universal", top: int = 5, start: str = "", end: str = "",
            size: str = "all", ranked: str = "all", tag: str = "") -> dict:
    """Single-game bests for one player, or for everyone who's played on your
    team ("universal"), plus career numbers and team records. start/end (ISO times)
    limit it to one session's games."""
    cfg = store.load_config()
    matches = store.all_matches()
    me = detect_me([m for m in matches if not m["hidden"]], cfg)
    mark_duplicates(matches)
    aliases = cfg.get("aliases") or []

    perfs, team_games = [], []
    cols: list[dict] = []
    for m in matches:
        if m["hidden"] or m.get("duplicate_of"):
            continue
        if start and m["played_at"] < start or end and m["played_at"] > end:
            continue
        if tag and tag not in (m.get("tags") or []):
            continue
        enrich(m, me)
        apply_aliases(m, aliases, me)
        if m["my"] is None:
            continue
        if size != "all" and m["playlist"] != size:  # the mode and ranked/casual filters
            continue
        if ranked == "ranked" and not m.get("ranked") or ranked == "casual" and m.get("ranked"):
            continue
        for c in m.get("columns", []):
            if c["key"] not in NON_STAT and c["key"] not in [x["key"] for x in cols]:
                cols.append({"key": c["key"], "label": c["label"]})
        mine = m["teams"][m["my_team"]]
        other = m["teams"][1 - m["my_team"]] if len(m["teams"]) == 2 else None
        for p in mine["players"]:
            pk = "me" if p is m["my"] else _pkey(p)
            perfs.append({
                "key": pk, "label": (my_display(me, aliases) or p["name"]) if pk == "me" else (p.get("alias") or p["name"]),
                "name": p["name"], "match_id": m["id"], "played_at": m["played_at"], "result": m["result"],
                "playlist": m["playlist"], "ranked": m.get("ranked"), "mvp": m["mvp"] == p["name"],
                "stats": p["stats"], "rstats": p.get("rstats") if "bpm" in (p.get("rstats") or {}) else None,
            })
        if other is not None and m["result"] in ("W", "L"):  # finished games only
            def side(t):
                out = {}
                for c in m.get("columns", []):
                    k = c["key"]
                    if k in NON_STAT:
                        continue
                    if k == "goals" and t.get("score") is not None:
                        out[k] = t["score"]  # final score, includes own goals
                    else:
                        vals = [q["stats"].get(k) for q in t["players"] if q["stats"].get(k) is not None]
                        out[k] = sum(vals) if vals else None
                out.setdefault("goals", t.get("score"))
                return out
            team_games.append({"match_id": m["id"], "played_at": m["played_at"], "result": m["result"],
                               "playlist": m["playlist"], "for": mine.get("score"), "against": other.get("score"),
                               "us": side(mine), "them": side(other)})

    # Who can be picked: you, anyone with an alias, or seen 2+ times on your team.
    counts: dict[str, dict] = {}
    for p in perfs:
        e = counts.setdefault(p["key"], {"key": p["key"], "label": p["label"], "games": 0,
                                         "aliased": p["key"].startswith("@") or p["key"] == "me"})
        e["games"] += 1
    pickable = sorted((e for e in counts.values() if e["aliased"] or e["games"] >= 2),
                      key=lambda e: (e["key"] != "me", -e["games"], e["label"].lower()))

    pool = perfs if player == "universal" else [p for p in perfs if p["key"] == player]
    keys = [c["key"] for c in cols]
    n_top = top if player != "universal" else max(top, 10)

    def entry(p, val):
        return {"value": val, "label": p["label"], "key": p["key"], "match_id": p["match_id"], "played_at": p["played_at"],
                "result": p["result"], "playlist": p["playlist"]}

    stat_records = []
    for k in keys + ["ping"] * 0:
        ranked_ = sorted((p for p in pool if (p["stats"].get(k) or 0) > 0),
                         key=lambda p: (-p["stats"][k], p["played_at"]))
        if not ranked_:
            continue
        stat_records.append({"key": k, "label": next(c["label"] for c in cols if c["key"] == k),
                             "top": [entry(p, p["stats"][k]) for p in ranked_[:n_top]]})

    from . import replay_stats as RS
    rpool = [p for p in pool if p["rstats"]]
    replay_records = []
    for k in RS.RECORD_KEYS:
        ranked_ = sorted((p for p in rpool if (p["rstats"].get(k) or 0) > 0),
                         key=lambda p: (-p["rstats"][k], p["played_at"]))
        if ranked_:
            replay_records.append({"key": k, "label": RS.LABELS[k], "top": [entry(p, p["rstats"][k]) for p in ranked_[:n_top]]})

    career = None
    if player != "universal" and pool:
        ordered = sorted(pool, key=lambda p: p["played_at"])
        best_streak = cur = 0
        for p in ordered:
            cur = cur + 1 if p["result"] == "W" else 0
            best_streak = max(best_streak, cur)
        wins = sum(p["result"] == "W" for p in pool)
        losses = sum(p["result"] == "L" for p in pool)
        career = {
            "games": len(pool), "wins": wins, "losses": losses,
            "win_rate": round(100 * wins / (wins + losses), 1) if wins + losses else None,
            "mvps": sum(p["mvp"] for p in pool), "best_win_streak": best_streak,
            "totals": {k: sum(p["stats"][k] for p in pool if p["stats"].get(k) is not None)
                       for k in keys if any(p["stats"].get(k) is not None for p in pool)},
            "averages": {k: round(statistics.mean(p["stats"][k] for p in pool if p["stats"].get(k) is not None), 2)
                         for k in keys if any(p["stats"].get(k) is not None for p in pool)},
            "first": ordered[0]["played_at"], "last": ordered[-1]["played_at"],
        }

    # Game records: both teams' totals per game (goals = final score).
    game_pool = team_games if player == "universal" else [
        g for g in team_games if g["match_id"] in {p["match_id"] for p in pool}]

    def gentry(g, val):
        return {"value": val, "match_id": g["match_id"], "played_at": g["played_at"], "result": g["result"],
                "playlist": g["playlist"], "for": g["for"], "against": g["against"],
                "us": g["us"], "them": g["them"]}

    game_records = []
    stat_keys = ["goals"] + [k for k in keys if k != "goals"]
    for k in stat_keys:
        rows = [g for g in game_pool if g["us"].get(k) is not None and g["them"].get(k) is not None]
        if not rows:
            continue
        lab = "Goals" if k == "goals" else next(c["label"].title() for c in cols if c["key"] == k)
        cats = [
            ("Most combined", lambda g: g["us"][k] + g["them"][k], max),
            ("Most for", lambda g: g["us"][k], max),
            ("Most against", lambda g: g["them"][k], max),
            ("Fewest for", lambda g: g["us"][k], min),
            ("Fewest against", lambda g: g["them"][k], min),
            ("Best +/-", lambda g: g["us"][k] - g["them"][k], max),
            ("Worst +/-", lambda g: g["us"][k] - g["them"][k], min),
        ]
        out = {}
        for name, fn, pick in cats:
            # ties go to the most recent game
            g = pick(rows, key=lambda g: (fn(g), g["played_at"]) if pick is max else (fn(g), [-ord(ch) for ch in g["played_at"]]))
            out[name] = gentry(g, fn(g))
        game_records.append({"key": k, "label": lab, "records": out})

    team_records = []
    ordered = sorted(game_pool, key=lambda r: r["played_at"])
    for res, title in (("W", "Longest win streak"), ("L", "Longest losing streak")):
        best = cur = 0
        best_end = None
        for r in ordered:
            cur = cur + 1 if r["result"] == res else 0
            if cur > best:
                best, best_end = cur, r
        if best:
            team_records.append({"label": title, "value": best, "match_id": best_end["match_id"],
                                 "played_at": best_end["played_at"], "result": best_end["result"],
                                 "playlist": best_end["playlist"]})

    return {
        "player": player,
        "range": [start, end] if start or end else None,  # lets the page check the filter was applied
        "players": pickable,
        "columns": cols,
        "games": len({p["match_id"] for p in pool}),
        "stat_records": stat_records,
        "replay_records": replay_records,
        "replay_games": len({p["match_id"] for p in rpool}),
        "career": career,
        "team_records": team_records,
        "game_records": game_records,
    }


def _team_replay(pool: list[dict], plist: list[dict]) -> dict:
    """Replay stats, our team vs theirs (average per player), for games where
    every player on both teams has replay data."""
    from . import replay_stats as RS
    from . import replay_position as RP
    rows = []
    rot_us, rot_them = [], []
    per_player: dict[str, list] = {}
    for m in pool:
        mine = m["teams"][m["my_team"]]["players"]
        other = m["teams"][1 - m["my_team"]]["players"]
        if not mine or not other or not all(RS.has(p) for p in mine + other):
            continue
        rows.append({"result": m["result"], "us": RS.avg_of(mine), "them": RS.avg_of(other)})
        if m["teams"][m["my_team"]].get("rotation") and m["teams"][1 - m["my_team"]].get("rotation"):
            rot_us.append((m["result"], m["teams"][m["my_team"]]["rotation"]))
            rot_them.append((m["result"], m["teams"][1 - m["my_team"]]["rotation"]))
        for p in mine:
            per_player.setdefault("me" if p is m["my"] else _pkey(p), []).append(p)

    def summarize(rs):
        out = {}
        for k in RS.KEYS:
            if not rs:
                break
            pair = [r for r in rs if r["us"].get(k) is not None and r["them"].get(k) is not None]
            if not pair:
                continue
            u = round(sum(r["us"][k] for r in pair) / len(pair), 2)
            t = round(sum(r["them"][k] for r in pair) / len(pair), 2)
            more = sum(1 for r in pair if r["us"][k] > r["them"][k])
            out[k] = {"us": u, "them": t, "diff": round(u - t, 2), "won_pct": round(100 * more / len(pair))}
        return out

    wins = [r for r in rows if r["result"] == "W"]
    losses = [r for r in rows if r["result"] == "L"]
    players = [{"key": e["key"], "label": e["label"], "is_me": e["is_me"], "games": len(per_player[e["key"]]),
                "avg": RS.avg_of(per_player[e["key"]]),
                "heat": RP.avg_heat([p.get("rheat") for p in per_player[e["key"]]])} for e in plist if e["key"] in per_player]

    def rot(sel):
        us_ = RP.avg_rotation([r for res, r in rot_us if sel(res)])
        them_ = RP.avg_rotation([r for res, r in rot_them if sel(res)])
        return {k: {"us": us_.get(k), "them": them_.get(k),
                    "diff": round(us_[k] - them_[k], 1) if k in us_ and k in them_ else None} for k, _l in RS.ROTATION}
    rotation = {"games": len(rot_us), "overall": rot(lambda r: True), "in_wins": rot(lambda r: r == "W"),
                "in_losses": rot(lambda r: r == "L")}
    return {"games": len(rows), "meta": RS.meta(), "rotation": rotation, "overall": summarize(rows),
            "in_wins": summarize(wins), "in_losses": summarize(losses), "players": players}


def history() -> list[dict]:
    """Every match with full scoreboards, for the History tab."""
    cfg = store.load_config()
    matches = store.all_matches()
    me = detect_me([m for m in matches if not m["hidden"]], cfg)
    mark_duplicates(matches)
    aliases = cfg.get("aliases") or []
    out = []
    for m in reversed(matches):
        enrich(m, me)
        apply_aliases(m, aliases, me)
        my_index = None
        if m["my"] is not None:
            my_index = [m["my_team"], m["teams"][m["my_team"]]["players"].index(m["my"])]
        teams = []
        for t in m["teams"]:
            teams.append({k: v for k, v in t.items() if k != "players"} | {
                "players": [{k: v for k, v in p.items() if k not in ("raw", "rheat", "rpads")} for p in t["players"]]})
        out.append({
            "id": m["id"], "played_at": m["played_at"], "playlist": m["playlist"], "ranked": m.get("ranked"),
            "result": m["result"], "is_final": m.get("is_final"), "hidden": m["hidden"], "edited": m["edited"],
            "duplicate_of": m.get("duplicate_of"), "recycled": bool(m.get("deleted_at")),
            "columns": m.get("columns", []), "teams": teams, "my_index": my_index, "mvp": m["mvp"],
            "has_replay": bool(m.get("replay")), "source": m.get("source") or "older",
        })
    return out


def match_detail(mid: int) -> dict | None:
    m = store.get_match(mid)
    if not m:
        return None
    cfg = store.load_config()
    me = detect_me([x for x in store.all_matches() if not x["hidden"]], cfg)
    enrich(m, me)
    apply_aliases(m, cfg.get("aliases") or [], me)
    try:
        from . import replays, replay_stats
        m["replay_info"] = replays.match_replay_info(mid)
        m["rstats_labels"] = replay_stats.meta()
    except Exception:
        m["replay_info"] = None
    m["my_index"] = None
    if m["my"] is not None:
        t = m["teams"][m["my_team"]]
        m["my_index"] = [m["my_team"], t["players"].index(m["my"])]
        try:  # how strong the other team was, vs the average player in your games
            from . import insights
            m["difficulty"] = insights.game_difficulty(m)
        except Exception:
            m["difficulty"] = None
    m.pop("my", None)
    return m
