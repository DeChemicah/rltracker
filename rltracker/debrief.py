"""Session report ("debrief"): what to improve after a session, for the team and for each player.

Rule-based: every finding comes from the stored stats of the session's games (replays), compared
with the other team, the lobby, the player's own usual and what went with winning in this session.
Each finding has a weight (how much it seems to matter), facts, a short explanation and a tip.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime

from . import insights as I, stats


# ---------------------------------------------------------------- formatting helpers
def _rec(r: dict) -> str:
    return f"{r['wins']}–{r['games'] - r['wins']}" if r and r.get("games") else "–"


def _wr(r: dict) -> str:
    return f"{round(r['win_rate'])}%" if r and r.get("win_rate") is not None else "–"


def _v(key: str, v) -> str:
    if v is None:
        return "–"
    if key.startswith("pct_") or key in ("shooting_pct", "goal_part"):
        return f"{v:.1f}%".replace(".0%", "%")
    if key == "avg_speed":
        return f"{v * 0.036:.0f} km/h"
    if key.startswith("avg_dist"):
        return f"{v / 100:.0f} m"
    if key in ("score", "bpm", "overfill"):
        return f"{v:.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def _ord(n) -> str:
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _fact(label, value, kind=""):
    return {"label": label, "value": value, "kind": kind}


def _wc_text(r: dict) -> tuple[str, str, str]:
    """A win condition as (label, value, kind): 'X above 12.3' → 'N% wins (else M%)'."""
    hi_wins = r["diff"] > 0
    side, other = (r["above"], r["below"]) if hi_wins else (r["below"], r["above"])
    word = "above" if hi_wins else "below"
    return (f"{r['label']} {word} {_v(r['key'], r['cut'])}", f"{_wr(side)} wins (else {_wr(other)})", "up")


def _wc_bad(r: dict) -> tuple[str, str]:
    """The losing side of a win condition: 'X above 21%' → '25% wins (else 61%)'."""
    hi_wins = r["diff"] > 0
    side, other = (r["below"], r["above"]) if hi_wins else (r["above"], r["below"])
    word = "below" if hi_wins else "above"
    return f"{r['label']} {word} {_v(r['key'], r['cut'])}", f"{_wr(side)} wins (else {_wr(other)})"


def _find(fs: list, weight: float, title: str, facts: list, text: str, tip: str):
    fs.append({"weight": round(weight, 1), "title": title, "facts": facts, "text": text, "tip": tip})


def _rank(fs: list, n: int = 7) -> list:
    out = sorted(fs, key=lambda f: -f["weight"])[:n]
    for i, f in enumerate(out, 1):
        f["rank"] = i
    return out


def _speed_conv(shots: list) -> dict:
    out = {}
    for name, lo, hi in (("soft", 0, 70), ("hard", 90, 999)):
        xs = [x for x in shots if lo <= (x.get("speed") or 0) < hi]
        g = sum(x.get("outcome") == "goal" for x in xs)
        out[name] = {"shots": len(xs), "goals": g, "pct": round(100 * g / len(xs)) if xs else None}
    return out


def _factor(D: dict, title_part: str):
    return next((f for f in D["factors"] if title_part.lower() in f["title"].lower()), None)


# ---------------------------------------------------------------- the team
def _team(games: list, D: dict, sm: dict, start: str, end: str) -> dict:
    fs, good = [], []
    c = D["clutch"]
    n = len(games)
    wins = sum(m["result"] == "W" for m in games)
    us_sh = I._avg([I.total(I.us(m), "shots") for m in games])
    them_sh = I._avg([I.total(I.them(m), "shots") for m in games])
    e = D["efficiency"]

    # 1. the first goal and kickoffs
    sf, cf = c["scored_first"], c["conceded_first"]
    if sf["games"] >= 3 and cf["games"] >= 3 and (sf["win_rate"] or 0) - (cf["win_rate"] or 0) >= 20:
        kf = _factor(D, "kickoffs won")
        facts = [_fact("Scored first", f"{_rec(sf)} ({_wr(sf)})", "up"), _fact("Conceded first", f"{_rec(cf)} ({_wr(cf)})", "down")]
        txt = "The first goal decided more games than almost anything else."
        if kf:
            rows = {r["label"]: r for r in kf["rows"]}
            for lab, kind in (("We won more", "up"), ("Even", "down"), ("They won more", "")):
                if lab in rows and rows[lab]["games"]:
                    facts.append(_fact(f"Kickoffs: {lab.lower()}", f"{_rec(rows[lab])} ({_wr(rows[lab])})", kind))
            ev = rows.get("Even")
            if ev and ev["games"] >= 3 and (ev["win_rate"] or 0) < 35:
                txt += f" In the {ev['games']} games where kickoffs came out even you won only {_wr(ev)}."
        _find(fs, (sf["win_rate"] - cf["win_rate"]) * 0.9, "Score first, especially off the kickoff", facts, txt,
              "Agree a fixed kickoff routine (who goes, who cheats up, who stays back for boost) and keep it all session. Treat the first minute as the most important one.")

    # 2. finishing
    ts, os_ = e.get("team_shooting"), e.get("opp_shooting")
    if ts is not None and os_ is not None and ts < os_ - 1.5:
        sc = _speed_conv([x for x in sm["shots"] if x["side"] == "for"])
        facts = [_fact("Your team's shooting", f"{ts}%", "down"), _fact("Their shooting", f"{os_}%")]
        if us_sh and them_sh:
            facts.append(_fact("Shots a game", f"{us_sh:.1f} vs {them_sh:.1f}", "up" if us_sh > them_sh else "down"))
        txt = ("Your team took more shots but converted fewer than the other team." if us_sh and them_sh and us_sh > them_sh
               else "The other team turned their chances into goals more often than you did.")
        if sc["soft"]["pct"] is not None and sc["hard"]["pct"] is not None:
            facts.append(_fact("Shots under 70 km/h", f"{sc['soft']['pct']}% go in", "down"))
            facts.append(_fact("Shots over 90 km/h", f"{sc['hard']['pct']}% go in", "up"))
            txt += f" Soft shots went in {sc['soft']['pct']}% of the time, hard ones {sc['hard']['pct']}%."
        _find(fs, (os_ - ts) * 3 + 10, "Finish harder: too many shots get saved", facts, txt,
              "Take the extra touch to set up power or a corner instead of a first-touch dink, and pass across when a shot is likely to be saved.")

    # 3. rotation: double commits and nobody back
    rot_us = [I.us(m).get("rotation") or {} for m in games]
    rot_them = [I.them(m).get("rotation") or {} for m in games]
    du, dt = I._avg([r.get("pct_double_up") for r in rot_us]), I._avg([r.get("pct_double_up") for r in rot_them])
    team_wc = [r for r in D["win_conditions"]["rows"] if r["src"] == "team" and r["key"] in ("pct_no_last_back", "pct_all_back", "pct_double_up", "pct_double_back")]
    if (du is not None and dt is not None and du > dt + 4) or any(abs(r["diff"]) >= 30 for r in team_wc):
        facts = []
        if du is not None and dt is not None:
            facts.append(_fact("Two+ of you in their third", f"{du:.0f}% of the time (them {dt:.0f}%)", "down" if du > dt else ""))
        for r in team_wc[:2]:
            lab, val, _k = _wc_text(r)
            facts.append(_fact(lab, val, "up"))
        _find(fs, max([abs(r["diff"]) for r in team_wc] + [(du - dt) * 3 if du and dt else 0]) * 0.8,
              "Never leave nobody back", facts,
              "The games that went wrong were the ones where too many of you got caught in front of the ball.",
              "One rule for the third player: if two are already going for the ball in their half, hold at midfield or back post, and say \"I'm back\".")

    # 4. possession
    bf = _factor(D, "ball hits")
    if bf:
        rows = {r["label"]: r for r in bf["rows"]}
        a, b = rows.get("We had more"), rows.get("They had more")
        if a and b and a["games"] >= 3 and b["games"] >= 3 and (a["win_rate"] or 0) - (b["win_rate"] or 0) >= 30:
            _find(fs, ((a["win_rate"] or 0) - (b["win_rate"] or 0)) * 0.6, "Win the possession battle",
                  [_fact("More ball hits than them", f"{_rec(a)} ({_wr(a)})", "up"), _fact("Fewer ball hits", f"{_rec(b)} ({_wr(b)})", "down")],
                  "When your team had more of the ball you won most games; when they did, you mostly lost.",
                  "Fewer speculative long hits and aerials, more controlled touches and short passes to keep the ball.")

    # 5. lobby strength and losing runs
    tiers = {t["tier"]: t for t in D["difficulty"].get("tiers") or []}
    tough = tiers.get("Tough")
    streaks = I._streaks([m["result"] for m in games])
    if tough and tough["games"] >= 3 and (tough["win_rate"] or 0) <= 35:
        facts = [_fact(f"{k} lobbies", f"{t['wins']}–{t['losses']}", "up" if k == "Easy" else "down" if k == "Tough" else "")
                 for k, t in tiers.items() if t["games"]]
        if streaks.get("L", 0) >= 4:
            facts.append(_fact("Longest losing run", f"{streaks['L']} games", "down"))
        _find(fs, (60 - (tough["win_rate"] or 0)) * 0.7, "Handle tough lobbies and losing runs", facts,
              "Most of the losses came against strong opponents.",
              "After three losses in a row, take a five-minute break and watch one goal you conceded. Play tough lobbies safer: score first and keep someone back.")

    # 6. the weakest minute
    bm = [x for x in c.get("by_minute") or [] if x["minute"] != "OT"]
    if bm:
        worst = min(bm, key=lambda x: x["diff"])
        if worst["diff"] <= -3:
            _find(fs, -worst["diff"] * 5, f"Tighten up in the {worst['label']} minute",
                  [_fact(worst["label"], f"{worst['for']}–{worst['against']} ({worst['diff']:+d})", "down")],
                  "This stretch of the clock cost you the most goals.",
                  "Take fewer risks in that minute; the third player stays back until your team has the ball.")

    # 7. when goals happen: off the kickoff, right after a goal
    gt = {w["key"]: w for w in (D.get("goal_timing") or {}).get("windows", [])}
    for key, title, txt, tip in (
            ("kickoff", "Stop conceding off the kickoff", "Goals within 10 s of a kickoff went against you.",
             "The player not going for the ball covers back post, and whoever takes the kickoff goes for a safe 50/50, not a pinch past the opponent."),
            ("after_for", "Reset after you score", "You conceded soon after scoring.",
             "After a goal, everyone is back in the same kickoff spots: go through the routine, don't relax on the restart."),
            ("after_against", "Don't concede twice in a row", "One goal against was often followed by another within 30 s.",
             "After conceding, play the next kickoff safe and keep two players back until you have the ball.")):
        w = gt.get(key)
        if w and w["against"] >= 2 and w["against"] - w["for"] >= 2:
            _find(fs, (w["against"] - w["for"]) * 6, title, [_fact(w["label"], f"{w['for']} for – {w['against']} against", "down")], txt, tip)

    # 9. how the goals against happened (from the goal causes)
    from . import diagnosis as DGN
    gb = DGN.goals_breakdown([m for m in games if m.get("replay")], None)
    fix = [c for c in gb["causes"] if c["key"] != "other" and c["goals"] >= 3 and (c["pct"] or 0) >= 18]
    if fix:
        c = max(fix, key=lambda c: c["goals"])
        _find(fs, (c["pct"] or 0) * 1.6, CAUSE_TITLE.get(c["key"], c["label"]),
              [_fact("Goals against", f"{c['goals']} of {gb['against']} ({c['pct']:.0f}%)", "down")] +
              [_fact(p["label"], f"involved in {p['goals']}", "") for p in gb["involved"][:3] if p["goals"]],
              f"\"{c['label']}\" was the most common reason a goal went in against you.", CAUSE_TIP.get(c["key"], ""))

    # 8. fading late in the session
    if n >= 6 and (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() < 20 * 3600:  # one session, not a week
        third = max(2, n // 3)
        a_ = sum(m["result"] == "W" for m in games[:third]) / third
        b_ = sum(m["result"] == "W" for m in games[-third:]) / third
        if a_ - b_ >= 0.25:
            _find(fs, (a_ - b_) * 60, "Stop earlier, or take a break",
                  [_fact(f"First {third} games", f"{round(a_ * 100)}% wins", "up"), _fact(f"Last {third} games", f"{round(b_ * 100)}% wins", "down")],
                  "Results dropped off as the session went on.",
                  "Take a 10-minute break around the middle of long sessions, or end the session when you notice slow rotations and missed touches.")

    # what went well
    late = [x for x in bm if x["minute"] in (4, 5)]
    ld = sum(x["diff"] for x in late)
    if ld > 0:
        good.append(f"Strong finishes: {ld:+d} goal difference in the last two minutes.")
    if c.get("comebacks"):
        good.append(f"Came back from behind to win {c['comebacks']} time{'s' if c['comebacks'] != 1 else ''}.")
    if streaks.get("W", 0) >= 3:
        good.append(f"Best winning run: {streaks['W']} in a row.")
    if us_sh and them_sh and us_sh > them_sh:
        good.append(f"Created more chances than the other team ({us_sh:.1f} shots a game vs {them_sh:.1f}).")
    easy = tiers.get("Easy")
    if easy and easy["games"] >= 3 and (easy["win_rate"] or 0) >= 75:
        good.append(f"Took care of easier lobbies: {easy['wins']}–{easy['losses']}.")
    third = max(1, n // 3)
    lt = games[-third:]
    lw = sum(m["result"] == "W" for m in lt)
    if lw / len(lt) > wins / n + 0.1:
        good.append(f"Finished the session strongly: {lw}–{len(lt) - lw} in the last {len(lt)} games.")

    wc_all = [dict(zip(("label", "value", "kind"), _wc_text(r))) for r in D["win_conditions"]["rows"][:6]]
    return {"findings": _rank(fs), "went_well": good, "win_conditions": wc_all,
            "by_minute": c.get("by_minute") or [], "tiers": D["difficulty"].get("tiers") or [],
            "clutch": {k: c[k] for k in ("close", "overtime", "scored_first", "conceded_first")}}


CAUSE_TITLE = {"kickoff": "Stop conceding off the kickoff", "demo": "Protect the last defender from demos",
               "double_commit": "Stop double committing", "giveaway": "Stop giving the ball away in your half",
               "fifty": "Win (or avoid) the dangerous 50/50s", "rebound": "Deal with rebounds", "nobody_back": "Always keep someone back",
               "low_boost": "Don't defend on an empty tank"}
CAUSE_TIP = {
    "kickoff": "Agree who goes for the kickoff and who covers back post, and keep the third player back until the kickoff is decided.",
    "demo": "When you're last back, stay away from the walls and keep moving; a teammate can bump the attacker coming for you.",
    "double_commit": "Call it: whoever is closer and facing the ball goes, the other holds and covers. If you're not sure you're first, you're second.",
    "giveaway": "In your own half, clear to the side walls or corners instead of the middle, and don't dribble under pressure.",
    "fifty": "Only challenge a 50/50 if a teammate is behind you; otherwise shadow and wait for them to make a mistake.",
    "rebound": "After a save or clear, the next player rotates to the back post quickly instead of following the ball.",
    "nobody_back": "The player furthest from the ball always stays goal-side; never all three in front of the ball.",
    "low_boost": "Take small pads on the way back and don't go for a challenge with under 20 boost when you're last back.",
}


# ---------------------------------------------------------------- one player
PLAYER_KEYS = ["score", "goals", "assists", "saves", "shots", "demos", "shooting_pct", "goal_part", "bpm", "pct_zero", "pct_full",
               "overfill", "pct_supersonic", "pct_behind_ball", "pct_last_back", "pct_first_forward", "avg_dist_ball",
               "aerial_touches", "kickoffs", "kickoff_wins", "demoed", "avg_shot_speed", "fifty_win_pct", "passes", "pct_wall",
               "double_commits", "rotation_cuts", "pct_held_back", "giveaways", "recovery_s", "touch_share", "boost_def", "pct_whiff"]


# advice for a stat that's below the lobby
LOBBY_TIPS = {
    "double_commits": "If a teammate is already going for the ball, hold back and cover instead; when in doubt, you're second.",
    "pct_whiff": "Slow down before the touch: arrive a moment early and on the ground, and jump later rather than flipping at a ball that's still moving away.",
    "rotation_cuts": "If a teammate is already going for the ball, let them have it and fill the space behind; only take it if they call it or clearly miss.",
    "pct_held_back": "When you see a teammate already driving at the ball, peel off: go back post or to midfield for the next touch instead of joining in.",
    "giveaways": "In your own half, play the ball to the side walls or corners and don't dribble under pressure.",
    "recovery_s": "When you're caught in front of the ball, turn and boost back to the far post straight away instead of turning at the ball.",
    "boost_def": "Pick up small pads on the way back so you have boost when the ball comes into your third.",
    "fifty_win_pct": "In a 50/50, flip into the ball with your car's nose low and aim at their side wall or corner instead of straight at the opponent.",
    "passes": "Look up before you hit: a soft touch towards a teammate or the back post keeps the ball, a hard clear usually gives it away.",
    "goals": "Get into scoring spots more: follow up your teammates' shots and wait at the back post for rebounds.",
    "shots": "Get into scoring spots more: follow up your teammates' shots and wait at the back post for rebounds.",
    "shooting_pct": "Set up power or a corner with an extra touch, and pass across when the shot is likely to be saved.",
    "avg_shot_speed": "Hit through the ball with a flip or a powershot instead of pushing it.",
    "saves": "Hold a better defensive spot: goal-side, in line with the ball and your net, and ready before the shot comes.",
    "clears": "Clear to the corners and the side walls instead of back into the middle.",
    "ball_hits": "Get more involved: take the 50/50s when you're closest instead of waiting.",
    "pct_possession": "Get more involved: take the 50/50s when you're closest instead of waiting.",
    "score": "Get more involved: take the 50/50s when you're closest instead of waiting.",
    "assists": "Look for the pass: centre from the corners to the middle instead of shooting from tight angles.",
    "goal_part": "Look for the pass: centre from the corners to the middle instead of shooting from tight angles.",
    "centres": "Look for the pass: centre from the corners to the middle instead of shooting from tight angles.",
    "bpm": "Collect more boost: route your rotation over small pads.",
    "kickoff_wins": "Practise one fast kickoff (speed flip or diagonal flip) until you win it most of the time.",
    "aerial_touches": "Practise aerial challenges in free play or training packs.",
    "demoed": "Check behind you when slow or on the ground near opponents.",
}
LOBBY_KEYS = ["goals", "shots", "shooting_pct", "saves", "assists", "score", "ball_hits", "pct_possession", "avg_shot_speed",
              "goal_part", "centres", "clears", "bpm", "kickoff_wins", "aerial_touches", "fifty_win_pct", "passes",
              "double_commits", "rotation_cuts", "pct_held_back", "giveaways", "recovery_s", "boost_def", "pct_whiff"]
TEAM_KEYS = ["score", "goals", "assists", "saves", "shots", "ball_hits", "demos", "centres", "clears", "aerial_touches", "passes"]


def _player(key: str, label: str, start: str, end: str, tables: dict, labels: dict) -> dict:
    T = tables[key]
    A = {r["key"]: r for r in I.stat_table(who=key)["rows"]}
    D = I.dashboard(who=key, start=start, end=end)
    g, *_ = I.load(who=key, start=start, end=end)
    sm = I.shot_map(g)
    fs, good = [], []
    you = "You" if key == "me" else label

    def ch(k):  # session vs this player's usual
        t, a = T.get(k), A.get(k)
        if not t or not a or t["you"] is None or not a["you"] or a["games"] < t["games"] + 5:
            return None
        return (t["you"] - a["you"]) / abs(a["you"])

    def pc(k):
        return (T.get(k) or {}).get("pct")

    # finishing
    sp = T.get("shooting_pct")
    if sp and sp["you"] is not None and sp["games"] >= 5:
        c_ = ch("shooting_pct")
        mine = [x for x in sm["shots"] if x["mine"]]
        saved = sum(x.get("outcome") == "saved" for x in mine)
        if (c_ is not None and c_ <= -0.12) or (sp["good"] is not None and sp["good"] <= 35):
            sc = _speed_conv(mine)
            facts = [_fact("Shooting", f"{_v('shooting_pct', sp['you'])}" + (f" (usually {_v('shooting_pct', A['shooting_pct']['you'])})" if c_ is not None else ""), "down")]
            if mine:
                facts.append(_fact("Shots saved", f"{saved} of {len(mine)}"))
            ss = T.get("avg_shot_speed")
            if ss and ss["you"] is not None:
                facts.append(_fact("Avg shot speed", f"{ss['you']:.0f} km/h (lobby {ss['lobby']:.0f})", "down" if ss["lobby"] and ss["you"] < ss["lobby"] else ""))
            txt = "Too many of the shots were saved."
            if sc["soft"]["shots"] >= 5 and sc["hard"]["shots"] >= 5 and sc["soft"]["pct"] is not None:
                txt += f" Shots under 70 km/h went in {sc['soft']['pct']}% of the time; over 90 km/h, {sc['hard']['pct']}%."
            _find(fs, 30 + abs((c_ or 0)) * 60, "Finish harder", facts, txt,
                  "Set up power or a corner with an extra touch, and pass across when the shot is likely to be saved.")

    # role: anchor or always forward
    lb, ff = T.get("pct_last_back"), T.get("pct_first_forward")
    wc_me = [r for r in D["win_conditions"]["rows"] if r["src"] == "me"]
    if lb and ff and lb["pct"] is not None and ff["pct"] is not None:
        if lb["pct"] >= 80 and ff["pct"] <= 30:
            facts = [_fact("Last back", f"{_v('pct_last_back', lb['you'])} of the time ({_ord(lb['pct'])} pct)"),
                     _fact("Most forward", f"{_v('pct_first_forward', ff['you'])} ({_ord(ff['pct'])} pct)", "down")]
            gc = ch("goals")
            if gc is not None and gc <= -0.15:
                facts.append(_fact("Goals", f"{T['goals']['you']:.2f} a game ({gc * 100:+.0f}%)", "down"))
            w = next((r for r in wc_me if r["key"] == "pct_last_back"), None)
            if w:
                lab, val = _wc_bad(w)
                facts.append(_fact(lab, val, "down"))
            _find(fs, 40 + (abs(w["diff"]) * 0.5 if w else 0), "Share the last-player job", facts,
                  f"{you} played the anchor most of the time: furthest back and least often first to the ball.",
                  "Rotate through instead of parking at the back: when a teammate is back post, push up and become the second attacker.")
        elif ff["pct"] >= 80 and lb["pct"] <= 25:
            facts = [_fact("Most forward", f"{_v('pct_first_forward', ff['you'])} of the time ({_ord(ff['pct'])} pct)"),
                     _fact("Last back", f"{_v('pct_last_back', lb['you'])} ({_ord(lb['pct'])} pct)", "down")]
            _find(fs, 38, "Rotate back after you challenge", facts,
                  f"{you} spent a lot of time furthest up the field and rarely took the last-player spot.",
                  "After a challenge or shot, rotate out wide and back past your teammates instead of following the ball.")

    # boost
    full, waste, zero = T.get("pct_full"), T.get("overfill"), T.get("pct_zero")
    if full and full["pct"] is not None and full["pct"] >= 75 and waste and waste["you"] is not None:
        facts = [_fact("At 100 boost", f"{_v('pct_full', full['you'])} of the time ({_ord(full['pct'])} pct)"),
                 _fact("Boost wasted", f"{waste['you']:.0f} a game (avg teammate {waste['mates']:.0f})" if waste["mates"] else f"{waste['you']:.0f} a game", "down")]
        w = next((r for r in wc_me if r["key"] == "pct_full"), None)
        if w:
            lab, val = _wc_bad(w)
            facts.append(_fact(lab, val, "down"))
        _find(fs, 30 + (abs(w["diff"]) * 0.5 if w else 0), "Spend your boost instead of sitting on it", facts,
              f"{you} held a full tank a lot and wasted boost by taking pads while already full.",
              "Skip big pads above 70 boost (leave them for teammates), take small pads on rotation, and use a full tank to challenge or pressure.")
    elif zero and zero["good"] is not None and zero["good"] <= 30:
        facts = [_fact("At 0 boost", f"{_v('pct_zero', zero['you'])} of the time", "down")]
        ab = T.get("avg_boost")
        if ab and ab["you"] is not None:
            facts.append(_fact("Average boost", f"{ab['you']:.0f} (avg teammate {ab['mates']:.0f})" if ab["mates"] else f"{ab['you']:.0f}"))
        _find(fs, 32, "Stop running out of boost", facts, f"{you} spent a lot of time on an empty tank, which makes every challenge harder.",
              "Grab small pads on the way back and on rotation, and save some boost for the next touch instead of going all in.")

    # getting demoed
    dm = T.get("demoed")
    if dm and dm["good"] is not None and dm["good"] <= 25 and dm["you"] is not None:
        _find(fs, 22, "Get demoed less", [_fact("Demoed", f"{dm['you']:.2f} a game (lobby {dm['lobby']:.2f})", "down")],
              f"{you} got demolished more than most players in your lobbies.",
              "Check behind you when slow or on the ground near opponents, and avoid driving straight at a supersonic opponent.")

    # what went with winning (this player's own stats)
    tops = [r for r in wc_me if abs(r["diff"]) >= 25 and r["key"] not in ("pct_last_back", "pct_full")][:3]
    if tops:
        facts = [_fact(*_wc_text(r)) for r in tops]
        _find(fs, max(abs(r["diff"]) for r in tops) * 0.45, f"What went with winning for {you if key != 'me' else 'you'}", facts,
              "These were the clearest differences between this player's wins and losses this session.",
              "Lean towards the winning side of these in your next session.")

    # drops against their usual
    drops = [(k, ch(k)) for k in ("goals", "assists", "saves", "shots", "score")]
    drops = [(k, c_) for k, c_ in drops if c_ is not None and c_ <= -0.2]
    if drops:
        facts = [_fact(I._SL.get(k, k), f"{T[k]['you']:.2f} a game ({c_ * 100:+.0f}% vs usual)", "down") for k, c_ in drops]
        _find(fs, 18 + 25 * max(-c_ for _k, c_ in drops), "Down on your usual", facts,
              "Compared with this player's other games.", "Look at one or two losses from this session and spot where these chances went.")

    covered = {k for f in fs for k in ("shooting_pct", "pct_zero", "demoed") if f["title"].startswith(("Finish", "Stop running", "Get demoed"))}
    covered |= {k for k, _c in drops}  # already in "Down on your usual"

    # below the lobby in the things that win games
    weak = sorted((T[k] for k in LOBBY_KEYS if k in T and k not in covered and T[k]["good"] is not None
                   and T[k]["good"] <= 40 and T[k]["games"] >= 5), key=lambda r: r["good"])[:3]
    if weak:
        facts = [_fact(r["label"], f"{_v(r['key'], r['you'])} (lobby {_v(r['key'], r['lobby'])}, {_ord(r['pct'])} pct)", "down") for r in weak]
        _find(fs, 20 + (40 - weak[0]["good"]) * 0.8, f"Below the lobby: {', '.join(r['label'].lower() for r in weak)}", facts,
              f"Compared with every player in these lobbies, {('you were' if key == 'me' else label + ' was')} in the bottom part for these.",
              LOBBY_TIPS.get(weak[0]["key"], "Pick one of these to focus on next session."))

    # lowest on the team
    others = [k for k in tables if k != key]
    if others:
        low = []
        for k in TEAM_KEYS:
            mine_v = (T.get(k) or {}).get("you")
            vals = [(tables[o].get(k) or {}).get("you") for o in others]
            vals = [v for v in vals if v is not None]
            if mine_v is None or not vals:
                continue
            best_other = min(vals)
            avg_other = sum(vals) / len(vals)
            if mine_v < best_other and avg_other and (avg_other - mine_v) / abs(avg_other) >= 0.12:
                low.append((k, mine_v, avg_other, (avg_other - mine_v) / abs(avg_other)))
        low.sort(key=lambda x: -x[3])
        if low:
            facts = [_fact(I._SL.get(k, (T.get(k) or {}).get("label", k)), f"{_v(k, v)} a game (teammates {_v(k, a)})", "down") for k, v, a, _r in low[:3]]
            _find(fs, 16 + 30 * low[0][3], f"Lowest on the team in {', '.join(I._SL.get(k, (T.get(k) or {}).get('label', k)).lower() for k, *_ in low[:3])}",
                  facts, "Compared with the other players on the team in the same games.",
                  LOBBY_TIPS.get(low[0][0], "Pick one of these to focus on next session."))

    # kickoff win rate
    def ko(tb):
        a, b = (tb.get("kickoff_wins") or {}).get("you"), (tb.get("kickoffs") or {}).get("you")
        return (a / b) if a is not None and b and b >= 0.8 else None
    mine_ko = ko(T)
    team_ko = [x for x in (ko(tables[o]) for o in others) if x is not None]
    if mine_ko is not None and mine_ko < 0.5 and (not team_ko or mine_ko < min(team_ko)):
        facts = [_fact("Kickoffs won", f"{round(100 * mine_ko)}% of the ones taken", "down"),
                 _fact("Kickoffs taken", f"{T['kickoffs']['you']:.1f} a game")]
        for o in others:
            x = ko(tables[o])
            if x is not None:
                facts.append(_fact(labels.get(o, o), f"{round(100 * x)}% won"))
        _find(fs, 22 + (0.5 - mine_ko) * 60, "Win more of your kickoffs", facts,
              "Kickoffs won count as one of the clearest signs of which team gets the first goal.",
              LOBBY_TIPS["kickoff_wins"] + " Or let the teammate who wins them most take them.")

    # went well: high percentiles and rises against the usual
    strong = sorted((r for r in T.values() if r["good"] is not None and r["good"] >= 65 and r["games"] >= 5 and not r["neutral"]),
                    key=lambda r: -r["good"])[:4]
    for r in strong:
        per = "" if r["key"].startswith(("pct_", "max_", "avg_")) or r["key"] in ("shooting_pct", "goal_part") else " a game"
        rank = (f"lower than {100 - r['pct']}% of performances in your lobbies (lower is better)" if r["lower_better"]
                else f"{_ord(r['pct'])} percentile in your lobbies")
        good.append(f"{r['label']}: {_v(r['key'], r['you'])}{per}, {rank}.")
    for k in ("goals", "assists", "saves", "shots", "demos"):
        c_ = ch(k)
        if c_ is not None and c_ >= 0.15:
            good.append(f"{I._SL.get(k, k)} up {c_ * 100:.0f}% on the usual ({T[k]['you']:.2f} a game).")
    # best on the team
    if others:
        tops_team = []
        for k in TEAM_KEYS:
            mine_v = (T.get(k) or {}).get("you")
            vals = [v for v in ((tables[o].get(k) or {}).get("you") for o in others) if v is not None]
            if mine_v is not None and vals and mine_v > max(vals) * 1.1:
                tops_team.append(f"{I._SL.get(k, (T.get(k) or {}).get('label', k))} ({_v(k, mine_v)} a game)")
        if tops_team:
            good.append(f"Best on the team in {', '.join(tops_team[:3])}.")
    mvps = sum(m.get("mvp") == m["my"]["name"] for m in g)
    if mvps:
        good.append(f"MVP {mvps} time{'s' if mvps != 1 else ''}.")
    if g:
        bg = max(g, key=lambda m: m["my"]["stats"].get("score") or 0)
        good.append(f"Best game: {bg['my']['stats'].get('score')} points ({bg['result']} {I.us(bg).get('score')}–{I.them(bg).get('score')}).")

    table = [{"key": k, "label": T[k]["label"], "session": _v(k, T[k]["you"]), "usual": _v(k, A[k]["you"]) if k in A and A[k]["games"] > T[k]["games"] else "–",
              "mates": _v(k, T[k]["mates"]), "lobby": _v(k, T[k]["lobby"]), "pct": T[k]["pct"], "good": T[k]["good"]}
             for k in PLAYER_KEYS if k in T]
    return {"key": key, "label": label, "games": len(g), "findings": _rank(fs, 6), "went_well": good[:8], "table": table}


# ---------------------------------------------------------------- the whole report
def session_report(start: str, end: str) -> dict:
    games, me, cfg, aliases = I.load(start=start, end=end)
    if not games:
        return {"games": 0}
    disp = stats.my_display(me, aliases)
    D = I.dashboard(start=start, end=end)
    sm = I.shot_map(games)
    # team members: you plus whoever played most of the session with you
    cnt, lab = Counter(), {}
    for m in games:
        for p in I.us(m)["players"]:
            k = "me" if p is m["my"] else p.get("key")
            if k:
                cnt[k] += 1
                lab[k] = (disp or "You") if k == "me" else (p.get("alias") or p["name"])
    members = ["me"] + [k for k, c in cnt.most_common() if k != "me" and c >= max(3, 0.4 * len(games))]
    tables = {k: {r["key"]: r for r in I.stat_table(who=k, start=start, end=end)["rows"]} for k in members}
    # how much history the grades had in this mode before the session
    allg, *_ = I.load()
    modes = Counter(m["playlist"] for m in games)
    before = Counter(m["playlist"] for m in allg if m["played_at"] < start)
    thin = [pl for pl in modes if before.get(pl, 0) < 10]
    t0, t1 = datetime.fromisoformat(games[0]["played_at"]), datetime.fromisoformat(games[-1]["played_at"])
    wins = sum(m["result"] == "W" for m in games)
    return {
        "games": len(games), "wins": wins, "losses": len(games) - wins, "start": start, "end": end,
        "minutes": round((t1 - t0).total_seconds() / 60), "modes": dict(modes),
        "goals_for": sum(I.us(m).get("score") or 0 for m in games), "goals_against": sum(I.them(m).get("score") or 0 for m in games),
        "results": [{"id": m["id"], "r": m["result"], "for": I.us(m).get("score"), "against": I.them(m).get("score")} for m in games],
        "difficulty": D["difficulty"].get("avg"), "thin_modes": thin,
        "team": _team(games, D, sm, start, end),
        "players": [_player(k, lab[k], start, end, tables, lab) for k in members],
    }
