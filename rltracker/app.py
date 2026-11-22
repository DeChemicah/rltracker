"""Local web app: http://127.0.0.1:5177"""
from __future__ import annotations

import os

from flask import Flask, abort, jsonify, request, send_from_directory

from . import export, insights, replays, stats, store
from .scanner import Scanner

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
scanner = Scanner()


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/summary")
def api_summary():
    return jsonify(stats.summary(
        size=request.args.get("size", "all"),
        ranked=request.args.get("ranked", "all"),
        last=int(request.args.get("last", 0) or 0),
        who=request.args.get("who", "me") or "me",
        start=request.args.get("start", ""), end=request.args.get("end", ""),
        tag=request.args.get("tag", ""),
    ))


@app.put("/api/match/<int:mid>/notes")
def api_match_notes(mid):
    if not store.get_match(mid):
        abort(404)
    b = request.get_json(force=True) or {}
    return jsonify(store.set_notes(mid, note=b.get("note"), tags=b.get("tags"), add=b.get("add"), remove=b.get("remove")))


@app.post("/api/notes/bulk")
def api_notes_bulk():
    """Add or remove a tag on several games at once (e.g. a whole session)."""
    b = request.get_json(force=True) or {}
    ids = [int(x) for x in (b.get("ids") or [])][:500]
    for mid in ids:
        store.set_notes(mid, add=b.get("add"), remove=b.get("remove"))
    return jsonify(ok=True, games=len(ids))


@app.get("/api/tags")
def api_tags():
    return jsonify(store.all_tags())


@app.get("/api/matches")
def api_matches():
    return jsonify(stats.match_list())


@app.get("/api/match/<int:mid>")
def api_match(mid):
    m = stats.match_detail(mid)
    if not m:
        abort(404)
    return jsonify(m)


@app.post("/api/match/<int:mid>/hide")
def api_match_hide(mid):
    store.set_hidden(mid, bool(request.get_json(force=True).get("hidden", True)))
    return jsonify(ok=True)


@app.get("/api/team")
def api_team():
    args = dict(lineup=request.args.get("lineup", ""), size=request.args.get("size", "all"),
                ranked=request.args.get("ranked", "all"))
    return jsonify({**stats.team(**args), **insights.team_extras(**args)})


def _cutoff() -> int:
    q = request.args.get("cutoff")
    return max(5, int(q)) if q else int(store.load_config().get("session_cutoff_min") or 45)


@app.get("/api/sessions")
def api_sessions():
    return jsonify(insights.sessions(_cutoff()))


@app.get("/api/session_report")
def api_session_report():
    from . import debrief
    return jsonify(debrief.session_report(request.args.get("start", ""), request.args.get("end", "")))


@app.get("/api/session_shots")
def api_session_shots():
    return jsonify(insights.session_shots(request.args.get("start", ""), request.args.get("end", "")))


@app.get("/api/live")
def api_live():
    return jsonify({**insights.live(_cutoff()), "scanner": {k: scanner.state.get(k) for k in ("running", "last_new")}})


@app.get("/live")
def live_page():
    return send_from_directory(STATIC, "live.html")


def _dash_args():
    q = request.args
    return dict(size=q.get("size", "all"), ranked=q.get("ranked", "all"), last=int(q.get("last", 0) or 0),
                who=q.get("who", "me") or "me", start=q.get("start", ""), end=q.get("end", ""), tag=q.get("tag", ""))


@app.get("/api/stat_help")
def api_stat_help():
    """{stat label (lower case): one-line explanation} for tooltips."""
    out = {}
    for k, lab, _g, _s in insights._stat_catalogue():
        h = insights.STAT_HELP.get(k)
        if h:
            out[lab.lower()] = h
            out[k] = h
    return jsonify(out)


@app.get("/api/stat_table")
def api_stat_table():
    return jsonify(insights.stat_table(**_dash_args()))


@app.get("/api/stat_detail")
def api_stat_detail():
    return jsonify(insights.stat_detail(request.args.get("key", ""), **_dash_args()))


@app.get("/api/insights")
def api_insights():
    return jsonify(insights.dashboard(size=request.args.get("size", "all"), ranked=request.args.get("ranked", "all"),
                                      last=int(request.args.get("last", 0) or 0), who=request.args.get("who", "me") or "me",
                                      start=request.args.get("start", ""), end=request.args.get("end", ""),
                                      tag=request.args.get("tag", "")))


@app.get("/api/replay_meta")
def api_replay_meta():
    from . import replay_stats
    return jsonify(replay_stats.meta())


@app.get("/api/compare_players")
def api_compare_players():
    q = request.args
    return jsonify(insights.compare_players(q.get("a", "me"), q.get("b", ""), last=int(q.get("last", 0) or 0),
                                            frm=q.get("from", ""), to=q.get("to", ""), start=q.get("start", ""), end=q.get("end", ""),
                                            c=q.get("c", "")))


@app.get("/api/player")
def api_player():
    p = insights.profile(request.args.get("key", ""))
    if not p:
        abort(404)
    return jsonify(p)


@app.get("/api/impact")
def api_impact():
    r = insights.teammate_impact(request.args.get("key", "me"))
    if not r:
        abort(404)
    return jsonify(r)


@app.get("/api/replays")
def api_replays():
    return jsonify(replays.summary())


@app.post("/api/match/<int:mid>/keep_replay")
def api_keep_replay(mid):
    ok = replays.set_keep(mid, bool(request.get_json(force=True).get("keep", True)))
    return jsonify(ok=ok, replay=replays.match_replay_info(mid))


@app.get("/api/export.<fmt>")
def api_export(fmt):
    if fmt == "csv":
        body = export.to_csv()
        mime = "text/csv"
    elif fmt == "json":
        body = export.to_json()
        mime = "application/json"
    else:
        abort(404)
    from datetime import date
    return app.response_class(body, mimetype=mime, headers={
        "Content-Disposition": f'attachment; filename="rl-tracker-{date.today():%Y-%m-%d}.{fmt}"'})


@app.get("/api/records")
def api_records():
    return jsonify(stats.records(player=request.args.get("player", "universal"),
                                 start=request.args.get("start", ""), end=request.args.get("end", "")))


@app.get("/api/history")
def api_history():
    return jsonify(stats.history())


@app.get("/api/settings")
def api_settings_get():
    cfg = store.load_config()
    return jsonify({
        **cfg,
        "auto_me": stats.detect_me(store.all_matches(include_hidden=False), {"player_names": []}),
    })


@app.put("/api/settings")
def api_settings_put():
    body = request.get_json(force=True)
    cfg = store.load_config()
    if "player_names" in body:
        cfg["player_names"] = [n.strip() for n in body["player_names"] if n.strip()]
    if "aliases" in body:
        cfg["aliases"] = _clean_aliases(body["aliases"])
    if "poll_seconds" in body:
        cfg["poll_seconds"] = max(2, int(body["poll_seconds"]))
    if "dash_layout" in body:  # Dashboard cards: order and hidden ones
        dl = body["dash_layout"] or {}
        cfg["dash_layout"] = {"order": [str(x)[:40] for x in (dl.get("order") or [])][:60],
                              "hidden": [str(x)[:40] for x in (dl.get("hidden") or [])][:60]}
    if "viewing" in body:  # who appears in the "Viewing" picker and search (empty = everyone)
        cfg["viewing"] = [str(x)[:80] for x in (body["viewing"] or [])][:60]
    if "update_repo" in body:  # where updates come from ("owner/name" on GitHub)
        cfg["update_repo"] = str(body["update_repo"] or "").strip().strip("/")[:100]
    if "pins" in body:  # stats starred on the Stats page, shown on Home
        cfg["pins"] = [str(x)[:40] for x in (body["pins"] or [])][:24]
    if "replay_retention" in body:
        cfg["replay_retention"] = bool(body["replay_retention"])
    if "session_cutoff_min" in body:
        cfg["session_cutoff_min"] = max(5, min(600, int(body["session_cutoff_min"])))
    store.save_config(cfg)
    scanner.poke()
    return jsonify(ok=True)


def _clean_aliases(items) -> list[dict]:
    out = {}
    for a in items or []:
        name, alias = (a.get("name") or "").strip(), (a.get("alias") or "").strip()
        if name and alias:
            out[stats.norm_name(name) or name.lower()] = {"name": name, "alias": alias}
    return list(out.values())


@app.post("/api/alias")
def api_alias():
    """Upsert one alias (empty alias removes it). Used from the match view."""
    body = request.get_json(force=True)
    name, alias = (body.get("name") or "").strip(), (body.get("alias") or "").strip()
    if not name:
        abort(400)
    cfg = store.load_config()
    keep = [a for a in cfg.get("aliases") or [] if not stats.same_player(a["name"], name)]
    if alias:
        keep.append({"name": name, "alias": alias})
    cfg["aliases"] = _clean_aliases(keep)
    store.save_config(cfg)
    return jsonify(ok=True, aliases=cfg["aliases"])


@app.get("/api/version")
def api_version():
    from . import updater
    from .version import VERSION
    return jsonify({"version": VERSION, "repo": updater.repo()})


@app.get("/api/update/check")
def api_update_check():
    from . import updater
    return jsonify(updater.check(force=request.args.get("force") == "1"))


@app.post("/api/update/apply")
def api_update_apply():
    """Only runs when the person clicks Update in Settings."""
    from . import updater
    res = updater.apply()
    if res.get("ok"):
        updater.restart_soon()
    return jsonify(res)


@app.get("/api/status")
def api_status():
    with store.conn() as c:
        replays_n = c.execute("SELECT COUNT(*) FROM replay_files WHERE status='extracted'").fetchone()[0]
    return jsonify({"running": scanner.state.get("running"), "last_new": scanner.state.get("last_new"),
                    "replays": replays_n, "folders": replays.replay_folders(), "storage": store.storage_stats()})


def main(port: int = 5177, open_browser: bool = True):
    import socket
    import webbrowser

    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:  # already running: just open it
            if open_browser:
                webbrowser.open(f"http://127.0.0.1:{port}")
            return
    store.init()
    replays.init()
    scanner.start()
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"RL Tracker running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    print("Watching:", *replays.replay_folders(), sep="\n  ")
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
