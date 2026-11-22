"""The only scoreboard stats the app records, in display order.

Everything comes from replays: score, goals, assists, saves and shots from the replay's
scoreboard; ball hits and demos worked out from the recording. MMR isn't tracked.
"""
TRACKED_STATS = ["score", "goals", "assists", "ball_hits", "saves", "shots", "demos"]
REPLAY_ONLY = ["ball_hits", "demos"]  # older games without a replay don't have these


def keep_tracked(columns: list[dict], allowed: list[str] | None = None) -> list[dict]:
    order = {k: i for i, k in enumerate(allowed or TRACKED_STATS)}
    return sorted((c for c in columns if c.get("key") in order), key=lambda c: order[c["key"]])
