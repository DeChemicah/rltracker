"""Turn a replay into a compact, full-detail track file.

One snapshot per network frame (~30-36 per second, exactly as recorded):
ball + every player's position, rotation, velocity, spin, boost, throttle,
steer and component states, plus an event list and match metadata. Saved as
a compressed numpy archive, typically 0.5-1 MB per match.

Only reads the .replay file (through replay_io / rrrocket). Never touches the game.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np

from . import replay_io, store

VERSION = 2  # 2: the game's own shot / save moments ("shot" / "save" events)
TRACKS_DIR = os.path.join(store.DATA_DIR, "tracks")

# state bit flags per player per frame
PRESENT, JUMP, DOUBLE_JUMP, DODGE, BOOSTING, HANDBRAKE, FLIP_CAR, DEMOLISHED = 1, 2, 4, 8, 16, 32, 64, 128
COMPONENT_BIT = {"CarComponent_Jump": JUMP, "CarComponent_DoubleJump": DOUBLE_JUMP, "CarComponent_Dodge": DODGE,
                 "CarComponent_Boost": BOOSTING, "CarComponent_FlipCar": FLIP_CAR}


def _i16(v, scale=1.0):
    return int(max(-32768, min(32767, round((v or 0.0) * scale))))


def _uid(u: dict) -> str | None:
    rid = (u or {}).get("remote_id") or {}
    local = (u or {}).get("local_id") or 0  # split-screen guests share the host's account id
    for plat, val in rid.items():
        if isinstance(val, dict):  # e.g. PlayStation {name, online_id, ...}
            val = val.get("online_id") or val.get("name") or json.dumps(val, sort_keys=True)
        if str(val) not in ("", "0"):
            return f"{plat}:{val}" + (f":guest{local}" if local else "")
    return None


def extract(path: str) -> tuple[dict, dict, list]:
    """Return (meta, arrays, events) for one replay."""
    d = replay_io.load(path)
    hdr, frames, objs = d["header"], d["frames"], d["objects"]

    kind: dict[int, str] = {}          # actor id -> car / ball / pri / comp:<bit> / team0 / team1 / pickup
    car_pri: dict[int, int] = {}       # car actor -> PRI actor
    comp_car: dict[int, int] = {}      # component actor -> car actor
    pri_info: dict[int, dict] = {}     # PRI actor -> {name, uid, team_actor}
    team_num: dict[int, int] = {}      # team actor -> 0/1
    players: list[dict] = []           # stable player list (index = column in arrays)
    pkey_to_idx: dict[str, int] = {}

    def player_of_pri(pri: int) -> int | None:
        info = pri_info.get(pri)
        if not info or not info.get("name"):
            return None
        key = info.get("uid") or f"name:{info['name']}"
        if key not in pkey_to_idx:
            pkey_to_idx[key] = len(players)
            players.append({"name": info["name"], "uid": info.get("uid"), "team": None})
        idx = pkey_to_idx[key]
        if info.get("team_actor") in team_num:
            players[idx]["team"] = team_num[info["team_actor"]]
        return idx

    F = len(frames)
    MAXP = 16
    t = np.zeros(F, np.float32)
    clock = np.zeros(F, np.int16)
    ball = {"pos": np.zeros((F, 3), np.int16), "rot": np.zeros((F, 4), np.int16),
            "vel": np.zeros((F, 3), np.int16), "angvel": np.zeros((F, 3), np.int16)}
    car = {"pos": np.zeros((F, MAXP, 3), np.int16), "rot": np.zeros((F, MAXP, 4), np.int16),
           "vel": np.zeros((F, MAXP, 3), np.int16), "angvel": np.zeros((F, MAXP, 3), np.int16),
           "boost": np.zeros((F, MAXP), np.uint8), "throttle": np.full((F, MAXP), 128, np.uint8),
           "steer": np.full((F, MAXP), 128, np.uint8), "state": np.zeros((F, MAXP), np.uint8)}

    # "held" current values, copied into the arrays every frame
    cur_ball = {"pos": [0, 0, 93], "rot": [0, 0, 0, 32767], "vel": [0, 0, 0], "angvel": [0, 0, 0]}
    cur_car: dict[int, dict] = {}      # car actor -> held values
    comp_active: dict[int, bool] = {}
    events: list[dict] = []
    clock_now = 300
    demo_seen: set[str] = set()
    pickup_seen: dict[int, int] = {}  # pad actor -> last pickup counter
    pri_hist: dict[int, int] = {}      # car actor -> PRI actor, never cleared
    stat_seen: dict[tuple, int] = {}   # (PRI actor, "shot"/"save") -> highest counter seen

    def car_state(actor: int) -> dict:
        return cur_car.setdefault(actor, {"pos": [0, 0, 17], "rot": [0, 0, 0, 32767], "vel": [0, 0, 0], "angvel": [0, 0, 0],
                                          "boost": 85, "throttle": 128, "steer": 128, "handbrake": False})

    def rb_vals(rb: dict) -> dict:
        loc, rot = rb.get("location") or {}, rb.get("rotation") or {}
        lv, av = rb.get("linear_velocity") or {}, rb.get("angular_velocity") or {}
        out = {"pos": [_i16(loc.get("x")), _i16(loc.get("y")), _i16(loc.get("z"))],
               "rot": [_i16(rot.get("x"), 32767), _i16(rot.get("y"), 32767), _i16(rot.get("z"), 32767), _i16(rot.get("w"), 32767)]}
        if lv:
            out["vel"] = [_i16(lv.get("x")), _i16(lv.get("y")), _i16(lv.get("z"))]
        if av:
            out["angvel"] = [_i16(av.get("x")), _i16(av.get("y")), _i16(av.get("z"))]
        return out

    for fi, fr in enumerate(frames):
        t[fi] = fr.get("time") or 0.0
        for aid in fr.get("deleted_actors") or []:
            if kind.get(aid) == "car":
                pri = car_pri.get(aid)
                pi = player_of_pri(pri) if pri is not None else None
                if pi is not None:
                    events.append({"type": "car_removed", "frame": fi, "player": pi})
            kind.pop(aid, None)
            cur_car.pop(aid, None)
            car_pri.pop(aid, None)
            comp_car.pop(aid, None)
        for na in fr.get("new_actors") or []:
            aid, name = na["actor_id"], objs[na["object_id"]]
            if name.startswith("Archetypes.Car.Car_"):
                kind[aid] = "car"
                car_state(aid)
            elif name.startswith("Archetypes.Ball."):
                kind[aid] = "ball"
            elif name.endswith("PRI_TA"):
                kind[aid] = "pri"
                pri_info.setdefault(aid, {})
            elif name.startswith("Archetypes.CarComponents."):
                comp = name.split(".")[-1]
                kind[aid] = f"comp:{COMPONENT_BIT.get(comp, 0)}"
            elif name == "Archetypes.Teams.Team0":
                kind[aid] = "team"; team_num[aid] = 0
            elif name == "Archetypes.Teams.Team1":
                kind[aid] = "team"; team_num[aid] = 1
            elif "VehiclePickup_Boost" in name:
                kind[aid] = "pickup:" + name.split("_")[-1]
        for u in fr.get("updated_actors") or []:
            aid, attr = u["actor_id"], u["attribute"]
            aname = objs[u["object_id"]]
            k = kind.get(aid)
            if aname == "TAGame.RBActor_TA:ReplicatedRBState" and "RigidBody" in attr:
                vals = rb_vals(attr["RigidBody"])
                if k == "ball":
                    cur_ball.update(vals)
                elif k == "car":
                    car_state(aid).update(vals)
            elif aname == "Engine.Pawn:PlayerReplicationInfo":
                link = attr.get("ActiveActor") or {}
                if link.get("active") and link.get("actor", -1) >= 0:  # actor id 0 is valid
                    car_pri[aid] = link["actor"]
                    pri_hist[aid] = link["actor"]  # remembered even after the car is unlinked/removed
                else:  # car wrecked/unlinked: stop drawing it, but keep the history
                    car_pri.pop(aid, None)
            elif aname == "TAGame.CarComponent_TA:Vehicle":
                comp_car[aid] = (attr.get("ActiveActor") or {}).get("actor")
            elif aname == "TAGame.CarComponent_TA:ReplicatedActive":
                comp_active[aid] = bool((attr.get("Byte") or 0) % 2)
            elif aname.startswith("TAGame.CarComponent_Boost_TA:ReplicatedBoost"):
                rb = attr.get("ReplicatedBoost") or {}
                amt = rb.get("boost_amount", attr.get("Byte"))
                car_id = comp_car.get(aid)
                if amt is not None and car_id is not None:
                    car_state(car_id)["boost"] = int(round(amt * 100 / 255))
            elif aname == "TAGame.Vehicle_TA:ReplicatedThrottle" and k == "car":
                car_state(aid)["throttle"] = attr.get("Byte", 128)
            elif aname == "TAGame.Vehicle_TA:ReplicatedSteer" and k == "car":
                car_state(aid)["steer"] = attr.get("Byte", 128)
            elif aname == "TAGame.Vehicle_TA:bReplicatedHandbrake" and k == "car":
                car_state(aid)["handbrake"] = bool(attr.get("Boolean"))
            elif aname == "Engine.PlayerReplicationInfo:PlayerName":
                pri_info.setdefault(aid, {})["name"] = attr.get("String")
            elif aname == "Engine.PlayerReplicationInfo:UniqueId":
                pri_info.setdefault(aid, {})["uid"] = _uid(attr.get("UniqueId"))
            elif aname == "Engine.PlayerReplicationInfo:Team":
                pri_info.setdefault(aid, {})["team_actor"] = (attr.get("ActiveActor") or {}).get("actor")
            elif aname == "TAGame.GameEvent_Soccar_TA:SecondsRemaining":
                clock_now = attr.get("Int", clock_now)
            elif aname == "TAGame.VehiclePickup_TA:NewReplicatedPickupData":
                pk = attr.get("PickupNew") or {}
                car_id = pk.get("instigator")
                pri = car_pri.get(car_id)
                pi = player_of_pri(pri) if pri is not None else None
                # picked_up is a counter (1, 3, 5, …; 255 = respawned). The game re-sends the
                # current state now and then, so only a new counter value is a new pickup.
                cnt = pk.get("picked_up")
                if cnt and cnt != 255 and pi is not None and pickup_seen.get(aid) != cnt:
                    pickup_seen[aid] = cnt
                    pad = (k or "pickup:?").split(":", 1)[1]
                    events.append({"type": "pickup", "frame": fi, "player": pi, "pad": pad, "n": cnt})
            elif aname in ("TAGame.PRI_TA:MatchShots", "TAGame.PRI_TA:MatchSaves") and k == "pri":
                # the game's own shot / save counter: each increase is the moment it counted one
                # (the current value is re-sent now and then, so only an increase counts)
                cnt, typ = attr.get("Int") or 0, "shot" if aname.endswith("Shots") else "save"
                prev = stat_seen.get((aid, typ), 0)
                pi = player_of_pri(aid)
                if cnt > prev and pi is not None:
                    for n in range(prev + 1, cnt + 1):
                        events.append({"type": typ, "frame": fi, "player": pi, "n": n})
                stat_seen[(aid, typ)] = max(prev, cnt)
            elif aname == "TAGame.Ball_TA:HitTeamNum":
                events.append({"type": "ball_hit_team", "frame": fi, "team": attr.get("Byte")})
            elif aname.startswith("TAGame.Car_TA:ReplicatedDemolish"):
                dm = attr.get("DemolishExtended") or attr.get("Demolish") or attr.get("DemolishFx") or {}
                if not dm:
                    continue
                sig = json.dumps(dm, sort_keys=True)  # the same demo is re-sent several times
                if sig in demo_seen:
                    continue
                demo_seen.add(sig)

                def actor(v):
                    return v.get("actor") if isinstance(v, dict) else v
                att_pri = actor(dm.get("attacker_pri"))
                att_car, vic_car = actor(dm.get("attacker")), actor(dm.get("victim"))
                if att_pri is None or att_pri < 0:
                    att_pri = pri_hist.get(att_car)
                a_pi = player_of_pri(att_pri) if att_pri is not None else None
                v_pri = car_pri.get(vic_car, pri_hist.get(vic_car))
                v_pi = player_of_pri(v_pri) if v_pri is not None else None
                events.append({"type": "demo", "frame": fi, "attacker": a_pi, "victim": v_pi,
                               "pos": list(cur_car.get(vic_car, {}).get("pos", [0, 0, 0]))})

        # write this frame's held values
        clock[fi] = clock_now
        for key in ball:
            ball[key][fi] = cur_ball[key]
        for car_id, st in cur_car.items():
            pri = car_pri.get(car_id)
            pi = player_of_pri(pri) if pri is not None else None
            if pi is None or pi >= MAXP:
                continue
            for key in ("pos", "rot", "vel", "angvel"):
                car[key][fi, pi] = st[key]
            car["boost"][fi, pi] = st["boost"]
            car["throttle"][fi, pi] = st["throttle"]
            car["steer"][fi, pi] = st["steer"]
            bits = PRESENT | (HANDBRAKE if st["handbrake"] else 0)
            for comp_id, cid in comp_car.items():
                if cid == car_id and comp_active.get(comp_id):
                    bits |= int(kind.get(comp_id, "comp:0").split(":")[1])
            car["state"][fi, pi] = bits

    P = len(players)
    for key in car:
        car[key] = car[key][:, :P].copy()
    # line players up with the header (exact scoreboard stats + platform ids)
    for p in players:
        h = next((x for x in hdr["players"] if x["name"] == p["name"]), None)
        if h:
            p.update({"id": h["id"], "platform": h["platform"], "header_stats": h["stats"], "bot": h["bot"]})
            if p["team"] is None:
                p["team"] = h["team"]
        else:
            p["id"] = p.get("uid") or f"name:{p['name']}"
    for g in hdr["goals"]:
        events.append({"type": "goal", "frame": g["frame"], "team": g["team"],
                       "player": next((i for i, p in enumerate(players) if p["name"] == g["player"]), None)})
    events.sort(key=lambda e: e["frame"])

    meta = {"version": VERSION, "file": os.path.basename(path), "header": hdr, "players": players,
            "frames": F, "seconds": float(t[-1] - t[0]) if F else 0.0}
    arrays = {"t": t, "clock": clock, **{f"ball_{k}": v for k, v in ball.items()}, **{f"car_{k}": v for k, v in car.items()}}
    return meta, arrays, events


def save(match_key: str, meta: dict, arrays: dict, events: list) -> str:
    """Write data/tracks/<match_key>.npz and re-read it to be sure."""
    os.makedirs(TRACKS_DIR, exist_ok=True)
    path = os.path.join(TRACKS_DIR, f"{match_key}.npz")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays,
                            meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
                            events=np.frombuffer(json.dumps(events).encode(), np.uint8))
    meta2, arrays2, events2 = load_track(tmp)  # verify before it replaces anything
    assert meta2["frames"] == meta["frames"] and arrays2["car_pos"].shape == arrays["car_pos"].shape and len(events2) == len(events)
    os.replace(tmp, path)
    return path


def load_track(path: str) -> tuple[dict, dict, list]:
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k not in ("meta", "events")}
        meta = json.loads(bytes(z["meta"]).decode())
        events = json.loads(bytes(z["events"]).decode())
    return meta, arrays, events
