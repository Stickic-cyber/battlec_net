#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decode a UNSW Battlecode 2026 "Snake" .replay file into structured data.

Usage:
    python decode_replay.py <file.replay> [-o out.json] [--events N]

Reads packed Cap'n Proto wire format via reader.py and interprets every event
using field offsets reconstructed from the official replay-viewer JS codegen.
"""
import json
import sys

from reader import Message, Struct, struct_list, pointer_list, _u32, _i32, _u16, SZ

DIR = ["N", "E", "S", "W"]
EVENT_NAMES = {
    0: "roundStart", 1: "turnStart", 2: "pearlCountdown", 3: "tileChange",
    4: "dragonAction", 5: "engineLog", 6: "dragonLog", 7: "dragonIndicator",
    8: "debugDraw", 9: "dragonUpdate", 10: "dragonSplit", 11: "dragonDeath",
    12: "sonarPing",
}


def _list_ptr(msg, seg_id, poff):
    """Follow a list pointer word -> (content_seg, content_off)."""
    return pointer_list(msg, seg_id, poff)["content"], pointer_list(msg, seg_id, poff)["seg"]


def list_len(msg, seg_id, ptr_off):
    """Number of elements in a fixed-size (non-composite) list at pointer off."""
    return pointer_list(msg, seg_id, ptr_off)["count"]


def list_item(msg, seg_id, ptr_off, i, width):
    """Read element i of a fixed-size list of width bytes (little endian)."""
    info = pointer_list(msg, seg_id, ptr_off)
    seg = msg.segments[info["seg"]]
    co = info["content"]
    return int.from_bytes(seg[co + i * width: co + (i + 1) * width], "little")


def list_of_points(msg, seg_id, ptr_off):
    info = pointer_list(msg, seg_id, ptr_off)
    seg = msg.segments[info["seg"]]
    first = info["first"]
    dw = info["data_words"]; pc = info["ptr_count"]
    out = []
    for i in range(info["count"]):
        base = first + i * (dw * 8 + pc * 8)
        out.append({"x": _i32(seg, base), "y": _i32(seg, base + 4)})
    return out


def _point(msg, seg_id, ptr_off):
    cs, co = msg.follow(seg_id, ptr_off)
    st = Struct(msg, cs, co, "Point")
    return {"x": st.i32(0), "y": st.i32(4)}


def decode_player_action(msg, seg_id, ptr_off):
    cs, co = msg.follow(seg_id, ptr_off)
    st = Struct(msg, cs, co, "PlayerAction")
    which = st.u16(0)
    if which == 0:
        po = st._ptroff(0)
        n = list_len(msg, st.seg, po)
        return {"kind": "move", "steps": [DIR[list_item(msg, st.seg, po, i, 2)] for i in range(n)]}
    if which == 1:
        return {"kind": "split", "childSegmentCount": st.i32(4)}
    return {"kind": "suicide"}


def decode_event(msg, seg_id, data_off):
    st = Struct(msg, seg_id, data_off, "Event")
    which = st.u16(0)
    name = EVENT_NAMES.get(which, "unknown")
    # All variants hang off the Event's single pointer (ptr0).
    rseg, roff = msg.follow(seg_id, st._ptroff(0))
    e = Struct(msg, rseg, roff, {
        0: "EventRoundStart", 1: "EventTurnStart", 2: "EventPearlCountdown",
        3: "EventTileChange", 4: "EventDragonAction", 5: "EventEngineLog",
        6: "EventDragonLog", 7: "EventDragonIndicator", 8: "EventDebugDraw",
        9: "EventDragonUpdate", 10: "EventDragonSplit", 11: "EventDragonDeath",
        12: "EventSonarPing",
    }.get(which, "EventRoundStart"))

    if which == 0:
        return {"type": "roundStart", "round": e.i32(0)}
    if which == 1:
        return {"type": "turnStart", "id": e.i32(0)}
    if which == 2:
        return {"type": "pearlCountdown", "tile": _point(msg, e.seg, e._ptroff(0)),
                "countdown": e.i32(4)}
    if which == 3:
        return {"type": "tileChange", "tile": _point(msg, e.seg, e._ptroff(0)),
                "hasPearl": e.bit(0)}
    if which == 4:
        d = {"type": "dragonAction", "id": e.i32(0), "tle": e.bit(32)}
        if not e.null_ptr(0):
            d["action"] = decode_player_action(msg, e.seg, e._ptroff(0))
        if not e.null_ptr(1):
            ins = Struct(msg, *msg.follow(e.seg, e._ptroff(1)), "InstructionUsage")
            d["instructions"] = {"count": ins.u64(0), "exceeded": ins.bit(64)}
        return d
    if which == 5:
        return {"type": "engineLog", "id": e.i32(0), "text": e.text(0)}
    if which == 6:
        return {"type": "dragonLog", "id": e.i32(0), "text": e.text(0)}
    if which == 7:
        return {"type": "dragonIndicator", "id": e.i32(0), "text": e.text(0)}
    if which == 8:
        return {"type": "debugDraw", "id": e.i32(0)}
    if which == 9:
        return {"type": "dragonUpdate", "id": e.i32(0), "facing": DIR[e.u16(4)],
                "head": _point(msg, e.seg, e._ptroff(0)), "tail": _point(msg, e.seg, e._ptroff(1))}
    if which == 10:
        return {"type": "dragonSplit", "parentId": e.i32(0), "childId": e.i32(4),
                "team": e.u16(8), "childFacing": DIR[e.u16(10)],
                "parentBody": list_of_points(msg, e.seg, e._ptroff(0)),
                "childBody": list_of_points(msg, e.seg, e._ptroff(1))}
    if which == 11:
        return {"type": "dragonDeath", "id": e.i32(0), "reason": e.u16(4)}
    if which == 12:
        d = {"type": "sonarPing", "senderId": e.i32(0), "direction": DIR[e.u16(4)]}
        if not e.null_ptr(0):
            d["origin"] = _point(msg, e.seg, e._ptroff(0))
        if not e.null_ptr(1):
            d["end"] = _point(msg, e.seg, e._ptroff(1))
        res = e.u16(6)
        d["_sonarResult"] = ["noHit", "hitId"][res] if res in (0, 1) else res
        if res == 1:
            d["hitId"] = e.i32(12)
        return d
    return {"type": "unknown", "which": which}


def decode(path):
    msg = Message(open(path, "rb").read())
    root_seg, root_off = msg.follow(0, 0)
    replay = Struct(msg, root_seg, root_off, "Replay")
    meta = {
        "formatVersion": replay.u32(0),
        "map": replay.text(0),
        "botA": replay.text(1),
        "botB": replay.text(2),
        "seed": None if replay.u16(4) == 0 else replay.u64(8),
    }
    if not replay.null_ptr(4):
        res = Struct(msg, *msg.follow(replay.seg, replay._ptroff(4)), "GameResult")
        mr = {"terminated": res.bit(0), "endReason": res.u16(2),
              "winner": None if res.u16(4) == 0 else res.u16(6)}
        for t, idx in (("teamA", 0), ("teamB", 1)):
            tsd = Struct(msg, *msg.follow(res.seg, res._ptroff(idx)), "TeamStanding")
            mr[t] = {"dragonCount": tsd.i32(0), "longestDragon": tsd.i32(4),
                     "totalLength": tsd.i32(8)}
        meta["result"] = mr

    evseg, _ = replay.ptr(3)
    count, dw, pc, first = struct_list(msg, replay.seg, replay._ptroff(3))
    events = []
    for i in range(count):
        off = first + i * (dw * 8 + pc * 8)
        events.append(decode_event(msg, evseg, off))
    return meta, events


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    path = sys.argv[1]
    out = None
    if "-o" in sys.argv:
        out = sys.argv[sys.argv.index("-o") + 1]
    meta, events = decode(path)
    result = {"meta": meta, "eventCount": len(events), "events": events}
    if out:
        json.dump(result, open(out, "w"), ensure_ascii=False)
        print(f"wrote {len(events)} events to {out}")
    else:
        # brief summary
        from collections import Counter
        print("map name:", meta["map"].split("MAP_NAME ")[1].split("\n")[0] if "MAP_NAME" in meta["map"] else "?")
        print("botA:", meta["botA"], "| botB:", meta["botB"])
        print("formatVersion:", meta["formatVersion"])
        print("total events:", len(events))
        kinds = Counter(e["type"] for e in events)
        print("types:", dict(kinds))
        nact = sum(1 for e in events if e["type"] == "dragonAction")
        print("dragonAction events:", nact)


if __name__ == "__main__":
    main()
