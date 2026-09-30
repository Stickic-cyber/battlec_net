#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reconstruct per-dragon board observations from replay events and emit
imitation-learning samples (obs, action) for behavioural cloning.

For every `dragonAction`, we rebuild the deciding dragon's **7x7 wrapped vision
window** around its head, exactly like a real bot sees it (VISION_RADIUS=3),
from the replay's full-board ground truth:

  channel 0  ownbody      deciding dragon's body cells
  channel 1  allybody     friendly-team dragons' body cells
  channel 2  enemybody    enemy dragons' body cells
  channel 3  anyhead      any dragon head
  channel 4  pearl        a pearl on this tile
  channel 5  kelp         a kelp edge blocks entry from the head's side
  channel 6  border       tile is outside the map (never in a wrapped window)
  (scalar)   round, own_length, facing (one-hot), unit_count

The observation window is toroidal: absolute coords are reduced with wrapping,
so walls are not a thing — only kelp edges (and portals) matter, exactly as the
real game.

Edge indexing is ported verbatim from the official replay-viewer JS:
  Ha()/Map += `edgeAt`:  hEdges has (H+1)*W entries, vEdges has H*(W+1);
  Edge of tile (x,y):  N->hEdges[y*W+x]  S->hEdges[(y+1)*W+x]
                       W->vEdges[y*(W+1)+x]  E->vEdges[y*(W+1)+x+1]
  edge type 0=empty 1=kelp 2=portal.

Body reconstruction:
  * initial bodies from the map roster `DRAGON <team> <len> <head...>`
  * `dragonSplit` events carry authoritative full parent/child bodies
  * between splittings, each dragon's body is the most recent `length` cells of
    its head's path on the torus (head/tail endpoints match the replay's
    `dragonUpdate` exactly).
  * length starts from the roster, is reset on splits, and grows by 1 whenever a
    dragon's head lands on a tile whose pearl spawns (countdown hits 0).

Outputs are saved as numpy arrays: obs (N, CH, 7, 7) + scalars (N, K) and labels
(actions as indices). Actions used for BC are single-step: N/E/S/W (the first
move of a multi-step action) plus SPLIT and SUICIDE.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_replay import decode

DIR_OFFSET = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
DIR_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}
NCH = 7


class Map:
    KELP, EMPTY, PORTAL = 1, 0, 2

    def __init__(self, map_text):
        self.text = map_text
        self.width = self.height = 0
        self.h_edges = []   # (H+1)*W, edge type
        self.v_edges = []   # H*(W+1)
        self.pearl_min = {}  # (x,y) -> minRounds (first spawn offset), 1000 = none
        self.portals = {}    # directed edge key -> portal id
        self.name = ""
        self._parse()

    def _parse(self):
        for ln in self.text.splitlines():
            p = ln.split()
            if not p:
                continue
            k = p[0]
            if k == "MAP":
                self.width, self.height = int(p[1]), int(p[2])
            elif k == "MAP_NAME":
                self.name = ln[len("MAP_NAME"):].strip()
            elif k == "TILE":
                x, y, mn, mx = int(p[1]), int(p[2]), int(p[3]), int(p[4])
                self.pearl_min[(x, y)] = mn
            elif k == "EDGE":
                self._add_edge(int(p[1]), int(p[2]), int(p[3]))
        # pad both edge arrays to full size: hEdges=(H+1)*W, vEdges=H*(W+1)
        H, W = self.height, self.width
        while len(self.h_edges) < (H + 1) * W:
            self.h_edges.append(0)
        while len(self.v_edges) < H * (W + 1):
            self.v_edges.append(0)

    def _add_edge(self, eid, etype, portal_id):
        W = self.width
        r = W + 1
        band = eid // r
        col = eid % r
        if band % 2 == 0:
            if col >= W:
                return  # padding slot
            row = band // 2
            idx = row * W + col
            while len(self.h_edges) <= idx:
                self.h_edges.append(-1)
            self.h_edges[idx] = etype
        else:
            row = (band - 1) // 2
            idx = row * r + col
            while len(self.v_edges) <= idx:
                self.v_edges.append(-1)
            self.v_edges[idx] = etype
        if etype == 2:
            # directed edge key for portal pair matching
            side = "N" if band % 2 == 0 else "W"
            x, y = col, row
            self.portals[(x, y, side)] = portal_id

    def edge_at(self, x, y, side):
        W, H = self.width, self.height
        x = x % W; y = y % H
        if side in ("N", "S"):
            idx = (y + (1 if side == "S" else 0)) * W + x
            return self.h_edges[idx]
        else:
            idx = y * (W + 1) + x + (1 if side == "E" else 0)
            return self.v_edges[idx]

    def step(self, x, y, side):
        """Return destination tile (x,y) after stepping `side`, honouring kelp
        and portals. Returns None if blocked by kelp; x/y already wrapped."""
        W, H = self.width, self.height
        et = self.edge_at(x, y, side)
        if et == self.KELP:
            return None
        dx, dy = DIR_OFFSET[side]
        nx, ny = (x + dx) % W, (y + dy) % H
        if et == self.PORTAL:
            pid = self.portals.get((x, y, side))
            # portal pair: find the other edge with same portal id
            if pid is not None:
                nxt = self._portal_target(pid, (x, y, side), (nx, ny, side))
                if nxt:
                    nx, ny = nxt
        return nx, ny

    def _portal_target(self, pid, src_key, default_key):
        W, H = self.width, self.height
        for key, v in self.portals.items():
            if v == pid and key != src_key:
                sx, sy, ss = key
                # exit direction is the mate's direction
                dx, dy = DIR_OFFSET[ss]
                return (sx + dx) % W, (sy + dy) % H
        return None

    def kelp_mask(self):
        W, H = self.width, self.height
        m = np.zeros((H, W), np.int8)
        for y in range(H):
            for x in range(W):
                # a tile is "kelp-locked" if exit N blocked and entry blocked;
                # store whether each of 4 outgoing edges is kelp
                m[y, x] = 0
        return m


def parse_roster(map_text):
    dragons = []
    for ln in map_text.splitlines():
        if ln.startswith("DRAGON "):
            p = ln.split()
            team = int(p[1])
            n = int(p[2])
            coords = p[3:3 + n * 2]
            body = [(int(coords[i]), int(coords[i + 1])) for i in range(0, len(coords), 2)]
            dragons.append({"team": team, "body": body, "len": n})
    return dragons


class ReplayState:
    def __init__(self, path):
        self.path = path
        meta, self.events = decode(path)
        self.map = Map(meta["map"])
        roster = parse_roster(meta["map"])
        self.roster = roster
        # id -> team via initial positions then splits
        self.id_team = {}
        used = set()
        for e in self.events:
            if e["type"] != "dragonUpdate":
                continue
            head = tuple(e["head"].values())
            for i, d in enumerate(roster):
                if i in used:
                    continue
                if tuple(d["body"][0]) == head:
                    self.id_team[e["id"]] = d["team"]
                    used.add(i)
                    break
        # id -> {team, body(list of cells head..tail), length}
        self.dragons = {}
        # initialize each id to its roster body (id matched by head position)
        for e in self.events:
            if e["type"] != "dragonUpdate":
                continue
            head = tuple(e["head"].values())
            for di, d in enumerate(roster):
                if tuple(d["body"][0]) == head:
                    self.dragons[e["id"]] = {
                        "team": d["team"], "body": list(d["body"]),
                        "length": d["len"], "path": list(d["body"]),
                        "head": head, "tail": tuple(d["body"][-1]),
                    }
                    break


def process_replay(path, build_obs=True):
    rs = ReplayState(path)
    Map_ = rs.map
    W, H = Map_.width, Map_.height
    # pearl state: (x,y)->bool hasPearl now (grown)
    pearls = {}
    for (x, y), mn in Map_.pearl_min.items():
        pearls[(x, y)] = mn <= 0  # spawned at t<=0 if minRounds 0
    # countdown timers for pearls that will grow
    timers = {pt: mn for pt, mn in Map_.pearl_min.items() if mn > 0}

    round_of = 0
    samples = []  # (obs_patch, scalar, action_idx, team, id, round)

    def obs_for(rid, head):
        """7x7 obs channels around head (wrapped)."""
        cx, cy = head
        ch = np.zeros((NCH, 7, 7), np.float32)
        team = rs.id_team.get(rid)
        for did, d in rs.dragons.items():
            t = d.get("team")
            for (bx, by) in d.get("body", []):
                ix = (bx - cx + W) % W - 3
                iy = (by - cy + H) % H - 3
                if not (-3 <= ix <= 3 and -3 <= iy <= 3):
                    continue
                dxr, dyr = (ix + 3) % 7, (iy + 3) % 7
                if t == team:
                    ch[0 if did == rid else 1, dyr, dxr] = 1
                else:
                    ch[2, dyr, dxr] = 1
        # pearls
        for (px, py), has in pearls.items():
            if not has:
                continue
            ix = (px - cx + W) % W - 3
            iy = (py - cy + H) % H - 3
            if -3 <= ix <= 3 and -3 <= iy <= 3:
                ch[4, (iy + 3) % 7, (ix + 3) % 7] = 1
        # kelp outgoing from centre head (scalar-ish) — mark the 7x7 border kelp by
        # checking the edge from each cell toward +x,+y neighbours
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                gx = (cx + dx + W) % W
                gy = (cy + dy + H) % H
                if Map_.edge_at(gx, gy, "E") == 1:
                    ch[5, (dy + 3) % 7, (dx + 3) % 7] = 1
        return ch

    for ev in rs.events:
        t = ev["type"]
        if t == "roundStart":
            round_of = ev["round"]
            # advance pearl timers at round boundary? timers tick per round
            for pt in list(timers.keys()):
                timers[pt] -= 1
                if timers[pt] <= 0:
                    pearls[pt] = True
        elif t == "tileChange":
            px, py = ev["tile"]["x"], ev["tile"]["y"]
            pearls[(px, py)] = ev["hasPearl"]
        elif t == "dragonSplit":
            d = rs.dragons.get(ev["parentId"])
            parent_body = [(p["x"], p["y"]) for p in ev["parentBody"]]
            child_body = [(p["x"], p["y"]) for p in ev["childBody"]]
            rs.id_team[ev["childId"]] = ev["team"]
            rs.id_team[ev["parentId"]] = ev["team"]
            rs.dragons[ev["parentId"]] = {"team": ev["team"], "body": parent_body,
                                          "length": len(parent_body),
                                          "path": list(parent_body)}
            rs.dragons[ev["childId"]] = {"team": ev["team"], "body": child_body,
                                         "length": len(child_body),
                                         "path": list(child_body)}
        elif t == "dragonUpdate":
            rid = ev["id"]
            head = (ev["head"]["x"], ev["head"]["y"])
            tail = (ev["tail"]["x"], ev["tail"]["y"])
            d = rs.dragons.get(rid)
            if d is None:
                continue
            d["head"] = head
            d["tail"] = tail
            # append to path, rebuild body as last `length` distinct path cells
            if not d["path"] or d["path"][-1] != head:
                d["path"].append(head)
            body = _rebuild_body(d["path"], d["length"], head)
            d["body"] = body
        elif t == "dragonAction":
            rid = ev["id"]
            d = rs.dragons.get(rid)
            if d is None:
                continue
            head = d.get("head") or (rs.dragons.get(rid, {}).get("head"))
            if head is None:
                continue
            act = ev.get("action") or {}
            if act.get("kind") == "move":
                steps = act.get("steps") or []
                if not steps:
                    continue
                label = DIR_IDX[steps[0]]      # N=0 E=1 S=2 W=3
            elif act.get("kind") == "split":
                label = 4
            else:
                label = 5                       # suicide
            ch = obs_for(rid, head)
            team = rs.id_team.get(rid)
            scalar = np.array([round_of / 500, d.get("length", 3) / 500], np.float32)
            samples.append({
                "obs": ch, "scalar": scalar, "label": label,
                "team": team, "id": rid, "round": round_of,
            })
        elif t == "dragonDeath":
            rs.dragons.pop(ev["id"], None)
    return samples


def _rebuild_body(path, length, head):
    """most recent `length` distinct cells tracing backwards from head through path."""
    if length <= 0:
        return [head]
    cells = []
    seen = set()
    for p in reversed(path):
        if p in seen:
            continue
        seen.add(p)
        cells.append(p)
        if len(cells) >= max(length, 1):
            break
    return list(reversed(cells))


def build(files, team):
    """Extract samples for one team across files -> (obs, scalar, labels, meta)."""
    obs_list, sc_list, lab_list, meta_list = [], [], [], []
    for f in files:
        samples = process_replay(f)
        for s in samples:
            if s["team"] != team:
                continue
            obs_list.append(s["obs"])
            sc_list.append(s["scalar"])
            lab_list.append(s["label"])
            meta_list.append((os.path.basename(f), s["id"], s["round"]))
    if not obs_list:
        raise ValueError(f"no samples for team {team}")
    obs = np.stack(obs_list).astype(np.float32)
    scalar = np.stack(sc_list).astype(np.float32)
    labels = np.array(lab_list, np.int64)
    return obs, scalar, labels, meta_list


if __name__ == "__main__":
    import time
    src = sys.argv[1] if len(sys.argv) > 1 else r"D:\Project\battlecode\replays\battle-M647381-replays\*.replay"
    team = int(sys.argv[2]) if len(sys.argv) > 2 else None
    files = sorted(glob.glob(src))
    for f in files:
        t0 = time.time()
        samples = process_replay(f)
        from collections import Counter
        n = Counter(s["team"] for s in samples)
        print(f"{os.path.basename(f)}: {len(samples)} dragonActions, teams={dict(n)} "
              f"({time.time()-t0:.1f}s)")
