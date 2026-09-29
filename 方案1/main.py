"""
UNSW Battlecode — Competitive Dragon Bot (mybotnew)
===================================================
Core Systems:
1. Full Self-Body Tracking & Vision-Verified Tail Release (`own_release_time`)
2. Body-Aware Flood Fill + Tail-Escape Detection + Next-Step Exit Degree
3. Safe Portal Handling (prevents blind portal-exit collisions)
4. Zero Friendly-Fire Sacrifice Rule & Emergency Tail-Reversal Split
5. Unified Single-Pass Graph BFS Target Selection & Ally De-confliction
6. 64-Bit Authenticated Sonar Protocol with Multi-Hop King Beacon & Enemy Intel
7. Endgame Sacrifice-Feed Strategy (Last 5% Rounds: Round 475-498):
   - Short dragons converge toward KING / longer allies (starting Round 462),
     send a pre-detonation feed signal (`MSG_FEED_HERE`) 1 turn prior, and
     self-destruct in-place WITHOUT collision (by outputting no MOVE/SPLIT before
     ENDTURN -> `died: no valid action`) at distance 1-2 from the long dragon,
     crystallizing ceil(L/2) pearls directly in front of the KING!
8. Virtual-Clock CPU Budget Guard (`time.perf_counter()`):
   Uses the sandbox's 1 ns = 1 CPU point clock to guarantee zero timeouts.
"""
import sys
import time
from collections import deque

import helper as unswbc
from helper import Direction, EdgeType, Position, Team, Constants

# =============================================================================
# 1. GLOBALS & PERSISTENT STATE
# =============================================================================

ct: unswbc.Controller
game: unswbc.Game

_DIRS = Direction.get_direction_list()  # [NORTH, EAST, SOUTH, WEST]

world_map: dict = {}          # (x, y) -> {'kelp': set, 'portals': dict, 'has_pearl': bool, 'pearl_time': int, 'round': int, 'spawns': bool}
portal_endpoints: dict = {}   # (pid, enter_dir) -> list of ((src_x, src_y), (naive_exit_x, naive_exit_y))
portal_dest_ready: dict = {}  # (x, y, d) -> (dest_x, dest_y) once partner portal is discovered

map_width: int = 0
map_height: int = 0
my_team: Team = None
my_team_magic: int = 0

# Exact body tracking: list of (x, y) from head (index 0) to tail (index L-1)
own_body: list = []
last_target: tuple = None
turn_start_clock: float = 0.0

# Team & enemy intel from sonar:
# intel_type values:
ITYPE_TARGET      = 0  # (ix, iy) is sender's current target tile
ITYPE_ENEMY_HEAD  = 1  # (ix, iy) is a spotted enemy head
ITYPE_KING_BEACON = 2  # (ix, iy) is the known KING's position (direct or gossiped)
ITYPE_FEED_HERE   = 3  # (ix, iy) is where a short dragon is about to self-destruct to feed KING!

ally_states: dict = {}        # dragon_id -> {'x': x, 'y': y, 'len': length, 'tx': ix, 'ty': iy, 'itype': itype, 'round': rnd}
enemy_reports: list = []      # list of {'x': ex, 'y': ey, 'round': rnd}
feed_beacons: list = []       # list of {'x': fx, 'y': fy, 'round': rnd, 'id': sid}
known_king_info: dict = {'id': -1, 'x': -1, 'y': -1, 'len': 0, 'round': -99}
enemy_pinged_me: bool = False


# =============================================================================
# 2. 64-BIT AUTHENTICATED SONAR PROTOCOL
# =============================================================================

def _sonar_checksum(payload_top55: int) -> int:
    v = payload_top55 ^ (payload_top55 >> 18) ^ (payload_top55 >> 36)
    return (v * 0x15B) & 0x1FF


def encode_sonar_packet(sender_id: int, x: int, y: int, length: int,
                        ix: int, iy: int, itype: int, rnd: int) -> int:
    top55 = (
        ((my_team_magic & 0xF) << 51) |
        ((sender_id & 0xFF) << 43) |
        ((x & 0x3F) << 37) |
        ((y & 0x3F) << 31) |
        ((min(length, 255) & 0xFF) << 23) |
        ((ix & 0x3F) << 17) |
        ((iy & 0x3F) << 11) |
        ((itype & 0x3) << 9) |
        (rnd & 0x1FF)
    )
    chk = _sonar_checksum(top55)
    return (top55 << 9) | chk


def decode_sonar_packet(msg: int, current_rnd: int):
    chk = msg & 0x1FF
    top55 = msg >> 9
    if _sonar_checksum(top55) != chk:
        return None
    magic = (top55 >> 51) & 0xF
    if magic != my_team_magic:
        return None
    msg_rnd = top55 & 0x1FF
    if ((current_rnd - msg_rnd) & 0x1FF) > 2:
        return None
    sid = (top55 >> 43) & 0xFF
    sx = (top55 >> 37) & 0x3F
    sy = (top55 >> 31) & 0x3F
    slen = (top55 >> 23) & 0xFF
    ix = (top55 >> 17) & 0x3F
    iy = (top55 >> 11) & 0x3F
    itype = (top55 >> 9) & 0x3
    return sid, sx, sy, slen, ix, iy, itype


def process_sonar(rnd: int):
    """Decode incoming sonar messages, detect enemy sonar pings, and update King/Feed intel."""
    global enemy_reports, feed_beacons, known_king_info, enemy_pinged_me
    my_id = ct.get_id()
    enemy_pinged_me = False

    for raw in ct.get_sonar_messages():
        parsed = decode_sonar_packet(raw, rnd)
        if parsed is None:
            # Message failed our team authentication -> an enemy dragon's sonar ray just hit us!
            enemy_pinged_me = True
            continue
        sid, sx, sy, slen, ix, iy, itype = parsed
        if sid == my_id:
            continue
        ally_states[sid] = {
            'x': sx, 'y': sy, 'len': slen,
            'tx': ix, 'ty': iy, 'itype': itype, 'round': rnd
        }
        # Update known king if this sender is longer (or if previous king info is very old > 25 rounds)
        if slen > known_king_info['len'] or (rnd - known_king_info['round'] > 25) or sid == known_king_info['id']:
            if slen >= known_king_info['len'] or rnd - known_king_info['round'] > 25:
                known_king_info = {'id': sid, 'x': sx, 'y': sy, 'len': slen, 'round': rnd}

        if itype == ITYPE_ENEMY_HEAD:
            enemy_reports.append({'x': ix, 'y': iy, 'round': rnd})
        elif itype == ITYPE_KING_BEACON:
            # Direct or gossiped King location
            if slen >= known_king_info['len']:
                known_king_info = {'id': sid, 'x': ix, 'y': iy, 'len': slen, 'round': rnd}
            elif rnd - known_king_info['round'] >= 1:
                known_king_info['x'] = ix
                known_king_info['y'] = iy
                known_king_info['round'] = rnd
        elif itype == ITYPE_FEED_HERE:
            feed_beacons.append({'x': ix, 'y': iy, 'round': rnd, 'id': sid})

    stale_ids = [aid for aid, st in ally_states.items() if rnd - st['round'] > 6]
    for aid in stale_ids:
        del ally_states[aid]
    enemy_reports = [er for er in enemy_reports if rnd - er['round'] <= 8]
    feed_beacons = [fb for fb in feed_beacons if rnd - fb['round'] <= 4]


def broadcast_sonar(rnd: int, role: str, vis_enemies: list, chosen_target: tuple, feed_pos: tuple = None):
    """Send 1 compact 64-bit packet per direction (only if we have allies)."""
    if ct.get_unit_count() <= 1:
        return

    pos = ct.get_position()
    my_id = ct.get_id()
    my_len = ct.get_length()

    if feed_pos is not None:
        # Tell nearby KING / longer dragon the exact tile where we will drop pearls!
        ix, iy, itype = feed_pos[0], feed_pos[1], ITYPE_FEED_HERE
    elif vis_enemies:
        ex, ey = vis_enemies[0]
        ix, iy, itype = ex, ey, ITYPE_ENEMY_HEAD
    elif role == "KING":
        # KING broadcasts its own position as the King Beacon (especially crucial in endgame!)
        ix, iy, itype = pos.x, pos.y, ITYPE_KING_BEACON
    elif rnd >= 455 and (rnd - known_king_info['round'] <= 3) and known_king_info['x'] >= 0:
        # In endgame, non-KING dragons gossip the KING's position across kelp corners!
        ix, iy, itype = known_king_info['x'], known_king_info['y'], ITYPE_KING_BEACON
    elif chosen_target is not None:
        ix, iy, itype = chosen_target[0], chosen_target[1], ITYPE_TARGET
    else:
        ix, iy, itype = pos.x, pos.y, ITYPE_TARGET

    pkt = encode_sonar_packet(my_id, pos.x, pos.y, my_len, ix, iy, itype, rnd)
    for d in _DIRS:
        ct.send_sonar(d, pkt)


# =============================================================================
# 3. WORLD MAP, PORTAL LINKING & EXACT BODY TRACKING
# =============================================================================

def _register_portal_edge(pid: int, x: int, y: int, d: Direction):
    dx, dy = d.get_offset()
    naive_across = ((x + dx) % map_width, (y + dy) % map_height)
    key = (pid, d)
    if key not in portal_endpoints:
        portal_endpoints[key] = []
    entry = ((x, y), naive_across)
    if entry not in portal_endpoints[key]:
        portal_endpoints[key].append(entry)

    opp_d = d.get_opposite()
    opp_key = (pid, opp_d)
    if opp_key in portal_endpoints:
        for (ox, oy), _ in portal_endpoints[opp_key]:
            if (ox, oy) != naive_across:
                portal_dest_ready[(x, y, d)] = (ox, oy)
                portal_dest_ready[(ox, oy, opp_d)] = (x, y)


def _reconstruct_body_from_vision(head_pos: tuple, vis_dragons: dict) -> list:
    my_id = ct.get_id()
    my_len = ct.get_length()
    my_segments = {
        pos_k: part for pos_k, part in vis_dragons.items()
        if part.get_team() == my_team and part.get_id() == my_id and pos_k != head_pos
    }
    chain = [head_pos]
    cur_x, cur_y = head_pos
    used = {head_pos}

    while len(chain) < my_len:
        found_prev = None
        for d in _DIRS:
            dx, dy = d.get_offset()
            nx, ny = (cur_x + dx) % map_width, (cur_y + dy) % map_height
            nk = (nx, ny)
            if nk in used or nk not in my_segments:
                continue
            part = my_segments[nk]
            if part.get_dir() == d.get_opposite():
                found_prev = nk
                break
        if found_prev is None:
            for d in _DIRS:
                dx, dy = d.get_offset()
                nk = ((cur_x + dx) % map_width, (cur_y + dy) % map_height)
                if nk not in used and nk in my_segments:
                    found_prev = nk
                    break
        if found_prev is None:
            break
        chain.append(found_prev)
        used.add(found_prev)
        cur_x, cur_y = found_prev

    return chain


def update_perception(rnd: int):
    """
    Update world_map, portal links, visible dragons, and exact own_body tracking.
    Returns (vis_dragons, vis_enemies, vis_enemy_any, own_release_time, own_body_set).
    """
    global map_width, map_height, my_team, my_team_magic, own_body

    if map_width == 0:
        map_width, map_height = game.get_map_size()
        my_team = ct.get_team()
        my_team_magic = 0xA if my_team == Team.A else 0xB

    pos = ct.get_position()
    head_k = (pos.x, pos.y)
    my_id = ct.get_id()
    my_len = ct.get_length()

    vis_dragons = {}
    vis_enemies = []
    vis_enemy_any = False
    vis_tiles_set = set()
    vis_own_segments = set()

    for tile in ct.get_tiles():
        if tile is None:
            continue
        tp = tile.get_position()
        k = (tp.x, tp.y)
        vis_tiles_set.add(k)

        part = tile.get_dragon()
        if part is not None:
            vis_dragons[k] = part
            if part.get_team() == my_team and part.get_id() == my_id:
                vis_own_segments.add(k)
            elif part.get_team() != my_team:
                vis_enemy_any = True
                if part.is_head():
                    vis_enemies.append(k)

        if k not in world_map:
            kelp = set()
            portals = {}
            for d in _DIRS:
                edge = tile.get_edge(d)
                et = edge.get_edge_type()
                if et == EdgeType.KELP:
                    kelp.add(d)
                elif et == EdgeType.PORTAL:
                    pid = edge.get_portal_id()
                    portals[d] = pid
                    if pid >= 0:
                        _register_portal_edge(pid, tp.x, tp.y, d)
            pt = tile.get_pearl_time()
            world_map[k] = {
                'kelp': kelp,
                'portals': portals,
                'has_pearl': tile.has_pearl(),
                'pearl_time': pt,
                'round': rnd,
                'spawns': (pt >= 0 or tile.has_pearl()),
            }
        else:
            wm = world_map[k]
            pt = tile.get_pearl_time()
            hp = tile.has_pearl()
            wm['has_pearl'] = hp
            wm['pearl_time'] = pt
            wm['round'] = rnd
            if pt >= 0 or hp:
                wm['spawns'] = True

    if not own_body:
        own_body = _reconstruct_body_from_vision(head_k, vis_dragons)
    else:
        if own_body[0] != head_k:
            own_body.insert(0, head_k)
        own_body = [
            bk for bk in own_body
            if (bk not in vis_tiles_set) or (bk in vis_own_segments)
        ]
        if len(own_body) > my_len:
            del own_body[my_len:]
        if len(own_body) < len(vis_own_segments):
            recon = _reconstruct_body_from_vision(head_k, vis_dragons)
            for rk in recon:
                if rk not in own_body and len(own_body) < my_len:
                    own_body.append(rk)

    own_release_time = {}
    cur_tracked_len = len(own_body)
    for idx, bk in enumerate(own_body):
        rel = cur_tracked_len - idx
        if rel > own_release_time.get(bk, 0):
            own_release_time[bk] = rel

    for vk in vis_own_segments:
        if vk not in own_release_time:
            own_release_time[vk] = 3

    own_body_set = set(own_release_time.keys())
    return vis_dragons, vis_enemies, vis_enemy_any, own_release_time, own_body_set


# =============================================================================
# 4. GRAPH STEP & BODY-AWARE FLOOD FILL
# =============================================================================

def step_tile_no_blind_portal(cx: int, cy: int, d: Direction):
    ck = (cx, cy)
    wm = world_map.get(ck)
    if wm is not None:
        if d in wm['kelp']:
            return None
        if d in wm['portals']:
            dest = portal_dest_ready.get((cx, cy, d))
            if dest is not None:
                dt = ct.get_tile(Position(dest[0], dest[1]))
                if dt is not None and dt.get_dragon() is None:
                    return dest[0], dest[1]
            return None

    dx, dy = d.get_offset()
    nx, ny = (cx + dx) % map_width, (cy + dy) % map_height
    nk = (nx, ny)
    nwm = world_map.get(nk)
    if nwm is not None:
        opp = d.get_opposite()
        if opp in nwm['kelp'] or opp in nwm['portals']:
            return None
    return nx, ny


def _count_head_open_exits(hx: int, hy: int, vis_dragons: dict, extra_blocked: tuple = None) -> int:
    exits = 0
    for d in _DIRS:
        res = step_tile_no_blind_portal(hx, hy, d)
        if res is None:
            continue
        nk = res
        if nk == extra_blocked:
            continue
        if nk in vis_dragons:
            continue
        exits += 1
    return exits


def flood_fill_eval(dest_k: tuple, eats_pearl: bool, vis_dragons: dict, own_release_time: dict):
    my_id = ct.get_id()
    my_len = ct.get_length()
    max_tiles = min(65, max(30, my_len * 2 + 12))
    pearl_delay = 1 if eats_pearl else 0
    tail_k = own_body[-1] if own_body else None

    other_bodies = {
        vk for vk, part in vis_dragons.items()
        if not (part.get_team() == my_team and part.get_id() == my_id)
    }

    queue = deque([(dest_k[0], dest_k[1], 1)])
    visited = {dest_k}
    area = 0
    exit_degree = 0
    can_reach_tail = False

    while queue and area < max_tiles:
        cx, cy, dist = queue.popleft()
        area += 1
        ndist = dist + 1

        for d in _DIRS:
            res = step_tile_no_blind_portal(cx, cy, d)
            if res is None:
                continue
            nx, ny = res
            nk = (nx, ny)

            if nk in other_bodies:
                continue

            rel = own_release_time.get(nk, 0)
            if rel > 0 and rel > (ndist - 1 - pearl_delay):
                continue

            if nk == tail_k and ndist >= 2:
                can_reach_tail = True

            if dist == 1:
                exit_degree += 1

            if nk in visited:
                continue
            visited.add(nk)
            queue.append((nx, ny, ndist))

    if area >= max_tiles:
        can_reach_tail = True

    return area, exit_degree, can_reach_tail


def evaluate_moves(vis_dragons: dict, own_release_time: dict, own_body_set: set, role: str):
    pos = ct.get_position()
    hx, hy = pos.x, pos.y
    here_tile = ct.get_tile(pos)
    my_len = ct.get_length()

    all_moves = []
    if here_tile is None:
        return all_moves, []

    for d in _DIRS:
        edge = here_tile.get_edge(d)
        et = edge.get_edge_type()
        if et == EdgeType.KELP:
            continue

        is_portal = (et == EdgeType.PORTAL)
        blind_portal = False
        if is_portal:
            known_dest = portal_dest_ready.get((hx, hy, d))
            if known_dest is not None:
                nx, ny = known_dest
                dest_tile = ct.get_tile(Position(nx, ny))
                if dest_tile is None:
                    blind_portal = True
            else:
                blind_portal = True
                dx, dy = d.get_offset()
                nx, ny = (hx + dx) % map_width, (hy + dy) % map_height
                dest_tile = None
        else:
            dx, dy = d.get_offset()
            nx, ny = (hx + dx) % map_width, (hy + dy) % map_height
            dest_tile = ct.get_tile(Position(nx, ny))

        dest_k = (nx, ny)

        if not blind_portal:
            if dest_k in own_body_set:
                continue
            if dest_tile is not None and dest_tile.get_dragon() is not None:
                continue

        danger = 0.0
        has_pearl = False
        pearl_time = -1

        if blind_portal:
            danger += 140.0
            area, exit_deg, can_reach_tail = 8, 1, False
        else:
            if dest_tile is not None:
                has_pearl = dest_tile.has_pearl()
                pearl_time = dest_tile.get_pearl_time()
            elif dest_k in world_map:
                has_pearl = world_map[dest_k]['has_pearl']
                pearl_time = world_map[dest_k]['pearl_time']

            area, exit_deg, can_reach_tail = flood_fill_eval(
                dest_k, has_pearl, vis_dragons, own_release_time
            )

            if exit_deg == 0:
                danger += 500.0
            elif exit_deg == 1 and not can_reach_tail and area <= my_len + 3:
                danger += 200.0

            if area <= my_len + 1 and not can_reach_tail:
                danger += 240.0 + (my_len + 1 - area) * 10.0
            else:
                danger -= min(area, 50) * 0.65
                danger -= (exit_deg - 1) * 4.0

        if not blind_portal:
            for adj_d in _DIRS:
                res = step_tile_no_blind_portal(nx, ny, adj_d)
                if res is None:
                    continue
                ak = res
                if ak == (hx, hy):
                    continue
                adj_part = vis_dragons.get(ak)
                if adj_part is not None and adj_part.is_head():
                    other_exits = _count_head_open_exits(ak[0], ak[1], vis_dragons, extra_blocked=dest_k)
                    if adj_part.get_team() == my_team:
                        if other_exits == 0:
                            danger += 180.0
                        elif other_exits == 1:
                            danger += 65.0
                        else:
                            danger += 28.0
                    else:
                        if role == "KING" or my_len >= 9:
                            danger += 65.0
                        elif other_exits == 0 and my_len <= 7 and ct.get_unit_count() >= 3:
                            danger -= 30.0
                        else:
                            danger += 20.0

            if role == "KING" and vis_dragons:
                for vk, vpart in vis_dragons.items():
                    if vpart.get_team() != my_team and vpart.is_head():
                        edx = min(abs(vk[0] - nx), map_width - abs(vk[0] - nx))
                        edy = min(abs(vk[1] - ny), map_height - abs(vk[1] - ny))
                        if edx + edy <= 2:
                            danger += 35.0

        viable = (
            (not blind_portal)
            and (exit_deg >= 1)
            and (area > my_len + 1 or can_reach_tail)
            and (danger < 90.0)
        )
        all_moves.append({
            'dir': d,
            'dest': dest_k,
            'danger': danger,
            'area': area,
            'exit_deg': exit_deg,
            'has_pearl': has_pearl,
            'pearl_time': pearl_time,
            'viable': viable,
        })

    all_moves.sort(key=lambda m: (m['danger'], not m['has_pearl'], -m['area']))
    viable_moves = [m for m in all_moves if m['viable']]
    return all_moves, viable_moves


# =============================================================================
# 5. ROLE SYSTEM, ENDGAME SACRIFICE-FEED & SMART SPLITTING
# =============================================================================

def determine_role(vis_enemies: list, vis_dragons: dict) -> str:
    my_id = ct.get_id()
    my_len = ct.get_length()
    unit_count = ct.get_unit_count()
    rnd = game.get_round_num()

    max_ally_len = known_king_info['len'] if (rnd - known_king_info['round'] <= 25 and known_king_info['id'] != my_id) else 0
    for aid, st in ally_states.items():
        if st['len'] > max_ally_len:
            max_ally_len = st['len']

    # Also check visible ally segment counts in 7x7 window
    vis_ally_counts = {}
    for vk, part in vis_dragons.items():
        if part.get_team() == my_team and part.get_id() != my_id:
            aid = part.get_id()
            vis_ally_counts[aid] = vis_ally_counts.get(aid, 0) + 1
    for aid, cnt in vis_ally_counts.items():
        if cnt > max_ally_len:
            max_ally_len = cnt

    is_longest = (my_len > max_ally_len) or (
        my_len == max_ally_len and all(
            my_id < aid for aid, st in ally_states.items() if st['len'] == my_len
        ) and all(
            my_id < aid for aid, cnt in vis_ally_counts.items() if cnt == my_len
        )
    )

    # In the endgame (rnd >= 460), only the true longest dragon is KING; everyone else is a potential FEEDER!
    if rnd >= 460:
        if is_longest and (unit_count <= 2 or my_len >= max(7, max_ally_len)):
            return "KING"
        return "FEEDER"

    if (unit_count >= 3 and is_longest) or (rnd >= 180 and is_longest) or my_len >= 14:
        return "KING"

    echoes = ct.get_sonar_echoes()
    if unit_count >= 4 and 5 <= my_len <= 10:
        if vis_enemies or echoes.enemy_head > 0:
            return "HUNTER"

    return "FARMER"


def check_endgame_sacrifice_feed(rnd: int, role: str, vis_dragons: dict, vis_enemy_any: bool):
    """
    In the last 5% of rounds (Round 475..498):
    Check if this non-KING dragon is within safe feeding distance (1-2 steps) of a
    longer (or equal-length lower-ID) ally dragon head, with no enemies in vision.
    Returns:
      - ('DETONATE_NOW', superior_ally_id) if we should self-destruct in-place this turn
        by outputting NO action before ENDTURN (zero-collision pearl crystallization!).
      - ('APPROACH_AND_PING', best_superior_head_pos) if a superior ally head is at
        distance 2..4 in vision and we should step next to it while sending ITYPE_FEED_HERE.
      - (None, None) otherwise.
    """
    # Only consider endgame feeding from Round 465 onward, and never on Round 500
    if rnd < 465 or rnd >= 499:
        return None, None
    if role == "KING" or ct.get_unit_count() < 2 or vis_enemy_any:
        return None, None

    pos = ct.get_position()
    hx, hy = pos.x, pos.y
    my_id = ct.get_id()
    my_len = ct.get_length()
    rounds_left = Constants.MAX_ROUNDS - rnd

    # Count visible segments for each ally in our 7x7 window
    vis_ally_segs = {}
    vis_ally_heads = []
    for vk, part in vis_dragons.items():
        if part.get_team() == my_team and part.get_id() != my_id:
            aid = part.get_id()
            vis_ally_segs[aid] = vis_ally_segs.get(aid, 0) + 1
            if part.is_head():
                vis_ally_heads.append((aid, vk[0], vk[1]))

    if not vis_ally_heads:
        return None, None

    # Determine the team's maximum known dragon length so we only feed the KING / long dragons
    max_team_len = my_len
    if rnd - known_king_info['round'] <= 8 and known_king_info['len'] > max_team_len:
        max_team_len = known_king_info['len']
    for st in ally_states.values():
        if st['len'] > max_team_len:
            max_team_len = st['len']
    for cnt in vis_ally_segs.values():
        if cnt > max_team_len:
            max_team_len = cnt

    # Identify long/KING ally heads in vision
    for aid, ax, ay in vis_ally_heads:
        est_len = max(
            vis_ally_segs.get(aid, 0),
            ally_states.get(aid, {}).get('len', 0),
            known_king_info['len'] if aid == known_king_info['id'] else 0,
        )
        # Recipient must be strictly longer than us (or tied with lower ID if both are long),
        # AND must be the team's KING or a genuinely long dragon (>= max(5, max_team_len - 2))
        # so two tiny L:2 dragons never wastefully detonate for each other!
        is_long_recipient = (
            (aid == known_king_info['id'] and est_len > my_len)
            or (est_len >= max(5, max_team_len - 2) and (
                est_len > my_len or (est_len == my_len and aid < my_id and my_len <= 6)
            ))
        )
        if not is_long_recipient:
            continue

        # Compute wrapped Manhattan distance & check direct/2-step kelp-free reachability
        dx = min(abs(ax - hx), map_width - abs(ax - hx))
        dy = min(abs(ay - hy), map_height - abs(ay - hy))
        mdist = dx + dy

        # Check if we are in the final 5% of rounds (Round 475..498)
        if rnd >= 475 and rounds_left >= mdist + 1:
            # Case 1: Distance 1 (directly adjacent across a non-kelp edge!)
            if mdist == 1:
                for d in _DIRS:
                    res = step_tile_no_blind_portal(hx, hy, d)
                    if res == (ax, ay):
                        return 'DETONATE_NOW', aid

            # Case 2: Distance 2 (connected by a 2-step kelp-free path to the long dragon head)
            if mdist == 2 and (rnd >= 480 or rounds_left <= my_len + 3):
                for d1 in _DIRS:
                    r1 = step_tile_no_blind_portal(hx, hy, d1)
                    if r1 is None:
                        continue
                    for d2 in _DIRS:
                        r2 = step_tile_no_blind_portal(r1[0], r1[1], d2)
                        if r2 == (ax, ay):
                            return 'DETONATE_NOW', aid

        # If we are at distance 2..6 (or Round 465..474), approach the long dragon head and ping ITYPE_FEED_HERE!
        if mdist <= 6:
            return 'APPROACH_AND_PING', (ax, ay)

    return None, None


def check_split_decision(role: str, all_moves: list, viable_moves: list, vis_enemies: list) -> int:
    my_len = ct.get_length()
    unit_count = ct.get_unit_count()
    unit_limit = game.get_unit_limit()
    rnd = game.get_round_num()

    if my_len < 4 or unit_count >= unit_limit:
        return 0

    # --- 1. EMERGENCY SURVIVAL SPLIT (Tail-Reversal Escape) ---
    if not viable_moves and rnd < 475:
        save_size = my_len - 2
        if ct.can_split(save_size):
            return save_size

    # --- 2. NEVER SPLIT THE KING OR ENDGAME FEEDERS ---
    if role in ("KING", "FEEDER") and (unit_count >= 3 or rnd >= 120):
        return 0

    # --- 3. STOP SPLITTING IN MID/LATE GAME TO MAXIMIZE LONGEST & TOTAL LENGTH ---
    if rnd > 280 and unit_count >= 3:
        return 0

    # --- 4. STRATEGIC EXPANSION SPLIT ---
    map_area = map_width * map_height
    target_units = min(unit_limit, max(4, min(14, map_area // 60)))
    if unit_count >= target_units:
        return 0

    if vis_enemies:
        return 0

    min_split_len = 6 if unit_count < 5 else 7
    if my_len < min_split_len:
        return 0

    if len(viable_moves) < 2 or viable_moves[0]['area'] < 18:
        return 0

    child_size = 3 if my_len >= 7 else (my_len // 2)
    if ct.can_split(child_size):
        return child_size
    return 0


# =============================================================================
# 6. UNIFIED GRAPH BFS TARGETING & PATHFINDING
# =============================================================================

def choose_best_move_via_bfs(viable_moves: list, vis_dragons: dict,
                             own_release_time: dict, role: str, rnd: int,
                             approach_king_pos: tuple = None):
    pos = ct.get_position()
    hx, hy = pos.x, pos.y
    my_id = ct.get_id()

    ally_heads = []
    for vk, part in vis_dragons.items():
        if part.get_team() == my_team and part.get_id() != my_id and part.is_head():
            ally_heads.append((vk[0], vk[1], part.get_id()))
    for aid, st in ally_states.items():
        if rnd - st['round'] <= 2:
            ally_heads.append((st['x'], st['y'], aid))

    ally_targets = {
        (st['tx'], st['ty']) for st in ally_states.values()
        if rnd - st['round'] <= 3 and st['itype'] == ITYPE_TARGET
    }

    other_bodies = {
        vk for vk, part in vis_dragons.items()
        if not (part.get_team() == my_team and part.get_id() == my_id)
    }

    # Immediate adjacent viable pearl check (unless we are a FEEDER right next to our King in >=475!)
    best_viable_danger = viable_moves[0]['danger']
    if not (role == "FEEDER" and rnd >= 472 and approach_king_pos is not None):
        for vm in viable_moves:
            if vm['has_pearl'] and vm['danger'] <= best_viable_danger + 14.0:
                return vm['dir'], vm['dest']

    # Determine Homing Target for FEEDER in endgame (Round >= 462), only if KING is reachable in remaining rounds
    king_homing_target = approach_king_pos
    if king_homing_target is None and role == "FEEDER" and rnd >= 462:
        if rnd - known_king_info['round'] <= 5 and known_king_info['x'] >= 0 and known_king_info['len'] > ct.get_length():
            kx, ky = known_king_info['x'], known_king_info['y']
            my_kdist = min(abs(hx - kx), map_width - abs(hx - kx)) + min(abs(hy - ky), map_height - abs(hy - ky))
            if my_kdist <= min(18, (Constants.MAX_ROUNDS - rnd) - 2):
                king_homing_target = (kx, ky)

    queue = deque()
    visited = {(hx, hy)}
    move_by_dir = {vm['dir']: vm for vm in viable_moves}

    for vm in viable_moves:
        dx, dy = vm['dest']
        visited.add((dx, dy))
        queue.append((dx, dy, vm['dir'], 1, 1 if vm['has_pearl'] else 0))

    best_score = -1e9
    best_dir = None
    best_target = None

    pref_dir = _DIRS[my_id % 4]
    pref_dx, pref_dy = pref_dir.get_offset()

    nodes = 0
    max_nodes = 220

    while queue and nodes < max_nodes:
        # Virtual clock CPU budget guard: stop BFS if we reach 65M points (0.065s)
        if (nodes & 31) == 0 and (time.perf_counter() - turn_start_clock) > 0.065:
            break

        cx, cy, first_dir, dist, p_delay = queue.popleft()
        nodes += 1
        ck = (cx, cy)

        tile_score = None
        wm = world_map.get(ck)

        # 1. If FEEDER has a known KING homing target in endgame, strongly reward getting within 1-2 tiles of KING!
        if king_homing_target is not None:
            kdx = min(abs(cx - king_homing_target[0]), map_width - abs(cx - king_homing_target[0]))
            kdy = min(abs(cy - king_homing_target[1]), map_height - abs(cy - king_homing_target[1]))
            kdist = kdx + kdy
            if kdist <= 2:
                tile_score = 240.0 - dist * 6.0 - kdist * 15.0
            else:
                tile_score = 160.0 - kdist * 8.0 - dist * 2.5

        # 2. If KING (or long dragon) hears a FEED_HERE beacon, prioritize moving toward the feeder!
        if role == "KING" and feed_beacons:
            for fb in feed_beacons:
                if ck == (fb['x'], fb['y']):
                    feed_score = 190.0 - dist * 5.0
                    if tile_score is None or feed_score > tile_score:
                        tile_score = feed_score

        # 3. Standard Pearl / Spawn / Frontier scoring
        if wm is None:
            rel_x = ((cx - hx + map_width // 2) % map_width) - map_width // 2
            rel_y = ((cy - hy + map_height // 2) % map_height) - map_height // 2
            dir_align = (rel_x * pref_dx + rel_y * pref_dy)
            std_score = 48.0 - dist * 2.2 + dir_align * 0.4
        else:
            age = rnd - wm['round']
            hp = wm['has_pearl']
            pt = wm['pearl_time']
            std_score = None

            if age == 0:
                if hp or (0 < pt <= dist):
                    ally_closer = False
                    # KING never yields pearls to smaller allies!
                    if role != "KING":
                        for ax, ay, aid in ally_heads:
                            adx = min(abs(cx - ax), map_width - abs(cx - ax))
                            ady = min(abs(cy - ay), map_height - abs(cy - ay))
                            adist = adx + ady
                            if adist < dist or (adist == dist and aid < my_id):
                                ally_closer = True
                                break
                    if not ally_closer:
                        std_score = 155.0 - dist * 6.0
                    else:
                        std_score = 35.0 - dist * 5.0
                elif 0 < pt <= dist + 2:
                    std_score = 95.0 - (pt * 5.0) - dist * 3.0
            else:
                expected_rem = pt - age if pt > 0 else -1
                if hp and age <= 20:
                    std_score = 90.0 - dist * 3.5 - age * 1.2
                elif pt > 0 and expected_rem <= dist:
                    std_score = 82.0 - dist * 3.2
                elif wm['spawns'] and age >= 18:
                    std_score = 42.0 + min(age, 40) * 0.4 - dist * 2.4

        if std_score is not None and (tile_score is None or std_score > tile_score):
            tile_score = std_score

        # 4. Enemy Intel adjustments (HUNTER pursues, KING avoids)
        if tile_score is not None and enemy_reports:
            for er in enemy_reports:
                edx = min(abs(cx - er['x']), map_width - abs(cx - er['x']))
                edy = min(abs(cy - er['y']), map_height - abs(cy - er['y']))
                edist = edx + edy
                if role == "KING" and edist <= 4:
                    tile_score -= (5 - edist) * 12.0
                elif role == "HUNTER" and edist <= 3:
                    tile_score += (4 - edist) * 10.0

        if tile_score is not None:
            if role != "KING" and ck in ally_targets:
                tile_score -= 30.0
            if last_target == ck:
                tile_score += 4.0
            tile_score -= move_by_dir[first_dir]['danger'] * 0.4
            if tile_score > best_score:
                best_score = tile_score
                best_dir = first_dir
                best_target = ck

        ndist = dist + 1
        if wm is None:
            continue

        for d in _DIRS:
            res = step_tile_no_blind_portal(cx, cy, d)
            if res is None:
                continue
            nx, ny = res
            nk = (nx, ny)
            if nk in visited:
                continue
            if nk in other_bodies:
                continue
            nwm = world_map.get(nk)
            if nwm is not None and len(nwm['kelp']) >= 3:
                continue
            rel = own_release_time.get(nk, 0)
            if rel > (ndist - 1 - p_delay):
                continue
            visited.add(nk)
            queue.append((nx, ny, first_dir, ndist, p_delay))

    return best_dir, best_target


# =============================================================================
# 7. TURN ORCHESTRATOR
# =============================================================================

def _make_trapped_sacrifice_move(pos: Position, vis_dragons: dict):
    here_tile = ct.get_tile(pos)
    if here_tile is None:
        ct.make_move(ct.get_dir())
        return

    # 1. Look for adjacent enemy head to take down with us
    for d in _DIRS:
        if here_tile.get_edge(d).get_edge_type() != EdgeType.KELP:
            dest = pos.add_dir(d)
            part = vis_dragons.get((dest.x, dest.y))
            if part is not None and part.get_team() != my_team and part.is_head():
                ct.make_move(d)
                return

    # 2. Otherwise self-destruct in-place WITHOUT moving (so we never hit an ally head!)
    # Not calling make_move / do_split leaves action = suicide (`died: no valid action`)!
    return


def execute_turn():
    global last_target, turn_start_clock

    turn_start_clock = time.perf_counter()
    rnd = game.get_round_num()
    pos = ct.get_position()
    my_len = ct.get_length()

    # 1. Update perception & sonar
    vis_dragons, vis_enemies, vis_enemy_any, own_release_time, own_body_set = update_perception(rnd)
    process_sonar(rnd)

    # 2. Role assignment
    role = determine_role(vis_enemies, vis_dragons)

    # 3. Check Endgame Sacrifice-Feed (Last 5% of rounds: Round 475..498)
    feed_action, feed_data = check_endgame_sacrifice_feed(rnd, role, vis_dragons, vis_enemy_any)
    if feed_action == 'DETONATE_NOW':
        # ZERO-COLLISION IN-PLACE SELF-DESTRUCT!
        # By returning without calling make_move or do_split, `end_turn()` finishes the turn
        # with no action (`died: no valid action`), crystallizing segments 0, 2, 4... into
        # ceil(L/2) pearls right in front of our superior ally / KING!
        ct.set_indicator_string(f"FEED_DETONATE->{feed_data} L:{my_len}")
        return

    ct.set_indicator_string(f"{role} L:{my_len} U:{ct.get_unit_count()}")

    # 4. Evaluate all passable and truly viable moves
    all_moves, viable_moves = evaluate_moves(vis_dragons, own_release_time, own_body_set, role)

    # 5. Check splitting (Emergency survival split OR strategic expansion split)
    split_size = check_split_decision(role, all_moves, viable_moves, vis_enemies)
    if split_size >= 2 and ct.can_split(split_size):
        ct.do_split(split_size)
        broadcast_sonar(rnd, role, vis_enemies, (pos.x, pos.y))
        return

    # 6. If no passable moves at all, sacrifice in-place (or kamikaze into adjacent enemy head)
    if not all_moves:
        _make_trapped_sacrifice_move(pos, vis_dragons)
        return

    # 7. If no viable moves exist, take the least dangerous passable move
    if not viable_moves:
        chosen = all_moves[0]
        ct.make_move(chosen['dir'])
        broadcast_sonar(rnd, role, vis_enemies, chosen['dest'])
        return

    # 8. Unified BFS targeting across only viable moves (including endgame King homing)
    approach_king_pos = feed_data if feed_action == 'APPROACH_AND_PING' else None
    best_dir, best_target = choose_best_move_via_bfs(
        viable_moves, vis_dragons, own_release_time, role, rnd,
        approach_king_pos=approach_king_pos
    )

    if best_dir is not None:
        last_target = best_target
        ct.make_move(best_dir)
        # If we are approaching the KING in endgame to feed next turn, ping our new position!
        next_pos = None
        if feed_action == 'APPROACH_AND_PING' and rnd >= 470:
            for vm in viable_moves:
                if vm['dir'] == best_dir:
                    next_pos = vm['dest']
                    break
        broadcast_sonar(rnd, role, vis_enemies, best_target, feed_pos=next_pos)
        return

    # 9. Fallback: safest viable move
    best_vm = viable_moves[0]
    cur_dir = ct.get_dir()
    chosen_vm = best_vm
    for vm in viable_moves:
        if vm['dir'] == cur_dir and vm['danger'] <= best_vm['danger'] + 5.0:
            chosen_vm = vm
            break

    last_target = chosen_vm['dest']
    ct.make_move(chosen_vm['dir'])
    broadcast_sonar(rnd, role, vis_enemies, chosen_vm['dest'])


# =============================================================================
# 8. MAIN ENTRYPOINT
# =============================================================================

def main():
    global ct, game
    ct, game = unswbc.init()

    while unswbc.update(ct, game):
        try:
            execute_turn()
        except Exception:
            here_tile = ct.get_tile(ct.get_position())
            moved = False
            if here_tile is not None:
                for d in _DIRS:
                    if here_tile.get_edge(d).get_edge_type() != EdgeType.KELP:
                        dest = ct.get_position().add_dir(d)
                        dt = ct.get_tile(dest)
                        if dt is None or dt.get_dragon() is None:
                            ct.make_move(d)
                            moved = True
                            break
            if not moved:
                ct.make_move(ct.get_dir())
        unswbc.end_turn()


if __name__ == "__main__":
    main()
