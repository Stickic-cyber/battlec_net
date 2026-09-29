"""Shared training/deployment observation encoder and rules/network boundary."""
import math

FEATURE_VERSION = 2
GLOBAL_NAMES = (
    "round", "length", "units", "width", "height", "body_known_fraction",
    "farmer", "king", "hunter", "feeder", "enemy_heads", "ally_heads",
    "visible_pearls", "king_report_fresh", "king_length", "feed_beacons",
    "map_known_fraction", "rounds_until_endgame", "endgame", "king_length_gap",
)
LOCAL_NAMES = (
    "available", "pearl", "spawn_enabled", "spawn_time", "danger", "space",
    "exits", "rule_viable", "forward", "left", "right", "reverse",
    "nearest_pearl", "pearl_progress", "nearest_enemy", "nearest_ally",
    "king_distance", "king_progress", "last_target_distance",
    "adjacent_kelp", "adjacent_occupied", "memory_age",
    "search_valid", "search_score", "search_gap", "search_distance", "search_dx", "search_dy",
    "space_length_ratio", "portal", "pearl_present", "enemy_present", "ally_present", "last_target_present",
)
FEATURE_NAMES = GLOBAL_NAMES + LOCAL_NAMES
FEATURE_DIM = len(FEATURE_NAMES)


class Decision:
    __slots__ = ("features", "mask", "teacher", "candidates", "role", "enemies", "target", "feeding")

    def __init__(self, features, mask, teacher, candidates, role, enemies, target, feeding):
        self.features, self.mask, self.teacher = features, mask, teacher
        self.candidates, self.role, self.enemies = candidates, role, enemies
        self.target, self.feeding = target, feeding


def distance(r, a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return min(dx, r.map_width - dx) + min(dy, r.map_height - dy)


def min_distance(r, point, positions):
    return min((distance(r, point, q) for q in positions), default=16)


def features(r, moves, role, visible, enemies, approach):
    ct, game = r.ct, r.game
    rnd, length = game.get_round_num(), ct.get_length()
    pos = ct.get_position()
    here = (pos.x, pos.y)
    allies = [p for p, part in visible.items()
              if part.get_team() == r.my_team and part.get_id() != ct.get_id() and part.is_head()]
    pearls = [(t.get_position().x, t.get_position().y) for t in ct.get_tiles() if t.has_pearl()]
    king_fresh = rnd - r.known_king_info["round"] <= 5 and r.known_king_info["x"] >= 0
    king = approach or ((r.known_king_info["x"], r.known_king_info["y"]) if king_fresh else None)
    global_values = [rnd / 499, min(length, 256) / 128, ct.get_unit_count() / 64,
                     r.map_width / 64, r.map_height / 64,
                     min(1, len(r.own_body) / max(1, length)),
                     *[float(role == name) for name in ("FARMER", "KING", "HUNTER", "FEEDER")],
                     min(len(enemies), 16) / 16, min(len(allies), 16) / 16,
                     len(pearls) / 49, float(king_fresh),
                     min(r.known_king_info["len"], 256) / 128 if king_fresh else 0,
                     min(len(r.feed_beacons), 8) / 8,
                     len(r.world_map) / (r.map_width * r.map_height),
                     max(0, 460 - rnd) / 460, float(rnd >= 460),
                     math.tanh((r.known_king_info["len"] - length) / 32) if king_fresh else 0]
    by_index = {r._DIRS.index(m["dir"]): m for m in moves}
    best_search = max((m.get("search_score", -1000) for m in moves), default=-1000)
    result, mask = [], []
    for i, direction in enumerate(r._DIRS):
        move = by_index.get(i)
        # No unknown portal exits in policy actions. Emergency rule fallback is separate.
        safe = move is not None and ct.get_tile(r.Position(*move["dest"])) is not None
        if safe:
            tile_here = ct.get_tile(pos)
            if tile_here.get_edge(direction).get_edge_type() == r.EdgeType.PORTAL:
                safe = (pos.x, pos.y, direction) in r.portal_dest_ready
            dest_tile = ct.get_tile(r.Position(*move["dest"]))
            safe = safe and dest_tile.get_dragon() is None
        mask.append(bool(safe))
        if not safe:
            result.append(global_values + [0.0] * len(LOCAL_NAMES))
            continue
        dest = move["dest"]
        tile = ct.get_tile(r.Position(*dest))
        pearl_dist = min_distance(r, dest, pearls)
        king_dist = distance(r, dest, king) if king else 16
        kelp, occupied = 0, 0
        for d in r._DIRS:
            kelp += tile.get_edge(d).get_edge_type() == r.EdgeType.KELP
            adjacent = ct.get_tile(tile.get_position().add_dir(d))
            occupied += adjacent is not None and adjacent.get_dragon() is not None
        eta = move["pearl_time"]
        search = move.get("search_score", -1000)
        search_valid = search > -999
        tx, ty = move.get("search_target", dest)
        target_dx = ((tx - here[0] + r.map_width // 2) % r.map_width) - r.map_width // 2
        target_dy = ((ty - here[1] + r.map_height // 2) % r.map_height) - r.map_height // 2
        local = [1.0, float(move["has_pearl"]), float(eta >= 0),
                 min(max(eta, 0), 100) / 100,
                 math.tanh(move["danger"] / 100), min(move["area"], 48) / 48,
                 move["exit_deg"] / 4, float(move["viable"]),
                 float(direction == ct.get_dir()), float(direction == ct.get_dir().get_left()),
                 float(direction == ct.get_dir().get_right()), float(direction == ct.get_dir().get_opposite()),
                 min(pearl_dist, 16) / 16,
                 max(-1, min(1, min_distance(r, here, pearls) - pearl_dist)),
                 min(min_distance(r, dest, enemies), 16) / 16,
                 min(min_distance(r, dest, allies), 16) / 16,
                 min(king_dist, 16) / 16,
                 max(-1, min(1, distance(r, here, king) - king_dist)) if king else 0,
                 min(distance(r, dest, r.last_target), 16) / 16 if r.last_target else 1,
                 kelp / 4, occupied / 4,
                 min(20, rnd - r.world_map.get(dest, {"round": rnd})["round"]) / 20,
                 float(search_valid), math.tanh(search / 150) if search_valid else -1,
                 max(-1, (search - best_search) / 100) if search_valid else -1,
                 min(move.get("search_distance", 0), 32) / 32,
                 target_dx / max(1, r.map_width / 2), target_dy / max(1, r.map_height / 2),
                 min(2, move["area"] / max(1, length)) / 2,
                 float(tile_here.get_edge(direction).get_edge_type() == r.EdgeType.PORTAL),
                 float(bool(pearls)), float(bool(enemies)), float(bool(allies)), float(r.last_target is not None)]
        result.append(global_values + local)
    assert all(len(row) == FEATURE_DIM for row in result)
    return result, mask, by_index


def prepare(r):
    """Execute forced/rule-only branches; return a movement decision otherwise."""
    if not hasattr(r, "_clear_graph_cache"):
        from optimizations import install_graph_cache
        r._clear_graph_cache = install_graph_cache(r)
    r._clear_graph_cache()
    r.turn_start_clock = r.time.perf_counter()
    rnd = r.game.get_round_num()
    pos = r.ct.get_position()
    visible, enemies, enemy_any, release, own = r.update_perception(rnd)
    r.process_sonar(rnd)
    role = r.determine_role(enemies, visible)
    feed_action, feed_data = r.check_endgame_sacrifice_feed(rnd, role, visible, enemy_any)
    if feed_action == "DETONATE_NOW":
        r.ct.set_indicator_string("RULE_FEED")
        return None
    moves, viable = r.evaluate_moves(visible, release, own, role)
    split = r.check_split_decision(role, moves, viable, enemies)
    if split >= 2 and r.ct.can_split(split):
        r.ct.do_split(split)
        r.broadcast_sonar(rnd, role, enemies, (pos.x, pos.y))
        return None
    if not moves:
        r._make_trapped_sacrifice_move(pos, visible)
        return None
    if not viable:
        chosen = moves[0]
        r.ct.make_move(chosen["dir"])
        r.broadcast_sonar(rnd, role, enemies, chosen["dest"])
        return None
    approach = feed_data if feed_action == "APPROACH_AND_PING" else None
    teacher_dir, target = r.choose_best_move_via_bfs(viable, visible, release, role, rnd, approach)
    if teacher_dir is None:
        chosen = viable[0]
        for move in viable:
            if move["dir"] == r.ct.get_dir() and move["danger"] <= viable[0]["danger"] + 5:
                chosen = move
                break
        teacher_dir, target = chosen["dir"], chosen["dest"]
    values, mask, candidates = features(r, moves, role, visible, enemies, approach)
    teacher = r._DIRS.index(teacher_dir)
    if not mask[teacher]:
        # Refuse to create a bogus supervised label or off-policy training record.
        raise RuntimeError("Rule teacher selected a masked movement")
    decision = Decision(values, mask, teacher, candidates, role, enemies, target,
                        feed_action == "APPROACH_AND_PING" and rnd >= 470)
    if sum(mask) == 1:
        commit(r, decision, teacher, "FORCED")
        return None
    return decision


def commit(r, decision, action, source="NET"):
    if not 0 <= action < 4 or not decision.mask[action]:
        raise ValueError("Policy selected masked action")
    chosen = decision.candidates[action]
    # Both rule-control and network variants use identical communication semantics.
    target = decision.target if action == decision.teacher else chosen["dest"]
    r.last_target = target
    r.ct.make_move(chosen["dir"])
    r.ct.set_indicator_string(f"{source} {decision.role} L:{r.ct.get_length()}")
    r.broadcast_sonar(r.game.get_round_num(), decision.role, decision.enemies, target,
                      feed_pos=chosen["dest"] if decision.feeding else None)
