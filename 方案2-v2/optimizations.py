"""Per-dragon, per-turn cache shared by native training and sandbox deployment."""
def install_graph_cache(rules):
    original = rules.step_tile_no_blind_portal
    cache = {}
    def step(x, y, direction):
        key = (x, y, direction)
        if key not in cache:
            cache[key] = original(x, y, direction)
        return cache[key]
    rules.step_tile_no_blind_portal = step
    return cache.clear
