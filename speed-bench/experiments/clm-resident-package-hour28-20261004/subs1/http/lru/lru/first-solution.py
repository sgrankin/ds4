def solve(payload, config):
    capacity = config["capacity"]
    if capacity <= 0:
        return {"cache": [], "hits": 0, "misses": len(payload), "evictions": []}

    cache = {}
    hits = 0
    misses = 0
    evictions = []

    for key in payload:
        if key in cache:
            hits += 1
            del cache[key]
            cache[key] = True
        else:
            misses += 1
            cache[key] = True
            if len(cache) > capacity:
                lru = next(iter(cache))
                del cache[lru]
                evictions.append(lru)

    return {
        "cache": list(cache.keys()),
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
    }
