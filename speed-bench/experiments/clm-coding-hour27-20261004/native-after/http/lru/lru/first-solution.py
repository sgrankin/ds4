def solve(payload, config):
    capacity = config
    if isinstance(config, dict):
        capacity = config.get("capacity", 0)
    try:
        capacity = int(capacity)
    except (TypeError, ValueError):
        capacity = 0

    hits = 0
    misses = 0
    evictions = []
    cache = []

    if capacity <= 0:
        for key in payload:
            misses += 1
        return {"cache": [], "hits": 0, "misses": misses, "evictions": []}

    for key in payload:
        found = -1
        for index in range(len(cache)):
            if cache[index] == key:
                found = index
                break
        if found >= 0:
            hits += 1
            cache.pop(found)
            cache.append(key)
        else:
            misses += 1
            cache.append(key)
            if len(cache) > capacity:
                evicted = cache.pop(0)
                evictions.append(evicted)

    return {"cache": cache, "hits": hits, "misses": misses, "evictions": evictions}
