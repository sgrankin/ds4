def solve(payload, config):
    capacity = config["capacity"]
    if capacity <= 0:
        return {"cache": [], "hits": 0, "misses": len(payload), "evictions": []}

    cache = []
    hits = 0
    misses = 0
    evictions = []

    for key in payload:
        found = False
        position = 0
        while position < len(cache):
            if cache[position] == key:
                found = True
                break
            position = position + 1

        if found:
            hits = hits + 1
            cache.pop(position)
            cache.append(key)
        else:
            misses = misses + 1
            cache.append(key)
            if len(cache) > capacity:
                evicted = cache.pop(0)
                evictions.append(evicted)

    return {"cache": cache, "hits": hits, "misses": misses, "evictions": evictions}
