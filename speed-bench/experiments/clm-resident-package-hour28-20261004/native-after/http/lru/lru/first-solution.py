def solve(payload, config):
    capacity = config["capacity"]
    hits = 0
    misses = 0
    evictions = []
    cache = []

    if capacity <= 0:
        misses = len(payload)
        return {"cache": cache, "hits": hits, "misses": misses, "evictions": evictions}

    for key in payload:
        found = False
        for i in range(len(cache)):
            if cache[i] == key:
                found = True
                cache.pop(i)
                cache.append(key)
                break
        if found:
            hits += 1
        else:
            misses += 1
            cache.append(key)
            if len(cache) > capacity:
                evicted = cache.pop(0)
                evictions.append(evicted)

    return {"cache": cache, "hits": hits, "misses": misses, "evictions": evictions}
