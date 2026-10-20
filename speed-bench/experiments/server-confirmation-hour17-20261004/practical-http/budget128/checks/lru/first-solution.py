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
        index = 0
        while index < len(cache):
            if cache[index] == key:
                found = True
                break
            index = index + 1

        if found:
            hits = hits + 1
            cache.pop(index)
            cache.append(key)
        else:
            misses = misses + 1
            cache.append(key)
            if len(cache) > capacity:
                evicted = cache.pop(0)
                evictions.append(evicted)

    return {"cache": cache, "hits": hits, "misses": misses, "evictions": evictions}
