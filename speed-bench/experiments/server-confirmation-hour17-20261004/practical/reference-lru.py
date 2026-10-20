def solve(payload, config):
    capacity = config['capacity']
    if capacity <= 0:
        return {'cache':[], 'hits':0, 'misses':len(payload), 'evictions':[]}
    cache = []
    evictions = []
    hits = 0
    misses = 0
    for key in payload:
        if key in cache:
            hits += 1
            cache.remove(key)
        else:
            misses += 1
        cache.append(key)
        if len(cache) > capacity:
            evictions.append(cache.pop(0))
    return {'cache':cache, 'hits':hits, 'misses':misses, 'evictions':evictions}
