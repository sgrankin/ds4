def merge_intervals(intervals):
    """Merge overlapping or touching closed integer intervals.

    Returns a new sorted list of [start, end] pairs. The input list and its
    inner pairs are not mutated.
    """
    pairs = sorted((start, end) for start, end in intervals)
    merged = []
    for start, end in pairs:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged
