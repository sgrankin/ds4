def solve(payload, config):
    left = payload["left"]
    right = payload["right"]
    include = config["include_unmatched"]
    result = []
    for litem in left:
        lkey = litem["key"]
        lval = litem["value"]
        matched = False
        for ritem in right:
            if ritem["key"] == lkey:
                matched = True
                result.append({"key": lkey, "left": lval, "right": ritem["value"]})
        if not matched and include:
            result.append({"key": lkey, "left": lval, "right": None})
    return result
