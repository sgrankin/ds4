def solve(payload, config):
    left_items = payload["left"]
    right_items = payload["right"]
    include_unmatched = config["include_unmatched"]
    result = []
    for left_item in left_items:
        left_key = left_item["key"]
        left_value = left_item["value"]
        matched = False
        for right_item in right_items:
            if right_item["key"] == left_key:
                matched = True
                result.append({
                    "key": left_key,
                    "left": left_value,
                    "right": right_item["value"]
                })
        if not matched and include_unmatched:
            result.append({
                "key": left_key,
                "left": left_value,
                "right": None
            })
    return result
