#!/usr/bin/env python3
"""Join five fresh DSP1 pairs to native verification phases and stored trailers."""
import argparse
import json
import re
import statistics
import struct
from pathlib import Path


CASES = ("ordinary", "strict-tool", "schema-answer", "ordinary-short")


def output_value(response):
    """Compare actual content/arguments/reasoning, excluding generated identities."""
    output = []
    for item in response["output"]:
        value = {key: item[key] for key in ("type", "status", "name", "namespace", "content", "summary") if key in item}
        if "arguments" in item:
            value["arguments"] = item["arguments"]
        output.append(value)
    return output


def cache_inventory(directory):
    endpoints = []
    for path in sorted(directory.glob("*.manifest")):
        data = path.read_bytes()
        assert data[:8] == b"DS4SC01\0", path
        _, tokens, kind, profile_bytes = struct.unpack_from("<4I", data, 8)
        state_bytes = struct.unpack_from("<Q", data, 24)[0]
        nseg = struct.unpack_from("<I", data, 40)[0]
        trailers, native_bytes = [], 0
        for index in range(nseg):
            offset = 96 + profile_bytes + index * 64
            blob, size, position = struct.unpack_from("<IIQ", data, offset)
            native_bytes += size if blob == 0 else 0
            if blob != 0 or size != 32:
                continue
            key = data[offset + 16:offset + 56].decode("ascii")
            header = (directory / f"ds4sc-{key}.chunk").read_bytes()
            if header[:4] != b"DSP1":
                continue
            _, version, saved_position, count, width, *layers = struct.unpack("<8I", header)
            assert version == 1 and saved_position == tokens and count == min(tokens, 128) and width == 5120, path
            appendix_bytes = 32 + 3 * count * width * 2
            assert position + appendix_bytes == state_bytes, path
            trailers.append({"offset": position, "bytes": appendix_bytes, "rows": count, "layers": layers})
        assert native_bytes == state_bytes, path
        endpoints.append({"manifest": path.name, "tokens": tokens, "kind": kind,
                          "state_bytes": state_bytes, "DSP1": trailers})
    files = [path for path in directory.iterdir() if path.suffix in (".manifest", ".chunk")]
    return {"endpoints": endpoints, "file_bytes": sum(path.stat().st_size for path in files)}


def analyze_arm(directory):
    client = json.loads((directory / "results.json").read_text())
    by_name = {row["name"]: row for row in client}
    assert len(client) == len(by_name) == 7, directory
    phases, current = [], None
    for line in (directory / "server.stderr").read_text().splitlines():
        if "ds4-server: chat" in line and " prompt start" in line:
            assert current is None, line
            current = {"cycles": [], "cache_hits": [], "last_decode_log": 0}
        if current is None:
            continue
        cycle = re.search(r"DSpark proposed=(\d+) accepted=(\d+)", line)
        if cycle:
            current.setdefault("first_cycle_after_decode_log", current["last_decode_log"])
            current["cycles"].append(tuple(map(int, cycle.groups())))
        if "state cache exact hit" in line:
            current["cache_hits"].append(line)
        decode = re.search(r" gen=(\d+).* decoding .*?avg=([\d.]+) t/s", line)
        if decode:
            current.update(last_decode_log=int(decode[1]), decode_tps=float(decode[2]))
        finish = re.search(r" gen=(\d+).* finish=(\S+).*? ([\d.]+)s$", line)
        if finish:
            current.update(generated_tokens=int(finish[1]), finish=finish[2], server_seconds=float(finish[3]))
            phases.append(current)
            current = None
    trace = re.split(r"(?m)^===== request \d+ .*?=====\n", (directory / "trace.log").read_text())[1:]
    requests = sorted(directory.glob("*.request.json"), key=lambda path: path.stat().st_mtime_ns)
    assert current is None and len(phases) == len(trace) == len(requests) == 7, directory
    rows = []
    for phase, block, request in zip(phases, trace, requests):
        name = request.name.removesuffix(".request.json")
        measurement = by_name[name]
        body = json.loads(request.read_text())
        traced = json.JSONDecoder().raw_decode(block.split("--- raw request json ---", 1)[1].lstrip())[0]
        assert body == traced, request
        response = json.loads(request.with_name(f"{name}.response.json").read_text())
        usage = response["usage"]
        assert measurement["usage"] == usage and phase["generated_tokens"] == usage["output_tokens"], request
        cycles = phase.pop("cycles")
        credits = re.findall(r"state cache SSD credits=(\d+) cached=(\d+) prompt=(\d+)", block)
        if measurement["measured"]:
            assert usage["input_tokens_details"]["cached_tokens"] > 0, request
        rows.append(phase | measurement | {
            "request": name, "status": response["status"], "cap": body["max_output_tokens"],
            "output": output_value(response), "q_cycles": len(cycles),
            "proposed": sum(p for p, _ in cycles), "accepted": sum(a for _, a in cycles),
            "cycle_lengths": cycles, "trace_ssd_credits": [list(map(int, values)) for values in credits],
        })
    measured = [row for row in rows if row["measured"]]
    assert len(measured) == 4 and {row["name"] for row in measured} == {f"measure-{case}" for case in CASES}, directory
    return {"arm": directory.name, "command": json.loads((directory / "command.json").read_text()),
            "requests": rows, "cache": cache_inventory(directory / "cache")}


def distribution(values):
    return {"samples": values, "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "sample_stdev": statistics.stdev(values)}


def comparison(pairs):
    base = [row["base_seconds"] for row in pairs]
    features = [row["features_seconds"] for row in pairs]
    deltas = [100 * (new / old - 1) for old, new in zip(base, features)]
    return {"base_seconds": distribution(base), "features_seconds": distribution(features),
            "paired_change_percent": distribution(deltas),
            "change_of_means_percent": 100 * (statistics.mean(features) / statistics.mean(base) - 1),
            "features_faster_pairs": sum(new < old for old, new in zip(base, features)),
            "order_groups": {order: {"pairs": [row["pair"] for row in pairs if row["order"] == order],
                                      "mean_paired_change_percent": statistics.mean(delta for row, delta in zip(pairs, deltas) if row["order"] == order)}
                             for order in ("AB", "BA")}, "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.workspace / "manifest.json").read_text())
    assert manifest["pairs"] == 5 and manifest["order"] == ["AB", "BA", "AB", "BA", "AB"]
    arms = []
    for pair in range(1, 6):
        both = {arm: analyze_arm(args.workspace / f"pair-{pair}" / arm) for arm in ("base", "features")}
        assert any(endpoint["DSP1"] for endpoint in both["features"]["cache"]["endpoints"]), pair
        assert not any(endpoint["DSP1"] for endpoint in both["base"]["cache"]["endpoints"]), pair
        arms.append({"pair": pair, "order": manifest["order"][pair - 1], **both})
    cases = {}
    for case in CASES:
        pairs = []
        for arm in arms:
            before = next(row for row in arm["base"]["requests"] if row["name"] == f"measure-{case}")
            after = next(row for row in arm["features"]["requests"] if row["name"] == f"measure-{case}")
            pairs.append({"pair": arm["pair"], "order": arm["order"],
                          "base_seconds": before["seconds"], "features_seconds": after["seconds"],
                          "quality": {"base": before["quality"], "features": after["quality"]},
                          "output_equal": before["output"] == after["output"],
                          "tokens": {"base": before["usage"]["output_tokens"], "features": after["usage"]["output_tokens"]},
                          "reasoning_tokens": {name: row["usage"].get("output_tokens_details", {}).get("reasoning_tokens") for name, row in (("base", before), ("features", after))},
                          "q_cycles": {"base": before["q_cycles"], "features": after["q_cycles"]},
                          "proposed": {"base": before["proposed"], "features": after["proposed"]},
                          "accepted": {"base": before["accepted"], "features": after["accepted"]}})
        cases[case] = comparison(pairs)
    aggregate = []
    for pair in range(1, 6):
        complete = [cases[case]["pairs"][pair - 1] for case in CASES[:-1]]
        aggregate.append({"pair": pair, "order": manifest["order"][pair - 1],
                          "base_seconds": sum(row["base_seconds"] for row in complete),
                          "features_seconds": sum(row["features_seconds"] for row in complete)})
    result = {"manifest": manifest, "cases": cases, "complete_task_aggregate": comparison(aggregate),
              "all_quality_pass": all(row["quality"] for pair in arms for arm in ("base", "features") for row in pair[arm]["requests"]),
              "all_measured_outputs_equal": all(row["output_equal"] for case in cases.values() for row in case["pairs"]),
              "cache_file_bytes": {name: distribution([pair[name]["cache"]["file_bytes"] for pair in arms]) for name in ("base", "features")},
              "arms": arms}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
