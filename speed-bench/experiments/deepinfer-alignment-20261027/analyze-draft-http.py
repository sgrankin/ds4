#!/usr/bin/env python3
"""Join the fixed item4 HTTP requests with draft cycles and native SSD headers."""
import json
import re
import struct
import sys
from pathlib import Path


def analyze(arm):
    measured = {}
    for result_file in arm.glob("*/results.json"):
        for row in json.loads(result_file.read_text()):
            name = row["name"] if "name" in row else f"cap-{row['cap']}"
            measured[f"{result_file.parent.name}/{name}"] = row
    phases, current = [], None
    for line in (arm / "server.stderr").read_text().splitlines():
        if "ds4-server: chat" in line and " prompt start" in line:
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
    trace = re.split(r"(?m)^===== request \d+ .*?=====\n", (arm / "trace.log").read_text())[1:]
    requests = sorted(arm.glob("*/*.request.json"), key=lambda path: path.stat().st_mtime_ns)
    assert len(phases) <= len(trace) <= len(requests)
    rows = []
    for phase, block, request in zip(phases, trace, requests):
        body = json.loads(request.read_text())
        traced = json.JSONDecoder().raw_decode(block.split("--- raw request json ---", 1)[1].lstrip())[0]
        assert body == traced, request
        response_file = request.with_name(request.name.replace(".request.json", ".response.json"))
        if response_file.exists():
            response = json.loads(response_file.read_text())
        else:
            events = json.loads(request.with_name("stream.events.json").read_text())
            response = next(event["response"] for event in events if event["type"] == "response.completed")
        usage = response["usage"]
        assert phase["generated_tokens"] == usage["output_tokens"], request
        name = str(request.relative_to(arm)).replace(".request.json", "")
        measurement = measured.get(name, {})
        if measurement:
            assert measurement["usage"] == usage, request
        cycles = phase.pop("cycles")
        credits = re.findall(r"state cache SSD credits=(\d+) cached=(\d+) prompt=(\d+)", block)
        rows.append(phase | {"request": name, "client_seconds": measurement.get("seconds"),
                    "quality": measurement.get("quality"),
                    "temperature": body["temperature"], "cap": body["max_output_tokens"],
                    "status": response["status"], "usage": usage,
                    "q_cycles": len(cycles), "proposed": sum(p for p, _ in cycles),
                    "accepted": sum(a for _, a in cycles), "cycle_lengths": cycles,
                    "trace_ssd_credits": [list(map(int, values)) for values in credits]})
    endpoints = []
    for path in sorted((arm / "cache").glob("*.manifest")):
        data = path.read_bytes()
        assert data[:8] == b"DS4SC01\0"
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
            header = (arm / "cache" / f"ds4sc-{key}.chunk").read_bytes()
            if header[:4] == b"DSP1":
                _, version, saved_position, count, width, *layers = struct.unpack("<8I", header)
                assert version == 1 and saved_position == tokens and count == min(tokens, 128) and width == 5120
                appendix_bytes = 32 + 3 * count * width * 2
                assert position + appendix_bytes == state_bytes
                trailers.append({"offset": position, "bytes": appendix_bytes, "rows": count, "layers": layers})
        assert native_bytes == state_bytes
        endpoints.append({"manifest": path.name, "tokens": tokens, "kind": kind,
                          "state_bytes": state_bytes, "DSP1": trailers})
    return {"arm": arm.name, "completed_requests": rows, "unfinished_request": current is not None,
            "endpoints": endpoints,
            "physical_cache_bytes": sum(path.stat().st_size for path in (arm / "cache").iterdir() if path.suffix in (".manifest", ".chunk"))}


workspace = Path(sys.argv[1])
print(json.dumps([analyze(workspace / arm) for arm in ("http-off", "http-on")], indent=2))
