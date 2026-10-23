#!/usr/bin/env python3
"""Real-model OpenResponses WebSocket qualification; run under pressure_run.py.

Requires websockets (the standard client library, not a custom frame parser).
Use --sdk to additionally exercise the official OpenAI Python client.
Raw requests, events, responses and server logs go only into --output.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
TERMINAL = {"response.completed", "response.failed", "response.incomplete"}
JSON_ONLY = "Follow the current request. Return only the requested JSON object, with no prose or Markdown."


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def response_text(response):
    return "".join(part.get("text", "") for item in response.get("output", [])
                   if item.get("type") == "message" for part in item.get("content", [])
                   if part.get("type") == "output_text").strip()


def stop(process):
    forced = False
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            forced = True
            process.kill()
            process.wait(timeout=5)
    return {"returncode": process.returncode if process is not None else None, "forced_kill": forced}


def http(base, output, name, method, path, payload=None, expected=200):
    save(output / (name + ".request.json"), {"method": method, "path": path, "body": payload})
    request = urllib.request.Request(base + path, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    try:
        with urllib.request.urlopen(request, timeout=180) as reply:
            status, raw = reply.status, reply.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    (output / (name + ".response.json")).write_bytes(raw)
    assert status == expected, (name, status, raw.decode(errors="replace"))
    return json.loads(raw)


class Server:
    def __init__(self, binary, model, output, store, cache, flint_profile=None):
        self.process = None
        self.output = output
        self.flint = model.is_dir() and (model / "flint-int4.json").is_file()
        output.mkdir()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        self.ws_url = f"ws://127.0.0.1:{port}/v1/responses"
        shaders = dict(re.findall(r'@\[@"(DS4_METAL_[A-Z0-9_]+_SOURCE)",\s*@"(src/metal/[^\"]+)"\]',
                                  (ROOT / "src/metal/ds4_metal.m").read_text()))
        env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4", "PT_"))}
        env.update({key: str(ROOT / value) for key, value in shaders.items()})
        env.update(DS4_SERVER_RESPONSE_STORE_DIR=str(store), DS4_STATE_CACHE_DIR=str(cache),
            DS4_LOCK_FILE=str(output / "server.lock"), DS4_STATE_CACHE_HOT_MB="0",
            DS4_STATE_CACHE_SSD_MB="512", DS4_SERVER_BATCH_LOG="1",
            DS4_SERVER_DECODE_COALESCE_US="2000", DS4_SERVER_CONTEXT_SUFFIX_REUSE="0",
            DS4_V41_DECODER_REPLAY_ROWS="0", DS4_V41_COMPRESSED_KV_F16="0",
            DS4_METAL_V41_RESIDENT_ROUTE_GAP="0", DS4_METAL_V41_RESIDENT_PREFILL_GAP="0",
            DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS="0")
        if self.flint:
            # Durable Responses JSON is independent of GPU-state caching.
            # Flint currently has its own in-memory KV layout, without the
            # DS4 SSD snapshot format or suffix relocation implementation.
            env = {key: value for key, value in env.items() if not key.startswith("DS4_STATE_CACHE_")}
        command = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx", "8192",
            "--prefill-chunk", "4096" if self.flint else "512", "--batched-session", "2",
            "--mtp-draft", "1", "--tokens", "1024" if self.flint else "128", "--trace", str(output / "trace.log")]
        if not self.flint:
            command += ["--ssd-streaming", "--ssd-streaming-cache-experts", "32GB", "--mixed-prefill-quantum", "512"]
        if flint_profile is not None:
            command += ["--flint-profile", flint_profile]
        save(output / "execution.json", {"command": command, "cwd": str(ROOT),
            "env": {k: v for k, v in env.items() if k.startswith("DS4")},
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()})
        try:
            with (output / "server.stdout").open("wb") as stdout, (output / "server.stderr").open("wb") as stderr:
                self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                stdout=stdout, stderr=stderr)
            started = time.monotonic()
            while time.monotonic() - started < 120:
                if self.process.poll() is not None:
                    raise RuntimeError(f"server startup exit {self.process.returncode}")
                try:
                    with urllib.request.urlopen(self.base + "/v1/models", timeout=1) as reply:
                        self.model_id = json.load(reply)["data"][0]["id"]
                    return
                except (OSError, ValueError, KeyError, IndexError):
                    time.sleep(.2)
            raise TimeoutError("120-second startup deadline")
        except BaseException:
            self.close()
            raise

    def close(self):
        shutdown = stop(self.process)
        save(self.output / "shutdown.json", shutdown)
        return not shutdown["forced_kill"] and shutdown["returncode"] == 0

    def create(self, input, **options):
        request = {"type": "response.create", "model": self.model_id,
            "temperature": 0, "reasoning": {"effort": "none"}, "max_output_tokens": 128,
            "store": False, "instructions": JSON_ONLY, "input": input} | options
        if self.flint:
            # Flint's packaged template does not expose a thinking-off switch.
            # Give its generated reasoning room without claiming effort=none.
            if "reasoning" not in options:
                request.pop("reasoning", None)
            request["max_output_tokens"] = max(1024, request["max_output_tokens"])
        return request


class Client:
    def __init__(self, server, name):
        from websockets.sync.client import connect
        self.log = (server.output / (name + ".events.jsonl")).open("w")
        self.events = []
        self.ws = connect(server.ws_url, additional_headers={"Authorization": "Bearer local"},
                          open_timeout=10, close_timeout=2, max_size=64 * 1024 * 1024, proxy=None)

    def record(self, direction, value):
        self.log.write(json.dumps({"direction": direction, "at": time.monotonic(), "value": value}) + "\n")
        self.log.flush()

    def send(self, value, fragmented=False):
        self.record("send", value)
        raw = json.dumps(value)
        self.ws.send([raw[:len(raw) // 2], raw[len(raw) // 2:]] if fragmented else raw)

    def receive(self):
        value = json.loads(self.ws.recv(timeout=180))
        self.events.append(value)
        self.record("receive", value)
        return value

    def complete(self):
        events = []
        while True:
            event = self.receive()
            events.append(event)
            if event["type"] == "error":
                raise AssertionError(event)
            if event["type"] in TERMINAL:
                response = event["response"]
                assert event["type"] == "response.completed" and response["status"] == "completed", response
                sequence = [item["sequence_number"] for item in events]
                assert all(isinstance(n, int) for n in sequence) and all(a < b for a, b in zip(sequence, sequence[1:])), sequence
                assert events[0]["type"] == "response.created", events[0]
                assert any(item["type"] == "response.in_progress" for item in events), events
                return response, events

    def error(self, code=None):
        event = self.receive()
        assert event["type"] == "error" and event["status"] >= 400, event
        assert isinstance(event["error"]["message"], str) and isinstance(event["error"]["code"], str), event
        if code is not None:
            assert event["error"]["code"] == code, event
        return event

    def close(self):
        self.ws.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--restart-server", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flint-profile", choices=("auto", "m4-pro", "m5-pro", "m5-max"),
                        help="Select a native Flint kernel profile for qualification")
    parser.add_argument("--sdk", action="store_true", help="Also test the installed official OpenAI Python client")
    args = parser.parse_args()
    args.server, args.model, args.output = args.server.resolve(), args.model.resolve(), args.output.resolve()
    args.restart_server = (args.restart_server or args.server).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"pass": False, "quality_pass": False, "checks": {}, "quality": [], "error": None}
    server = None

    def check(name, passed, detail=None):
        result["checks"][name] = bool(passed)
        save(args.output / "result.json", result)
        assert passed, (name, detail)
        print("PASS", name, flush=True)

    def grade(name, response, gold, required=True):
        actual = response_text(response)
        valid_json = True
        try:
            actual = json.loads(actual)
        except ValueError:
            valid_json = False
        passed = valid_json and actual == gold
        result["quality"].append({"name": name, "actual": actual, "expected": gold,
            "valid_json": valid_json, "required": required, "pass": passed})
        if required:
            check(name, passed, actual)
        else:
            save(args.output / "result.json", result)
            print("QUALITY", name, "PASS" if passed else "FAIL", json.dumps(actual), flush=True)
        return actual, valid_json

    try:
        server = Server(args.server, args.model, args.output / "before", args.output / "responses", args.output / "cache",
                        args.flint_profile)
        with Client(server, "sequential") as client:
            client.send(server.create('Return JSON {"ready":true}.', max_output_tokens=48), fragmented=True)
            basic, events = client.complete()
            grade("basic_answer", basic, {"ready": True})
            deltas = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
            check("deltas_match_final", deltas.strip() == response_text(basic))
            check("ping_pong", client.ws.ping(b"qualification").wait(timeout=5))

            fact = {"code": "EMBER-73", "count": 416}
            client.send(server.create('Remember code=EMBER-73 and count=416. Return JSON with only code and count.'))
            parent, _ = client.complete()
            grade("ephemeral_parent", parent, fact)
            client.send(server.create('Return JSON with the remembered code and count.', previous_response_id=parent["id"]))
            ephemeral, _ = client.complete()
            grade("incremental_store_false", ephemeral, fact)

            # A failed continuation must evict its referenced ephemeral parent.
            client.send(server.create([{"type": "function_call_output", "call_id": "call_openresponses_missing",
                "output": "No matching tool call exists."}], previous_response_id=ephemeral["id"]))
            client.error()
            client.send(server.create("Continue.", previous_response_id=ephemeral["id"]))
            client.error("previous_response_not_found")
            check("failed_continuation_evicts_parent", True)
            for field, value in [("stream_id", "unsupported"), ("generate", False)]:
                client.send(server.create("Reply with OK.", **{field: value}))
                client.error()
            client.send({"type": "response.cancel"})
            client.error()
            client.send(server.create('Return JSON {"recovered":true}.', max_output_tokens=48))
            recovered, _ = client.complete()
            grade("error_socket_recovery", recovered, {"recovered": True})

        with Client(server, "ephemeral-reconnect") as client:
            client.send(server.create("Continue.", previous_response_id=recovered["id"]))
            client.error("previous_response_not_found")
            check("store_false_not_recoverable_after_disconnect", True)
        http(server.base, server.output, "ephemeral-not-stored", "GET", "/v1/responses/" + recovered["id"], expected=404)
        check("store_false_not_on_disk", True)

        # Two sockets exercise concurrency while each socket remains sequential.
        barrier = threading.Barrier(2)
        def concurrent(side):
            with Client(server, "concurrent-" + side) as client:
                barrier.wait(timeout=10)
                started = time.monotonic()
                client.send(server.create(f'Return JSON with side="{side}" and audit equal to the integer array 0 through 15.'))
                response, _ = client.complete()
                return side, response, (started, time.monotonic())
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(concurrent, ("A", "B")))
        for side, response, _ in rows:
            grade("concurrent_" + side, response, {"side": side, "audit": list(range(16))})
        check("concurrent_request_intervals_overlap", max(row[2][0] for row in rows) < min(row[2][1] for row in rows))

        # Facts are absent from the prompt: the client must really read storage.
        archive = args.output / "permanent-record.json"
        stored = {"code": "SABLE-34", "count": 947}
        save(archive, stored)
        tool = {"type": "function", "name": "lookup_record", "description": "Read the permanent record for a key.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}
        with Client(server, "tools") as client:
            client.send(server.create('Call lookup_record with key="current". Do not guess the record. After the tool result, return JSON with its code and count.',
                tools=[tool], instructions="Use the provided tool to read the permanent record before answering.", max_output_tokens=192))
            call_response, _ = client.complete()
            calls = [item for item in call_response["output"] if item["type"] == "function_call"]
            check("function_call_emitted", len(calls) == 1, call_response)
            call = calls[0]
            check("function_arguments", call["name"] == "lookup_record" and json.loads(call["arguments"]) == {"key": "current"}, call)
            tool_output = archive.read_text()
            save(server.output / "tool-read.json", {"path": str(archive), "sha256": hashlib.sha256(tool_output.encode()).hexdigest(), "reads": 1})
            client.send(server.create([{"type": "function_call_output", "call_id": call["call_id"], "output": tool_output}],
                previous_response_id=call_response["id"], tools=[tool]))
            answer, _ = client.complete()
            grade("tool_result_continuation", answer, stored)

        conversation = http(server.base, server.output, "conversation-create", "POST", "/v1/conversations", {})
        durable_fact = {"code": "RIVER-28", "count": 632}
        with Client(server, "durable") as client:
            client.send(server.create('Remember code=RIVER-28 and count=632. Return JSON with only code and count.',
                store=True, conversation=conversation["id"]))
            durable, _ = client.complete()
            grade("durable_parent", durable, durable_fact)
        with Client(server, "durable-reconnect") as client:
            client.send(server.create("Return JSON with the remembered code and count.", store=True, previous_response_id=durable["id"]))
            continued, _ = client.complete()
            grade("durable_reconnect", continued, durable_fact)

        # Keep the exact-answer observation separate from socket survival. An
        # ordinary control and fresh-socket retry bound the quality comparison.
        survivor_prompt = 'Return JSON {"survived":true}.'
        with Client(server, "disconnect-control") as client:
            client.send(server.create(survivor_prompt, max_output_tokens=48))
            control, _ = client.complete()
            grade("disconnect_control", control, {"survived": True}, required=False)

        # Close after an actual generated delta; another connection must survive.
        victim = Client(server, "disconnect-victim")
        try:
            victim.send(server.create("Write the integers from 1 through 1000, one per line, without skipping any.",
                instructions="Follow the user's request.", max_output_tokens=1024,
                **({"reasoning": {"summary": "detailed"}} if server.flint else {})))
            while True:
                event = victim.receive()
                assert event["type"] not in TERMINAL and event["type"] != "error", event
                observable = event["type"] == "response.output_text.delta" or (
                    server.flint and event["type"] == "response.reasoning_summary_text.delta")
                if observable and event.get("delta"):
                    break
            submitted = threading.Event()
            def surviving():
                with Client(server, "disconnect-survivor") as client:
                    client.send(server.create(survivor_prompt, max_output_tokens=48))
                    submitted.set()
                    return client.complete()[0]
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(surviving)
                assert submitted.wait(timeout=10)
                victim.close()
                victim = None
                survivor = future.result(timeout=180)
                check("disconnect_survivor_completed", survivor["status"] == "completed")
                _, valid_json = grade("disconnect_survivor", survivor, {"survived": True}, required=False)
                check("disconnect_survivor_valid_json", valid_json)
            check("closed_during_generation", True)
        finally:
            if victim is not None:
                victim.close()

        with Client(server, "disconnect-retry") as client:
            client.send(server.create(survivor_prompt, max_output_tokens=48))
            retry, _ = client.complete()
            grade("disconnect_retry", retry, {"survived": True}, required=False)

        if args.sdk:
            from openai import OpenAI
            sdk = OpenAI(base_url=server.base + "/v1", api_key="local")
            with sdk.responses.connect() as connection:
                connection.response.create(model=server.model_id, store=False, temperature=0,
                    max_output_tokens=1024 if server.flint else 48, input='Return JSON {"sdk":true}.',
                    **({} if server.flint else {"reasoning": {"effort": "none"}}))
                for event in connection:
                    if event.type == "error":
                        raise AssertionError(event.model_dump())
                    if event.type in TERMINAL:
                        sdk_response = event.response.model_dump(mode="json")
                        save(server.output / "sdk-response.json", sdk_response)
                        grade("official_sdk", sdk_response, {"sdk": True})
                        break

        if server.flint:
            # HF tokenizer reference: this raw input is 4,834 tokens, before
            # role/instruction overhead. Usage is authoritative, so the test
            # fails if a template/tokenizer change makes it a single chunk.
            long_fact = {"code": "SOLSTICE-92", "count": 583}
            notes = "Permanent record: code=SOLSTICE-92 and count=583. Keep this record unchanged.\n"
            notes += "".join(f"Journal note {i:03d}: routine background; no changes to the permanent record.\n"
                             for i in range(320))
            notes += "\nReturn one concise JSON object containing only the permanent record code and count."
            with Client(server, "long-context-continuation") as client:
                client.send(server.create(notes))
                long_parent, _ = client.complete()
                input_tokens = long_parent["usage"]["input_tokens"]
                check("long_context_prefill_exceeds_4096", input_tokens > 4096, long_parent["usage"])
                grade("long_context_parent", long_parent, long_fact)
                client.send(server.create("Return JSON with the same permanent record code and count.",
                                          previous_response_id=long_parent["id"]))
                long_continued, _ = client.complete()
                grade("long_context_after_decode_continuation", long_continued, long_fact)
                usage = long_continued["usage"]
                cached = usage.get("input_tokens_details", {}).get("cached_tokens")
                result["long_context"] = {"parent_input_tokens": input_tokens,
                    "continuation_input_tokens": usage["input_tokens"], "continuation_cached_tokens": cached}
                if cached is not None:
                    check("long_context_continuation_cache_hit", cached > 4096, usage)

        check("first_server_shutdown", server.close())
        first_logs = (server.output / "server.stderr").read_text(errors="replace")
        counts = [int(n) for n in re.findall(r"decode batch count=(\d+)", first_logs)]
        result["decode_batch_counts"] = counts
        if not server.flint:
            check("coordinator_decode_observed", bool(counts))
        else:
            # Flint uses serialized resident packets, rather than DS4's
            # multi-session decode-batch kernel. Socket overlap and completed
            # responses above still qualify the concurrent conversation path.
            result["decode_execution"] = "serialized_resident_packets"
        server = Server(args.restart_server, args.model, args.output / "after", args.output / "responses", args.output / "cache",
                        args.flint_profile)
        with Client(server, "restart") as client:
            client.send(server.create("Return JSON with the remembered code and count.", store=True, previous_response_id=continued["id"]))
            after, _ = client.complete()
            grade("response_restart_recovery", after, durable_fact)
            client.send(server.create("Return JSON with the remembered code and count.", store=True, conversation=conversation["id"]))
            after, _ = client.complete()
            grade("conversation_restart_recovery", after, durable_fact)
        check("restart_server_shutdown", server.close())
        server = None
        result["pass"] = all(result["checks"].values())
    except Exception as error:
        result["error"] = repr(error)
    finally:
        if server is not None:
            server.close()
        result["quality_pass"] = bool(result["quality"]) and all(item["pass"] for item in result["quality"])
        save(args.output / "result.json", result)
    print(json.dumps({"pass": result["pass"], "quality_pass": result["quality_pass"],
        "checks": len(result["checks"]), "error": result["error"]}), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
