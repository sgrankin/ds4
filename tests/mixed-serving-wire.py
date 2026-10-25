#!/usr/bin/env python3
"""Wire-only mixed-engine probe, invoked by mixed-serving.nu against a live server.

Reuse the existing WebSocket client/event checks; production has no Python link.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
import urllib.parse

from test_responses_websocket import Client, Server, TERMINAL, http, response_text, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("main_model")
    p.add_argument("helper_model")
    p.add_argument("output", type=Path)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timing-only", action="store_true", help="Measure the main engine alone, including an older server without helper support")
    p.add_argument("--mixed-timing-only", action="store_true", help="Stop after the isolated/mixed latency screen")
    p.add_argument("--helper-repeat", type=int, default=1, help="Repeat the helper state to screen long-prefill scheduling")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    flint = args.main_model in {"flint", "flint-int4"}
    server = SimpleNamespace(base=args.url.rstrip("/"),
        ws_url=args.url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/v1/responses",
        model_id=args.main_model, flint=flint, output=args.output)
    server.create = lambda text, **options: Server.create(server, text, **options)
    result = {"pass": False, "checks": {}, "timings": {}, "error": None}

    def check(name, condition, detail=None):
        result["checks"][name] = bool(condition)
        save(args.output / "result.json", result)
        assert condition, (name, detail)
        print("PASS", name, flush=True)

    def payload(state="A customer was billed twice and needs a refund.", question="Which team should handle the request?"):
        evidence = state if args.helper_repeat == 1 else (state + "\n") * args.helper_repeat
        return {"model": args.helper_model, "state": evidence, "questions": {"team": {
            "type": "choice", "instructions": question,
            "criteria": {"billing": "Payments, charges and refunds", "technical": "Broken software"}}}}

    def decide(name, request=None):
        start = time.monotonic()
        response = http(server.base, args.output, name, "POST", "/v1/systemone", request or payload())
        return response, (start, time.monotonic())

    def generate(name, text='Return JSON {"ready":true}.', **options):
        with Client(server, name) as client:
            start = time.monotonic()
            client.send(server.create(text, max_output_tokens=48, **options))
            response, _ = client.complete()
            return response, (start, time.monotonic())

    if args.timing_only:
        generate("timing-warm")
        samples = []
        for i in range(args.repeats):
            answer, interval = generate(f"timing-{i}")
            check(f"timing_{i}_answer", json.loads(response_text(answer)) == {"ready": True}, answer)
            samples.append(interval[1] - interval[0])
        result["timings"]["main_seconds"] = samples
        result["pass"] = all(result["checks"].values())
        save(args.output / "result.json", result)
        print(json.dumps(result), flush=True)
        return

    try:
        models = http(server.base, args.output, "models", "GET", "/v1/models")
        ids = [m["id"] for m in models["data"]]
        check("both_models_discovered", args.main_model in ids and args.helper_model in ids, ids)
        http(server.base, args.output, "unloaded-model", "GET", "/v1/models/missing-model", expected=404)
        http(server.base, args.output, "unloaded-known-model", "GET", "/v1/models/qwen3.8-flash-next", expected=404)
        wrong = http(server.base, args.output, "helper-on-responses", "POST", "/v1/responses",
            {"model": args.helper_model, "input": "Reply OK.", "max_output_tokens": 4}, expected=400)
        check("helper_not_routed_to_main", "error" in wrong)
        wrong = http(server.base, args.output, "main-on-decisions", "POST", "/v1/systemone",
            payload() | {"model": args.main_model}, expected=404)
        check("main_not_routed_to_helper", "error" in wrong)

        # Warm each engine before measuring isolated and overlapping requests.
        answer, _ = generate("warm-main")
        check("warm_main_answer", json.loads(response_text(answer)) == {"ready": True}, answer)
        answer, _ = decide("warm-helper")
        check("warm_helper_answer", answer["answers"]["team"]["choice"] == "billing", answer)
        for mode in ("isolated", "mixed"):
            timings = []
            for i in range(args.repeats):
                if mode == "isolated":
                    main_reply, main_interval = generate(f"{mode}-{i}-main")
                    helper_reply, helper_interval = decide(f"{mode}-{i}-helper")
                else:
                    barrier = threading.Barrier(2)
                    def request_main():
                        barrier.wait(timeout=10)
                        return generate(f"{mode}-{i}-main")
                    def request_helper():
                        barrier.wait(timeout=10)
                        return decide(f"{mode}-{i}-helper")
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        a, b = pool.submit(request_main), pool.submit(request_helper)
                        main_reply, main_interval = a.result(timeout=180)
                        helper_reply, helper_interval = b.result(timeout=180)
                    check(f"mixed_{i}_intervals_overlap", max(main_interval[0], helper_interval[0]) < min(main_interval[1], helper_interval[1]))
                check(f"{mode}_{i}_main_answer", json.loads(response_text(main_reply)) == {"ready": True}, main_reply)
                check(f"{mode}_{i}_helper_answer", helper_reply["answers"]["team"]["choice"] == "billing", helper_reply)
                timings.append({"main_seconds": main_interval[1]-main_interval[0], "helper_seconds": helper_interval[1]-helper_interval[0],
                    "helper_input_tokens": helper_reply["usage"]["input_tokens"], "main_output_tokens": main_reply["usage"]["output_tokens"]})
            result["timings"][mode] = timings
            save(args.output / "result.json", result)

        if args.mixed_timing_only:
            result["pass"] = all(result["checks"].values())
            print(json.dumps(result), flush=True)
            return

        # Separate requests must retain their own state and learned-head scratch.
        barrier = threading.Barrier(2)
        def separate_question(name, state):
            barrier.wait(timeout=10)
            return decide(name, payload(state))[0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(separate_question, "separate-billing", "The payment was charged twice. Refund the duplicate charge.")
            b = pool.submit(separate_question, "separate-technical", "The application crashes on launch with a software exception.")
            billing, technical = a.result(timeout=180), b.result(timeout=180)
        check("simultaneous_distinct_questions", billing["answers"]["team"]["choice"] == "billing" and technical["answers"]["team"]["choice"] == "technical")

        with Client(server, "ancestry") as client:
            client.send(server.create('Remember code=ASTER-51. Return JSON {"code":"ASTER-51"}.'))
            parent, _ = client.complete()
            check("ancestry_parent", json.loads(response_text(parent)) == {"code": "ASTER-51"})
            decide("between-turns-helper")
            client.send(server.create('Return JSON containing only the remembered code.', previous_response_id=parent["id"]))
            child, _ = client.complete()
            check("ancestry_survives_helper", json.loads(response_text(child)) == {"code": "ASTER-51"})
            client.send(server.create("Continue.", model=args.helper_model, previous_response_id=child["id"]))
            client.error()
            check("ancestry_cannot_switch_to_helper", True)

        # Disconnect a real generating socket while a helper request is active.
        victim = Client(server, "cancel-main")
        try:
            victim.send(server.create("Write the integers from 1 through 1000, one per line.",
                instructions="Follow the request.", max_output_tokens=1024,
                **({"reasoning": {"summary": "detailed"}} if flint else {})))
            while True:
                event = victim.receive()
                assert event["type"] not in TERMINAL and event["type"] != "error", event
                if event["type"].endswith(".delta") and event.get("delta"):
                    break
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(decide, "helper-during-main-disconnect")
                victim.close()
                victim = None
                answer, _ = future.result(timeout=180)
                check("helper_survives_main_disconnect", answer["answers"]["team"]["choice"] == "billing")
        finally:
            if victim is not None:
                victim.close()

        # Closing a decision socket during a long prompt must retire its work.
        target = urllib.parse.urlsplit(server.base)
        raw = json.dumps(payload("Routine service notes. " * 500)).encode()
        with socket.create_connection((target.hostname, target.port), timeout=10) as connection:
            connection.sendall(f"POST /v1/systemone HTTP/1.1\r\nHost: {target.hostname}\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n".encode() + raw)
            time.sleep(.02)
        answer, _ = decide("helper-after-disconnect")
        check("helper_recovers_after_own_disconnect", answer["answers"]["team"]["choice"] == "billing")
        answer, _ = generate("main-after-helper-disconnect")
        check("main_survives_helper_disconnect", json.loads(response_text(answer)) == {"ready": True})

        from openai import OpenAI
        sdk = OpenAI(base_url=server.base + "/v1", api_key="local")
        with sdk.responses.connect() as connection:
            connection.response.create(model=args.main_model, store=False, temperature=0,
                input='Return JSON {"sdk":true}.', max_output_tokens=1024 if flint else 48,
                **({} if flint else {"reasoning": {"effort": "none"}}))
            for event in connection:
                if event.type == "error":
                    raise AssertionError(event.model_dump())
                if event.type in TERMINAL:
                    response = event.response.model_dump(mode="json")
                    check("official_sdk_ws", json.loads(response_text(response)) == {"sdk": True}, response)
                    break
        result["pass"] = all(result["checks"].values())
    except Exception as error:
        result["error"] = repr(error)
    finally:
        save(args.output / "result.json", result)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
