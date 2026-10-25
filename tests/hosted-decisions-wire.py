#!/usr/bin/env python3
"""Wire/SDK leaves for hosted-decisions.nu; never substitutes for model execution."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

from test_responses_websocket import Client, TERMINAL, http, response_text, save


QUESTION = {"type": "choice", "name": "department", "instructions": "Who should handle this customer request?",
            "choices": [{"value": "billing", "description": "Payments, charges and refunds"},
                        {"value": "technical", "description": "Software faults and crashes"}]}
EVIDENCE = "The customer was charged twice and wants the duplicate payment refunded."
INSTRUCTIONS = ('Call ds4_decide exactly once, with questions=[' + json.dumps(QUESTION) + ']. '
                'Omit input so the tool uses its bound evidence. After receiving the result, '
                'answer with a JSON object containing one department key and the selected choice string. '
                'Do not call a tool again after its result.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("model")
    p.add_argument("helper")
    p.add_argument("output", type=Path)
    p.add_argument("--after-restart", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / ("restart-result.json" if args.after_restart else "result.json")
    result = {"pass": False, "checks": {}}
    server = SimpleNamespace(output=args.output, ws_url=args.url.replace("http://", "ws://") + "/v1/responses")

    def check(name, condition, detail=None):
        result["checks"][name] = bool(condition)
        save(result_path, result)
        assert condition, (name, detail)
        print("PASS", name, flush=True)

    def request(**options):
        return {"model": args.model, "input": EVIDENCE, "instructions": INSTRUCTIONS, "temperature": 0,
                "store": False, "max_output_tokens": 1536,
                "tools": [{"type": "ds4:decision", "model": args.helper}]} | (
                    {} if args.model in {"flint", "flint-int4"} else {"reasoning": {"effort": "none"}}) | options

    def post(name, body, expected=200):
        return http(args.url, args.output, name, "POST", "/v1/responses", body, expected)

    def answer(name, response, expected):
        raw = response_text(response)
        try:
            value = json.loads(raw)
        except ValueError:
            # A plain, correct department is an answer-format observation,
            # rather than a failed native invocation or missing replay state.
            result.setdefault("observations", {})[name + "_plain_answer"] = raw
            check(name, len(expected) == 1 and raw == next(iter(expected.values())), response)
        else:
            check(name, value == expected, response)

    def qualify(name, response, events=None):
        receipts = [item for item in response["output"] if item["type"] == "ds4:decision_call"]
        check(name + "_receipt", len(receipts) == 1 and receipts[0]["status"] == "completed", response)
        receipt = receipts[0]
        check(name + "_call_group", receipt["call_id"] in receipt["call_ids"], receipt)
        check(name + "_bound_input", receipt["input"] == EVIDENCE and receipt["model"] == args.helper, receipt)
        arguments = json.loads(receipt["arguments"])
        direct = http(args.url, args.output, name + "-direct", "POST", "/v1/decisions",
                      {"model": receipt["model"], "input": receipt["input"], "questions": arguments["questions"]})
        check(name + "_native_parity", receipt["result"] == direct, receipt)
        answer(name + "_answer", response, {"department": "billing"})
        check(name + "_no_client_calls", not any(i["type"] == "function_call" for i in response["output"]))
        if events is not None:
            check(name + "_single_terminal", sum(e["type"] in TERMINAL for e in events) == 1)
            numbers = [e["sequence_number"] for e in events]
            check(name + "_sequence", all(a < b for a, b in zip(numbers, numbers[1:])), numbers)
            added = [e for e in events if e["type"] == "response.output_item.added"]
            done = [e for e in events if e["type"] == "response.output_item.done"]
            check(name + "_receipt_lifecycle", any(e["item"]["id"] == receipt["id"] and e["item"]["status"] == "in_progress" for e in added)
                  and any(e["item"] == receipt for e in done))
            check(name + "_output_indices", all(response["output"][e["output_index"]]["id"] == e["item"]["id"] for e in added + done))
        return receipt

    try:
        if args.after_restart:
            parent = json.loads((args.output / "stored.response.json").read_text())
            restored = http(args.url, args.output, "restored", "GET", "/v1/responses/" + parent["id"])
            check("stored_receipt_survives_restart", restored["output"] == parent["output"])
            child = post("restored-child", request(previous_response_id=parent["id"], tools=[],
                instructions='Use the prior decision result. Do not use tools. Return JSON with its department.',
                input="Repeat the department the decision tool selected."))
            answer("restart_continuation", child, {"department": "billing"})
            check("restart_no_reexecution", not any(i["type"] == "ds4:decision_call" for i in child["output"]))
        else:
            models = http(args.url, args.output, "models", "GET", "/v1/models")
            check("loaded_models", {args.model, args.helper}.issubset({m["id"] for m in models["data"]}))
            unavailable = post("unavailable", request(tools=[{"type": "ds4:decision", "model": "missing-helper"}]), 400)
            check("unavailable_model_explained", "not loaded" in unavailable["error"]["message"], unavailable)
            ordinary = {"type": "function", "name": "ds4_decide", "parameters": {"type": "object", "properties": {}}}
            for index, tools in enumerate(([ordinary] + request()["tools"], request()["tools"] + [ordinary])):
                post(f"collision-{index}", request(tools=tools), 400)
            post("required", request(tool_choice="required"), 400)
            post("zero-calls", request(max_tool_calls=0), 400)
            check("preflight_rejections", True)

            first = post("stored", request(store=True))
            qualify("json", first)
            child = post("previous", request(previous_response_id=first["id"], tools=[],
                instructions='Return JSON with the department selected by the earlier tool.',
                input="Repeat the selected department."))
            answer("previous_receipt_history", child, {"department": "billing"})
            check("previous_no_reexecution", not any(i["type"] == "ds4:decision_call" for i in child["output"]))
            replay = post("full-replay", request(tools=[], instructions='Return JSON with the department selected by the earlier tool.',
                input=[{"role": "user", "content": EVIDENCE}] + first["output"] + [{"role": "user", "content": "Repeat the selected department."}]))
            answer("full_receipt_replay", replay, {"department": "billing"})

            with Client(server, "ws") as client:
                client.send(request(type="response.create"))
                ws, events = client.complete()
                qualify("ws", ws, events)
                client.send(request(type="response.create", tools=[], previous_response_id=ws["id"],
                    instructions='Return JSON with the department selected by the earlier tool.', input="Repeat the department from the prior tool result."))
                next_response, _ = client.complete()
                answer("ephemeral_ws_replay", next_response, {"department": "billing"})

            body = request(stream=True)
            save(args.output / "sse.request.json", body)
            wire = urllib.request.Request(args.url + "/v1/responses", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(wire, timeout=180) as reply:
                events = [json.loads(line[6:]) for line in reply.read().decode().splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
            save(args.output / "sse.events.json", events)
            qualify("sse", next(e["response"] for e in events if e["type"] in TERMINAL), events)

            from openai import OpenAI
            sdk = OpenAI(base_url=args.url + "/v1", api_key="local")
            sdk_response = sdk.responses.create(**request()).model_dump(mode="json", warnings=False)
            save(args.output / "sdk.response.json", sdk_response)
            qualify("sdk", sdk_response)
            sdk_replay = post("sdk-full-replay", request(tools=[],
                instructions="Return JSON with the department selected by the earlier tool.",
                input=[{"role": "user", "content": EVIDENCE}] + sdk_response["output"] +
                      [{"role": "user", "content": "Repeat the department from that tool result."}]))
            answer("sdk_full_output_replay", sdk_replay, {"department": "billing"})
            check("sdk_replay_no_reexecution", not any(i["type"] == "ds4:decision_call" for i in sdk_replay["output"]))
            with sdk.responses.connect() as connection:
                connection.response.create(**request())
                for event in connection:
                    if event.type == "error":
                        raise AssertionError(event.model_dump(mode="json"))
                    if event.type in TERMINAL:
                        sdk_ws = event.response.model_dump(mode="json", warnings=False)
                        save(args.output / "sdk-ws.response.json", sdk_ws)
                        qualify("sdk_ws", sdk_ws)
                        break
            none = post("none", request(tool_choice="none", instructions='Return only JSON {"ready":true}.', input="Reply as instructed."))
            check("none_never_invokes", not any(i["type"] == "ds4:decision_call" for i in none["output"]))
            client_call = post("ordinary-name", request(tools=[ordinary], instructions="Call ds4_decide exactly once with {}. Do not answer with text."))
            check("ordinary_name_remains_client_owned", any(i["type"] == "function_call" and i["name"] == "ds4_decide" for i in client_call["output"])
                  and not any(i["type"] == "ds4:decision_call" for i in client_call["output"]), client_call)
        result["pass"] = all(result["checks"].values())
    except Exception as error:
        result["error"] = repr(error)
    finally:
        save(result_path, result)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
