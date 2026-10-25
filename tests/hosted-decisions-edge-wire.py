#!/usr/bin/env python3
"""Focused hosted-tool edge cases against an already running real-model server.

Run after hosted-decisions-wire.py. Requests/events/results stay in output;
this leaf never launches a model or substitutes a mocked generation result.
"""
import argparse
import base64
import json
from pathlib import Path
from types import SimpleNamespace

from test_responses_websocket import Client, TERMINAL, http, response_text, save


QUESTION = {"type": "choice", "name": "department",
            "instructions": "Who should handle this customer request?",
            "choices": [{"value": "billing", "description": "Payments, charges and refunds"},
                        {"value": "technical", "description": "Software faults and crashes"}]}
EVIDENCE = "The customer was charged twice and wants the duplicate payment refunded."
TECHNICAL = "The application crashes on startup with a software exception. No payment issue is involved."
ARGUMENTS = json.dumps({"questions": [QUESTION]}, separators=(",", ":"))
ONCE = ("Call ds4_decide exactly once with arguments " + ARGUMENTS + ". Omit input: use the tool's bound evidence. "
        'After its result, answer with a JSON object containing one department key and the selected choice string. Do not call a tool again.')
EXTERNAL = {"type": "function", "name": "client_marker", "description": "Read a marker held by the client.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                           "required": ["key"], "additionalProperties": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("model")
    parser.add_argument("helper")
    parser.add_argument("output", type=Path)
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--mixed-only", action="store_true")
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "edge-result.json"
    result = {"pass": False, "checks": {}, "errors": {}, "observations": {}}
    server = SimpleNamespace(output=args.output,
        ws_url=args.url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/responses")

    def check(name, condition, detail=None):
        result["checks"][name] = bool(condition)
        save(result_path, result)
        assert condition, (name, detail)
        print("PASS", name, flush=True)

    def request(**options):
        return {"model": args.model, "input": EVIDENCE, "instructions": ONCE, "temperature": 0,
                "store": False, "max_output_tokens": 1536,
                "tools": [{"type": "ds4:decision", "model": args.helper}]} | (
                    {} if args.model in {"flint", "flint-int4"} else {"reasoning": {"effort": "none"}}) | options

    def post(name, body):
        return http(args.url, args.output, name, "POST", "/v1/responses", body)

    def receipts(response):
        return [item for item in response["output"] if item["type"] == "ds4:decision_call"]

    def mixed(client_first):
        name = "mixed-client-first" if client_first else "mixed-hosted-first"
        marker = "RIVER-81" if client_first else "EMBER-27"
        hosted = {"type": "ds4:decision", "model": args.helper, "input": EVIDENCE}
        order = ([EXTERNAL, hosted] if client_first else [hosted, EXTERNAL])
        first = 'client_marker({"key":"current"})' if client_first else "ds4_decide(" + ARGUMENTS + ")"
        second = "ds4_decide(" + ARGUMENTS + ")" if client_first else 'client_marker({"key":"current"})'
        instructions = ("Before answering, invoke BOTH tools in ONE parallel tool-call batch. Emit exactly two invocations "
            "in this source order: FIRST " + first + "; SECOND " + second + ". Do not wait between invocations. "
            "For ds4_decide omit input so the bound evidence is used. Do not invent either result. "
            'When both results are available, return a JSON object with department and marker fields containing the returned values.')
        parent = post(name, request(input="Evaluate the bound evidence and read the current client marker.",
            tools=order, instructions=instructions, store=True))
        calls = [item for item in parent["output"] if item["type"] in {"function_call", "ds4:decision_call"}]
        expected = ["function_call", "ds4:decision_call"] if client_first else ["ds4:decision_call", "function_call"]
        # Flint's current decode boundary yields at the first closed XML call.
        # A client-first pair consequently spans two Responses. Exercise that
        # supported path instead of attributing the early yield to the model.
        if client_first and args.model in {"flint", "flint-int4"} and [item["type"] for item in calls] == ["function_call"]:
            check(name + "_yields_client_before_helper", not receipts(parent), parent)
            result["observations"]["flint_parallel_boundary"] = "One XML invocation per generated round; client-first resumes before its hosted invocation."
            call = calls[0]
            child = post(name + "-resume", request(previous_response_id=parent["id"], tools=[hosted],
                input=[{"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps({"marker": marker})}],
                instructions="The client marker result is available. Now call ds4_decide once with " + ARGUMENTS +
                ". Omit input. Then answer with one JSON object whose department and marker fields contain both tool results."))
            check(name + "_hosted_after_client_result", len(receipts(child)) == 1 and receipts(child)[0]["status"] == "completed", child)
            raw = response_text(child)
            answer, end = json.JSONDecoder().raw_decode(raw)
            # Record Flint's known ordinary/hosted duplicate-close behavior as
            # answer-format quality, separately from actual tool execution.
            result["observations"]["flint_client_first_clean_json"] = not raw[end:].strip()
            check(name + "_answer", answer == {"department": "billing", "marker": marker}, child)
            return
        check(name + "_model_emitted_both_in_requested_order", [item["type"] for item in calls] == expected, parent)
        receipt = receipts(parent)[0]
        check(name + "_hosted_completed", receipt["status"] == "completed" and receipt["input"] == EVIDENCE, receipt)
        if args.model not in {"flint", "flint-int4"}:
            check(name + "_parallel_receipt_group", receipt["call_ids"] == [item["call_id"] for item in calls], receipt)
        call = next(item for item in calls if item["type"] == "function_call")
        check(name + "_external_owned_by_client", call["name"] == "client_marker" and json.loads(call["arguments"]) == {"key": "current"}, call)
        # Only the external result is supplied. The server already owns the
        # hosted result and must retain it in the parent's logical transcript.
        child = post(name + "-resume", request(previous_response_id=parent["id"], tools=[],
            input=[{"type": "function_call_output", "call_id": call["call_id"],
                    "output": json.dumps({"marker": marker})}],
            instructions='Both tools have now returned. Do not call tools. Return only JSON with department and marker from their results.'))
        check(name + "_external_only_resumption", json.loads(response_text(child)) == {"department": "billing", "marker": marker}, child)
        check(name + "_no_hosted_reexecution", not receipts(child), child)
        replay = post(name + "-full-replay", request(tools=[],
            input=[{"role": "user", "content": "Evaluate the bound evidence and read the current client marker."}] + parent["output"] +
                  [{"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps({"marker": marker})}],
            instructions='Both tools have returned. Return one JSON object with department and marker from their results.'))
        check(name + "_full_replay", json.loads(response_text(replay)) == {"department": "billing", "marker": marker}, replay)

    def bound_evidence():
        for name, evidence, expected in (("bound-billing", EVIDENCE, "billing"), ("bound-technical", TECHNICAL, "technical")):
            response = post(name, request(input="Classify the request held in the tool's bound evidence.",
                tools=[{"type": "ds4:decision", "model": args.helper, "input": evidence}]))
            calls = receipts(response)
            check(name + "_resolved_input", len(calls) == 1 and calls[0]["input"] == evidence, response)
            check(name + "_binding_not_copied_by_model", "input" not in json.loads(calls[0]["arguments"]), calls[0])
            check(name + "_answer", json.loads(response_text(response)) == {"department": expected}, response)

    def conversation():
        created = http(args.url, args.output, "edge-conversation-create", "POST", "/v1/conversations", {})
        identifier = created["id"]
        parent = post("edge-conversation-parent", request(conversation=identifier, store=True))
        check("conversation_parent_receipt", len(receipts(parent)) == 1 and receipts(parent)[0]["status"] == "completed", parent)
        listing = http(args.url, args.output, "edge-conversation-items", "GET",
            "/v1/conversations/" + identifier + "/items?order=asc&limit=100")
        receipt = receipts(parent)[0]
        check("conversation_publishes_receipt", any(item == receipt for item in listing["data"]), listing)
        child = post("edge-conversation-child", request(conversation=identifier, store=True, tools=[],
            input="Repeat the department returned by the earlier tool.",
            instructions='Use the earlier tool result. Do not call tools. Return only JSON containing its department.'))
        check("conversation_continues_hosted_history", json.loads(response_text(child)) == {"department": "billing"}, child)
        check("conversation_does_not_reexecute", not receipts(child), child)
        listing = http(args.url, args.output, "edge-conversation-items-after", "GET",
            "/v1/conversations/" + identifier + "/items?order=asc&limit=100")
        check("conversation_retains_one_receipt", sum(item["type"] == "ds4:decision_call" for item in listing["data"]) == 1, listing)

    def call_budget():
        instructions = ("You must make exactly two SEQUENTIAL decision calls. First call ds4_decide with " + ARGUMENTS + ". "
            "Omit input. Wait for that result. After the first result, call ds4_decide AGAIN with the SAME arguments exactly once. "
            "Do not answer or stop after the first result; the second call is mandatory. After the second result answer DONE.")
        response = post("edge-call-budget", request(instructions=instructions, max_tool_calls=1))
        calls = receipts(response)
        check("call_budget_model_attempted_second_call", len(calls) >= 2, response)
        check("call_budget_only_one_execution", sum(item["status"] == "completed" for item in calls) == 1, response)
        check("call_budget_failed_receipt", calls[-1]["status"] == "failed"
            and calls[-1]["result"].get("error", {}).get("code") == "max_tool_calls_exceeded", calls[-1])
        check("call_budget_terminal_failure", response["status"] == "failed", response)

    def output_budget():
        limit = 768
        instructions = ("First call ds4_decide exactly once with " + ARGUMENTS + ". Omit input. "
            "After its result, write ALL integers from 1 through 5000, one per line. Do not summarize or skip integers. "
            "Do not call any tool after the first result. Continue counting until the response limit.")
        response = post("edge-output-budget", request(instructions=instructions, max_output_tokens=limit))
        check("output_budget_crossed_hosted_round", len(receipts(response)) == 1 and receipts(response)[0]["status"] == "completed", response)
        check("output_budget_cumulative_usage", 0 < response["usage"]["output_tokens"] <= limit, response["usage"])
        check("output_budget_terminal_incomplete", response["status"] == "incomplete"
            and response["incomplete_details"]["reason"] == "max_output_tokens", response)

    def disconnect():
        # About 4k helper tokens, held in the binding rather than generated
        # arguments. Close as soon as the provider invocation is announced.
        long_evidence = ("Routine service note: the office printer works normally.\n" * 400) + EVIDENCE
        victim = Client(server, "edge-disconnect")
        try:
            victim.send(request(type="response.create", input="Classify the bound evidence.",
                tools=[{"type": "ds4:decision", "model": args.helper, "input": long_evidence}]))
            while True:
                event = victim.receive()
                assert event["type"] not in TERMINAL and event["type"] != "error", event
                if (event["type"] == "response.output_item.added" and event["item"]["type"] == "ds4:decision_call"
                        and event["item"]["status"] == "in_progress"):
                    result["observations"]["disconnect_receipt_id"] = event["item"]["id"]
                    break
            victim.close()
            victim = None
            check("disconnect_after_hosted_start", True)
        finally:
            if victim is not None:
                victim.close()
        recovered = post("edge-disconnect-recovery", request())
        check("disconnect_recovers_generator_and_helper", len(receipts(recovered)) == 1
            and receipts(recovered)[0]["status"] == "completed"
            and json.loads(response_text(recovered)) == {"department": "billing"}, recovered)

    def media():
        image = Path("tests/vision-fixtures/qwen38/maple.png").read_bytes()
        uri = "data:image/png;base64," + base64.b64encode(image).decode()
        question = {"type": "choice", "name": "color", "instructions": "What color is the circle in the picture?",
                    "choices": [{"value": "blue", "description": "The circle is blue"},
                                {"value": "red", "description": "The circle is red"}]}
        response = post("media-latest-user", request(input=[{"role": "user", "content": [
            {"type": "input_text", "text": "Look at the circle in this picture."},
            {"type": "input_image", "image_url": uri}]}],
            instructions="Call ds4_decide once with questions=" + json.dumps([question]) +
            ". Omit input to use the latest user image. After the result answer with a JSON object with its color value."))
        calls = receipts(response)
        check("media_hosted_completed", len(calls) == 1 and calls[0]["status"] == "completed", response)
        receipt = calls[0]
        check("media_image_bound_without_generated_copy", "input" not in json.loads(receipt["arguments"])
              and any(part.get("image_url") == uri for part in receipt["input"]), receipt)
        direct = http(args.url, args.output, "media-direct", "POST", "/v1/decisions",
            {"model": args.helper, "input": receipt["input"], "questions": json.loads(receipt["arguments"])["questions"]})
        check("media_native_parity", receipt["result"] == direct, receipt)
        check("media_helper_quality", receipt["result"]["answers"][0]["choice"] == "blue", receipt)
        check("media_generator_continued", json.loads(response_text(response)) == {"color": "blue"}, response)

    # Continue independent cases after a failure so a model refusing a requested
    # parallel batch does not hide the server's other lifecycle observations.
    cases = [("mixed_client_first", lambda: mixed(True)), ("mixed_hosted_first", lambda: mixed(False)),
             ("bound_evidence", bound_evidence), ("conversation", conversation),
             ("call_budget", call_budget), ("output_budget", output_budget), ("disconnect", disconnect)]
    if args.media:
        cases = [("media", media)]
    elif args.mixed_only:
        cases = cases[:2]
    for name, run in cases:
        try:
            run()
        except Exception as error:
            result["errors"][name] = repr(error)
            print("FAIL", name, repr(error), flush=True)
        save(result_path, result)
    result["pass"] = not result["errors"] and all(result["checks"].values())
    save(result_path, result)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
