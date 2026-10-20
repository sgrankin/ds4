#!/usr/bin/env python3
"""Small Conversations HTTP qualification; ROOT owns the server and restart."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("responses_hour11", ROOT / "speed-bench/experiments/responses-hour11-20261004/run.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
INSTRUCTIONS = "Use the conversation's active facts. Answer the newest question directly, with only the requested JSON."
QUERY = "What are the site and net usable units (packages times units_per_package minus damaged)? Reply JSON with site and total."
GOLD = {"site": "MAPLE", "total": 416}
GOLD_B = {"site": "CEDAR", "total": 202}


def user(text):
    return {"role": "user", "content": text}


def text(item):
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def api(c, name, method, route, payload=None, status=200, check=None):
    exchange = c.http(method, route, payload)
    body = exchange.get("body")
    passed = exchange["http_status"] == status
    try:
        if check:
            passed = passed and check(body)
    except (TypeError, KeyError, AttributeError, IndexError):
        passed = False
    c.record(name, exchange, passed)
    return body if isinstance(body, dict) else {}


def items(c, name, cid, order="asc", limit=100, after=None):
    route = f"/conversations/{cid}/items?order={order}&limit={limit}"
    if after:
        route += "&after=" + after
    return api(c, name, "GET", route, check=lambda b: b.get("object") == "list"
               and isinstance(b.get("data"), list) and type(b.get("has_more")) is bool)


def create(c, name, initial, label):
    body = api(c, name, "POST", "/conversations", {"metadata": {"label": label}, "items": initial},
               check=lambda b: b.get("object") == "conversation" and isinstance(b.get("id"), str)
               and isinstance(b.get("created_at"), int) and b.get("metadata") == {"label": label})
    if not body.get("id"):
        raise RuntimeError(name + " did not create a conversation")
    return body["id"]


def post(c, name, cid, new_input, gold=None, store=False, tool=False, cap=64):
    payload = {"conversation": cid, "input": new_input, "store": store,
               "temperature": 0, "instructions": INSTRUCTIONS, "max_output_tokens": cap}
    if tool:
        payload.update(tools=[old.TOOL], tool_choice="auto")
    response = c.post(name, payload, gold=gold, tool=tool and gold is None)
    if isinstance(response, dict):
        identifier = cid.get("id") if isinstance(cid, dict) else cid
        c.rows[-1]["conversation_echo_correct"] = response.get("conversation") == {"id": identifier}
        if not c.rows[-1]["conversation_echo_correct"]:
            c.rows[-1]["result"] = "FAIL"
    return response


def output_once(c, name, cid, response, prior_ids, expected_new_inputs):
    listed = items(c, name, cid)
    data = listed.get("data", [])
    output = (response or {}).get("output", [])
    out_ids = [item.get("id") for item in output]
    ids = [item.get("id") for item in data]
    passed = (len(data) == len(prior_ids)+expected_new_inputs+len(output)
              and ids[:len(prior_ids)] == prior_ids and all(out_ids)
              and all(ids.count(identifier) == 1 for identifier in out_ids)
              and [item for item in data if item.get("id") in out_ids] == output)
    c.rows[-1]["append_once_correct"] = bool(passed)
    if not passed:
        c.rows[-1]["result"] = "FAIL"
    return data


def route_checks(c):
    originals = [user(f"Route note {i:02d}.") for i in range(24)]
    cid = create(c, "routes-create", originals[:20], "routes")
    api(c, "routes-add-four", "POST", f"/conversations/{cid}/items", {"items": originals[20:]},
        check=lambda b: b.get("object") == "list" and len(b.get("data", [])) == 4)
    api(c, "routes-get", "GET", "/conversations/"+cid,
        check=lambda b: b.get("id") == cid and b.get("object") == "conversation"
        and b.get("metadata") == {"label": "routes"})
    listing = items(c, "routes-all-asc", cid)
    data = listing.get("data", [])
    ids = [item.get("id") for item in data]
    if len(data) != 24 or [text(item) for item in data] != [text(item) for item in originals]:
        c.rows[-1]["result"] = "FAIL"
        return cid
    api(c, "routes-default-desc20", "GET", f"/conversations/{cid}/items",
        check=lambda b: b.get("object") == "list" and b.get("has_more") is True
        and [x.get("id") for x in b.get("data", [])] == list(reversed(ids))[:20]
        and b.get("first_id") == ids[-1] and b.get("last_id") == ids[-20])
    page = items(c, "routes-page-after", cid, "asc", 3, ids[2])
    if [x.get("id") for x in page.get("data", [])] != ids[3:6] or page.get("has_more") is not True:
        c.rows[-1]["result"] = "FAIL"
    api(c, "routes-item-get", "GET", f"/conversations/{cid}/items/{ids[5]}", check=lambda b: b == data[5])
    api(c, "routes-item-delete", "DELETE", f"/conversations/{cid}/items/{ids[5]}",
        check=lambda b: b.get("id") == cid and b.get("object") == "conversation")
    api(c, "routes-deleted-item404", "GET", f"/conversations/{cid}/items/{ids[5]}", status=404)
    appended = api(c, "routes-items-post", "POST", f"/conversations/{cid}/items",
        {"items": [user("Manual route note.")]}, check=lambda b: b.get("object") == "list"
        and len(b.get("data", [])) == 1 and text(b["data"][0]) == "Manual route note.")
    final = items(c, "routes-edit-list", cid)
    if len(final.get("data", [])) != 24 or ids[5] in [x.get("id") for x in final.get("data", [])]:
        c.rows[-1]["result"] = "FAIL"
    api(c, "routes-delete-tombstone", "DELETE", "/conversations/"+cid,
        check=lambda b: b.get("id") == cid and b.get("deleted") is True)
    api(c, "routes-tombstone-get404", "GET", "/conversations/"+cid, status=404)
    retained = items(c, "routes-tombstone-items-retained", cid)
    if retained.get("data") != final.get("data"):
        c.rows[-1]["result"] = "FAIL"
    return cid


def before(c):
    route_checks(c)
    log = "Obsolete audit log.\n" + "Old audit log unchanged.\n"*96
    active = ("Active facts: site=MAPLE, packages=17, units_per_package=25, damaged=9.\n"
              + "Current handbook: retain the active inventory facts.\n"*64)
    cid = create(c, "active-create", [user(log), user(active)], "active")
    initial = items(c, "active-initial-items", cid).get("data", [])
    if len(initial) != 2:
        raise RuntimeError("active initial item count differs")
    source = post(c, "source-storefalse", cid, QUERY, GOLD)
    all_items = output_once(c, "source-appended-once", cid, source, [x["id"] for x in initial], 1)
    if old.response_id(source):
        api(c, "storefalse-response404", "GET", "/responses/"+source["id"], status=404)
    api(c, "delete-obsolete-log", "DELETE", f"/conversations/{cid}/items/{initial[0]['id']}",
        check=lambda b: b.get("id") == cid and b.get("object") == "conversation")
    edited_public = items(c, "edited-public-items", cid).get("data", [])
    if [x["id"] for x in edited_public] != [x["id"] for x in all_items[1:]]:
        c.rows[-1]["result"] = "FAIL"
    edited = post(c, "after-item-delete-memory", {"id": cid},
        "Report the active site and net usable units again as JSON with site and total.", GOLD, store=True)
    all_items = output_once(c, "edited-appended-once", cid, edited, [x["id"] for x in edited_public], 1)
    other = create(c, "isolated-create", [user("Active facts: site=CEDAR, packages=19, units_per_package=11, damaged=7.")], "isolated")
    post(c, "second-conversation-isolated", other, QUERY, GOLD_B)
    conflict_ref = api(c, "conversation-previous-exclusive", "POST", "/responses",
        {"model": c.args.model, "conversation": cid, "previous_response_id": old.response_id(edited), "input": "Continue."}, status=400)
    pending = post(c, "pending-tool", cid,
        'Call lookup_count with key "subject" now. After its result, return JSON with site, total and count.', tool=True, cap=128)
    pending_items = output_once(c, "pending-appended-once", cid, pending, [x["id"] for x in all_items], 1)
    if old.response_id(pending):
        api(c, "pending-storefalse-response404", "GET", "/responses/"+pending["id"], status=404)
    return c.finish({"conversation": cid, "other_conversation": other,
                     "pending": pending, "stored_response": edited, "pending_items": pending_items,
                     "store_dir": str(c.args.store_dir.resolve()), "model": c.args.model})


def after(c):
    manifest = json.loads((c.output/"manifest.json").read_text())
    cid, other = manifest["conversation"], manifest["other_conversation"]
    if manifest["store_dir"] != str(c.args.store_dir.resolve()):
        raise RuntimeError("restart must use the same durable store")
    restored = items(c, "restart-items-exact", cid).get("data", [])
    if restored != manifest["pending_items"]:
        c.rows[-1]["result"] = "FAIL"
    stored = manifest["stored_response"]
    if old.response_id(stored):
        c.get_exact("restart-stored-response", stored)
        api(c, "response-delete", "DELETE", "/responses/"+stored["id"],
            check=lambda b: b.get("deleted") is True and b.get("id") == stored["id"])
        api(c, "deleted-response404", "GET", "/responses/"+stored["id"], status=404)
    call = old.call_item(manifest["pending"])
    if call:
        final = post(c, "restart-tool-output", cid,
            [{"type": "function_call_output", "call_id": call["call_id"], "output": '{"count":61}'}],
            {"site": "MAPLE", "total": 416, "count": 61}, tool=True)
        output_once(c, "tool-output-appended-once", cid, final, [x["id"] for x in restored], 1)
        post(c, "continue-after-response-delete", cid, QUERY, GOLD)
    else:
        c.skip("restart-tool-output", "pending first attempt produced no valid call; no retry")
    post(c, "restart-second-conversation-isolated", other, QUERY, GOLD_B)
    scratch = create(c, "race-create", [user("A manual edit must remain intact during generation.")], "race")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(c.http, "POST", "/responses", {"model": c.args.model, "conversation": scratch,
            "input": "Write integers 1 through 30 separated by commas.", "store": False,
            "temperature": 0, "reasoning": {"effort": "none"}, "max_output_tokens": 128})
        time.sleep(.3)
        edit = api(c, "race-manual-append", "POST", f"/conversations/{scratch}/items",
                   {"items": [user("This manually appended item must survive.")]},
                   check=lambda b: len(b.get("data", [])) == 1)
        response = future.result(timeout=240)
    body = response.get("body")
    c.record("race-version-conflict-visible", response,
             response["http_status"] in (409, 500) and isinstance(body, dict) and bool(body.get("error")))
    survivors = items(c, "race-manual-item-retained", scratch).get("data", [])
    if len(survivors) != 2 or not edit.get("data") or survivors[-1] != edit["data"][0]:
        c.rows[-1]["result"] = "FAIL"
    api(c, "active-delete-tombstone", "DELETE", "/conversations/"+cid,
        check=lambda b: b.get("deleted") is True and b.get("id") == cid)
    api(c, "active-tombstone-get404", "GET", "/conversations/"+cid, status=404)
    items(c, "active-tombstone-items-retained", cid)
    api(c, "deleted-conversation-response400", "POST", "/responses",
        {"model": c.args.model, "conversation": cid, "input": "Continue."}, status=400)
    return c.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--phase", choices=("before", "after"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--timeout", type=float, default=240)
    c = old.Campaign(p.parse_args())
    try:
        return before(c) if c.args.phase == "before" else after(c)
    except Exception as exc:
        c.rows.append({"name": "dependent-step-error", "result": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
        return c.finish()


if __name__ == "__main__":
    raise SystemExit(main())
