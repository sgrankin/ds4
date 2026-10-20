#!/usr/bin/env python3
"""Score frozen first completions in bounded, restricted fresh Python workers."""
import argparse
import ast
import copy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import itertools
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

MODULES = {"decimal", "heapq", "collections"}
EXPORTS = {"decimal": {"Decimal", "ROUND_HALF_UP", "getcontext", "localcontext"},
           "heapq": {"heappush", "heappop", "heapify", "heappushpop", "heapreplace", "nsmallest", "nlargest"},
           "collections": {"deque", "defaultdict", "Counter", "OrderedDict"}}
BUILTINS = {"abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
            "isinstance", "len", "list", "max", "min", "range", "reversed", "round",
            "set", "sorted", "str", "sum", "tuple", "zip", "iter", "next", "StopIteration", "ValueError", "TypeError",
            "IndexError", "RuntimeError", "Exception"}
ATTRIBUTES = {"append", "extend", "pop", "popleft", "appendleft", "sort", "copy",
              "index", "count", "reverse", "get", "items", "keys", "values", "setdefault",
              "update", "add", "remove", "discard", "lstrip", "rstrip", "strip", "split",
              "find", "startswith", "endswith", "isdigit", "join", "replace", "quantize",
              "to_integral_value", "as_tuple", "normalize", "Decimal", "ROUND_HALF_UP",
              "getcontext", "localcontext", "prec", "heappush", "heappop", "heapify",
              "deque", "defaultdict", "Counter", "clear", "fromkeys", "union", "difference",
              "intersection", "difference_update", "intersection_update", "issubset", "issuperset",
              "partition", "rpartition", "removeprefix", "removesuffix", "to_integral_exact",
              "is_signed", "is_zero"} | set().union(*EXPORTS.values())
DANGEROUS = {"eval", "exec", "compile", "open", "input", "getattr", "setattr", "delattr",
             "globals", "locals", "vars", "dir", "type", "object", "super", "print",
             "breakpoint", "help", "exit", "quit"}


def extract_code(raw):
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0] not in ("```", "```python", "```py") or lines[-1] != "```":
            raise ValueError("unsupported or incomplete code fence")
        if sum(line.startswith("```") for line in lines) != 2:
            raise ValueError("multiple code fences")
        text = "\n".join(lines[1:-1])
    if not text:
        raise ValueError("empty completion")
    return text


def validate_code(code, required_function):
    tree = ast.parse(code)
    allowed_top = (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not isinstance(node, allowed_top):
            raise ValueError("unsupported top-level statement")
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if required_function not in functions:
        raise ValueError("missing requested function")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise ValueError("unsupported namespace statement")
        if isinstance(node, ast.Import):
            if any(alias.name not in MODULES for alias in node.names):
                raise ValueError("unsupported import")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in MODULES or any(alias.name not in EXPORTS[node.module] for alias in node.names):
                raise ValueError("unsupported import")
        if isinstance(node, ast.alias):
            if "__" in node.name or (node.asname and "__" in node.asname):
                raise ValueError("dunder import")
        if isinstance(node, ast.Name) and ("__" in node.id or node.id in DANGEROUS):
            raise ValueError("dangerous name")
        if isinstance(node, ast.Attribute) and node.attr not in ATTRIBUTES:
            raise ValueError("unsupported attribute: " + node.attr)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ValueError("dunder string access")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or "__" in node.name):
            raise ValueError("unsupported function decorator/name")
    return tree


def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    fromlist = fromlist or ()
    if level or name not in MODULES or any(item not in EXPORTS[name] for item in fromlist):
        raise ImportError("unsupported import")
    return __import__(name, globals, locals, fromlist, level)


def inputs(task_id, args):
    result = copy.deepcopy(args)
    if task_id == "merge_intervals":
        result[0] = [tuple(pair) for pair in result[0]]
    elif task_id == "topological_order":
        result[1] = [tuple(pair) for pair in result[1]]
    return result


def valid_shape(task_id, value):
    integer = lambda x: type(x) is int
    if task_id == "round_cents":
        return integer(value)
    if task_id == "merge_intervals":
        return type(value) is list and all(type(pair) is tuple and len(pair) == 2
                                          and all(integer(x) for x in pair) for pair in value)
    if task_id == "topological_order":
        return value is None or (type(value) is list and all(integer(x) for x in value))
    return type(value) is list and all(type(x) is str for x in value)


def worker(fixture_path, raw_path, task_id):
    fixture = json.loads(Path(fixture_path).read_text())
    task = next(t for t in fixture["tasks"] if t["id"] == task_id)
    raw = Path(raw_path).read_text(encoding="utf-8", errors="strict")
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    # Darwin rejects small AS/DATA/RSS limits on this host. The parent enforces
    # observed RSS; Linux additionally enforces address space in the kernel.
    memory_limit = 512 * 1024 * 1024
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    result = dict(task=task_id, status="INVALID", tests_total=len(task["tests"]), tests_passed=0)
    try:
        tree = validate_code(extract_code(raw), task["function"])
        safe = {name: getattr(__import__("builtins"), name) for name in BUILTINS}
        safe["__import__"] = restricted_import
        namespace = {"__builtins__": safe}
        exec(compile(tree, "<first-completion>", "exec"), namespace)
        function = namespace[task["function"]]
        failures = []
        for index, test in enumerate(task["tests"]):
            args = inputs(task_id, test["args"])
            before = copy.deepcopy(args)
            try:
                actual = function(*args)
                shape_ok = valid_shape(task_id, actual)
                canonical = json.loads(json.dumps(actual)) if shape_ok else None
                correct = shape_ok and canonical == test["expected"] and args == before
                if correct:
                    result["tests_passed"] += 1
                else:
                    failures.append(dict(index=index, expected=test["expected"], actual=canonical,
                                         shape_ok=shape_ok, input_unchanged=args == before))
            except Exception as error:
                failures.append(dict(index=index, exception=type(error).__name__, detail=str(error)[:200]))
        result["status"] = "CORRECT" if not failures else "INCORRECT"
        result["failures"] = failures
    except Exception as error:
        result["reason"] = type(error).__name__ + ": " + str(error)[:200]
    encoded = json.dumps(result)
    if len(encoded) > 65536:
        encoded = json.dumps(dict(task=task_id, status="INVALID", reason="excessive result output"))
    print(encoded)


def bounded_worker(command, cwd, wall_seconds=3.5, rss_kib=512*1024):
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, cwd=cwd)
    start = time.monotonic()
    peak_rss = 0
    reason = None
    try:
        while True:
            try:
                stdout, stderr = child.communicate(timeout=.05)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - start > wall_seconds:
                    reason = "execution worker timed out"
                else:
                    try:
                        usage = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(child.pid)],
                                               capture_output=True, text=True, timeout=.5)
                        if usage.returncode == 0 and usage.stdout.strip():
                            peak_rss = max(peak_rss, int(usage.stdout.strip()))
                            if peak_rss > rss_kib:
                                reason = "execution worker exceeded RSS limit"
                        elif child.poll() is None:
                            reason = "execution worker RSS monitor failed"
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        reason = "execution worker RSS monitor failed"
                if reason:
                    child.kill()
                    stdout, stderr = child.communicate()
                    break
    finally:
        if child.poll() is None:
            child.kill(); child.communicate()
    return child.returncode, stdout, stderr, reason, peak_rss


def validate_golds(fixture):
    for task in fixture["tasks"]:
        for test in task["tests"]:
            if task["id"] == "round_cents":
                actual = int((Decimal(test["args"][0]) * 100).to_integral_value(rounding=ROUND_HALF_UP))
                assert actual == test["expected"]
            elif task["id"] == "topological_order":
                n, edges = test["args"]
                valid = [list(order) for order in itertools.permutations(range(n))
                         if all(order.index(u) < order.index(v) for u, v in edges)]
                assert (min(valid) if valid else None) == test["expected"]
            elif task["id"] == "merge_intervals":
                original = test["args"][0]
                expected = test["expected"]
                assert all(expected[i][1] < expected[i + 1][0] for i in range(len(expected) - 1))
                endpoints = [x for pair in original + expected for x in pair]
                for point2 in range(2 * min(endpoints, default=0), 2 * max(endpoints, default=0) + 1):
                    assert any(2*a <= point2 <= 2*b for a,b in original) == any(
                        2*a <= point2 <= 2*b for a,b in expected)
    return sum(len(task["tests"]) for task in fixture["tasks"])


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:]); return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--validate-golds", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    count = validate_golds(fixture)
    if args.validate_golds:
        print(json.dumps(dict(status="PASS", tests=count, fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest())))
        return
    if args.run is None:
        parser.error("run is required unless --validate-golds")
    records = [json.loads(line) for line in (args.run / "generation.jsonl").read_text().splitlines() if line]
    expected = fixture["task_order"]
    if [record["task"] for record in records] != expected:
        raise ValueError("generation records must cover the frozen task order exactly once")
    scored = []
    for record in records:
        result = dict(generation=record, task=record["task"], status="INVALID")
        if record["cap_hit"] or not record["stopped"]:
            result["reason"] = "generation did not finish within the token budget"
        else:
            command = [sys.executable, "-I", "-S", str(Path(__file__).resolve()),
                "--worker", str(args.fixture.resolve()), str((args.run / record["raw_trace"]).resolve()),
                record["task"]]
            code, stdout, stderr, reason, peak = bounded_worker(command, args.run)
            result["worker_peak_rss_kib_observed"] = peak
            if reason or code:
                result["reason"] = reason or "execution worker exited " + str(code)
                result["stderr"] = stderr[:1000]
            else:
                result.update(json.loads(stdout))
        scored.append(result)
    out = dict(fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
               correct=sum(r["status"] == "CORRECT" for r in scored), total=len(scored), tasks=scored,
               execution_limits=dict(cpu_seconds=2, wall_seconds=3.5, observed_rss_kib=512*1024,
                   rss_poll_seconds=.05, kernel_address_space_bytes=512*1024*1024 if sys.platform != "darwin" else None,
                   darwin_memory_caveat="Observed RSS limit permits transient overshoot between polls"),
               interpretation="Single first completion; no repair, reranking, or best-of")
    with (args.run / "score.json").open("x") as stream:
        json.dump(out, stream, indent=2); stream.write("\n")
    print(json.dumps(dict(correct=out["correct"], total=out["total"], score=str(args.run / "score.json"))))
    sys.exit(0 if out["correct"] == out["total"] else 3)


if __name__ == "__main__":
    main()
