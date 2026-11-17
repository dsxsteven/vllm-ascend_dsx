# SPDX-License-Identifier: Apache-2.0
"""Record and compare a PCP8 service using fixed token prompts.

Run against an already admitted, ready service. This client never starts a
service or chooses devices. Record off/2048 and 2048/2048 separately with the
same settings. Inspect rank logs as well: successful HTTP alone does not prove
that global and local attention chunks, or FFN chunks, actually executed.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path


def request(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        body = response.read()
        return json.loads(body) if body else {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_response(response, expected_prompt_tokens):
    assert "error" not in response, response
    assert response["usage"]["prompt_tokens"] == expected_prompt_tokens, response["usage"]
    choice = response["choices"][0]
    assert choice["finish_reason"] in {"stop", "length"}, choice
    logprobs = choice["logprobs"]
    assert logprobs and logprobs["tokens"], "Missing output tokens/logprobs"
    scores = logprobs["token_logprobs"]
    assert len(scores) == len(logprobs["tokens"]) == response["usage"]["completion_tokens"]
    assert all(isinstance(x, (float, int)) and math.isfinite(x) for x in scores), "Non-finite/missing logprob"
    return {"tokens": logprobs["tokens"], "logprobs": scores, "text": choice["text"], "finish": choice["finish_reason"]}


def record(args):
    args.output.mkdir(parents=True, exist_ok=False)
    request(args.url, "/health")
    models = request(args.url, "/v1/models")
    assert args.model in {model["id"] for model in models["data"]}
    # Fixed token IDs make exact scheduler boundaries reproducible. The
    # tokenizer endpoint belongs to the same model as the tested service.
    text = "The ledger has red, blue and green entries. Each entry records a positive integer. " * 4000
    tokenized = request(args.url, "/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False})
    tokens = tokenized["tokens"]
    assert len(tokens) >= max(args.lengths)
    prompts = {length: tokens[:length] for length in args.lengths}
    manifest = {
        "model": args.model,
        "seed": 1234,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "lengths": args.lengths,
        "prompt_sha256": {str(length): digest(ids) for length, ids in prompts.items()},
        "cases": [],
    }
    (args.output / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")

    def one(length, repeat, mode):
        name = f"{mode}-{length}-r{repeat}"
        payload = {
            "model": args.model,
            "prompt": prompts[length],
            "temperature": 0,
            "seed": 1234,
            "max_tokens": args.max_tokens,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
        }
        start = time.monotonic()
        response = request(args.url, "/v1/completions", payload)
        # Preserve raw evidence before checking it, including failed cases.
        (args.output / f"{name}.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
        checked = validate_response(response, length)
        return {"name": name, "elapsed_seconds": time.monotonic() - start, **checked}

    try:
        for repeat in range(args.repeats):
            for length in args.lengths:
                manifest["cases"].append(one(length, repeat, "single"))
            # A short request can decode while a long request prefills. Actual
            # mixed scheduling still needs rank metadata evidence.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(one, n, repeat, "concurrent") for n in (min(args.lengths), max(args.lengths))
                ]
                manifest["cases"].extend(f.result() for f in futures)
        manifest["status"] = "passed_http_and_finite_logprobs"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        (args.output / "results.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def compare_results(base, candidate, logprob_atol=0.05):
    for key in ("model", "seed", "max_tokens", "repeats", "lengths", "prompt_sha256"):
        assert base[key] == candidate[key], f"Mismatched workload: {key}"
    assert base["status"] == candidate["status"] == "passed_http_and_finite_logprobs"
    expected_names = {
        f"{mode}-{length}-r{repeat}"
        for repeat in range(base["repeats"])
        for mode, lengths in (("single", base["lengths"]), ("concurrent", (min(base["lengths"]), max(base["lengths"]))))
        for length in lengths
    }
    failures = []
    states = []
    for name, state in (("baseline", base), ("candidate", candidate)):
        indexed = {case["name"]: case for case in state["cases"]}
        assert len(indexed) == len(state["cases"]) and set(indexed) == expected_names, f"Incomplete {name} cases"
        for case in indexed.values():
            assert case["tokens"] and len(case["tokens"]) == len(case["logprobs"])
            assert all(isinstance(x, (int, float)) and math.isfinite(x) for x in case["logprobs"])
        states.append(indexed)
        for case_name, case in indexed.items():
            reference = indexed[case_name.rsplit("-r", 1)[0] + "-r0"]
            if case["tokens"] != reference["tokens"] or case["finish"] != reference["finish"]:
                failures.append(f"{name} unstable: {case_name}")
    for name in sorted(expected_names):
        left, right = (state[name] for state in states)
        if left["tokens"] != right["tokens"] or left["finish"] != right["finish"]:
            failures.append(f"token/finish mismatch: {name}")
        elif any(abs(a - b) > logprob_atol for a, b in zip(left["logprobs"], right["logprobs"])):
            failures.append(f"logprob tolerance exceeded: {name}")
    return {"status": "failed" if failures else "passed", "failures": failures, "logprob_atol": logprob_atol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("record")
    run.add_argument("--url", required=True)
    run.add_argument("--model", default="dsv4")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--lengths", type=int, nargs="+", default=[1, 2047, 2048, 2049, 16383, 16384, 16385, 24577])
    run.add_argument("--max-tokens", type=int, default=16)
    run.add_argument("--repeats", type=int, default=3)
    diff = sub.add_parser("compare")
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--candidate", type=Path, required=True)
    diff.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record":
        if args.repeats < 2 or min(args.lengths) < 1 or len(set(args.lengths)) != len(args.lengths):
            parser.error("Use at least two repeats and unique positive lengths")
        record(args)
    else:
        result = compare_results(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise SystemExit(int(result["status"] != "passed"))


if __name__ == "__main__":
    main()
