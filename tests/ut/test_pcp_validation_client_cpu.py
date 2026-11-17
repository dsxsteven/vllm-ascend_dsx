# SPDX-License-Identifier: Apache-2.0
"""The service validator must fail closed on missing or unstable evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pcp_service_validator", Path(__file__).resolve().parents[1] / "e2e/pcp_attention_chunk_validation.py"
)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def results():
    return dict(
        model="test",
        seed=1234,
        max_tokens=2,
        repeats=2,
        lengths=[1, 2049],
        prompt_sha256={"1": "first", "2049": "second"},
        status="passed_http_and_finite_logprobs",
        cases=[
            dict(name=f"{mode}-{length}-r{repeat}", tokens=["token_id:1"], logprobs=[-0.1], finish="length")
            for mode in ("single", "concurrent")
            for length in (1, 2049)
            for repeat in (0, 1)
        ],
    )


def test_matching_complete_runs_pass():
    assert client.compare_results(results(), results())["status"] == "passed"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "prompt", "empty"])
def test_invalid_evidence_rejected(mutation):
    baseline, candidate = results(), results()
    if mutation == "missing":
        candidate["cases"].pop()
    elif mutation == "duplicate":
        candidate["cases"].append(copy.deepcopy(candidate["cases"][0]))
    elif mutation == "nan":
        candidate["cases"][0]["logprobs"][0] = float("nan")
    elif mutation == "prompt":
        candidate["prompt_sha256"]["1"] = "different"
    else:
        candidate["cases"][0]["tokens"] = []
    with pytest.raises(AssertionError):
        client.compare_results(baseline, candidate)


@pytest.mark.parametrize("mutation", ["tokens", "scores", "finish", "unstable_baseline"])
def test_regressions_and_unstable_reference_cannot_pass(mutation):
    baseline, candidate = results(), results()
    if mutation == "tokens":
        candidate["cases"][0]["tokens"] = ["token_id:2"]
    elif mutation == "scores":
        candidate["cases"][0]["logprobs"] = [-0.2]
    elif mutation == "finish":
        candidate["cases"][0]["finish"] = "stop"
    else:
        baseline["cases"][0]["tokens"] = ["token_id:2"]
        candidate = copy.deepcopy(baseline)
    assert client.compare_results(baseline, candidate)["status"] == "failed"
