"""Evaluate the chat through its streaming HTTP API."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import (
    CHAT_INPUT_COST_PER_1M,
    CHAT_OUTPUT_COST_PER_1M,
    DEFAULT_CHAT_MODEL,
    EMBEDDING_INPUT_COST_PER_1M,
)
from usage import calculate_cost, normalize_usage


DEFAULT_DATASET = Path(__file__).parent / "dataset.json"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"

load_dotenv(Path(__file__).parents[1] / ".env")


def environment_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as dataset_file:
        data = json.load(dataset_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")

    cases: list[dict[str, Any]] = []
    for index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Expected an object at index {index} of {path}")
        for required_field in ("id", "question", "expected_sources", "answer_requirements"):
            if required_field not in case:
                raise ValueError(f"Dataset item {index} is missing {required_field}")
        if not isinstance(case["answer_requirements"], list) or not all(
            isinstance(requirement, str) for requirement in case["answer_requirements"]
        ):
            raise ValueError(
                f"Dataset item {index} must contain a list of string answer requirements"
            )
        cases.append(case)
    return cases


def normalize_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    return normalized or "/"


def parse_sse(response: Any):
    event_name = "message"
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = "message"
            data_lines = []
        elif line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())


def call_streaming_api(base_url: str, question: str, timeout: float) -> dict[str, Any]:
    started = perf_counter()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/stream",
        data=json.dumps(
            {"messages": [{"role": "user", "content": question}]},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    answer_parts: list[str] = []
    sources: list[dict[str, str]] = []
    evidence: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    ttft_ms: float | None = None
    ttlt_ms: float | None = None
    done_ms: float | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for event_name, raw_data in parse_sse(response):
                data = json.loads(raw_data)
                now_ms = (perf_counter() - started) * 1_000
                if event_name == "token":
                    text = str(data.get("text", ""))
                    if text:
                        if ttft_ms is None:
                            ttft_ms = now_ms
                        ttlt_ms = now_ms
                        answer_parts.append(text)
                elif event_name == "sources":
                    sources = list(data.get("sources", []))
                elif event_name == "evidence":
                    evidence = list(data.get("evidence", []))
                elif event_name == "usage":
                    usage = data
                elif event_name == "error":
                    raise RuntimeError(str(data.get("message", "The API returned an error.")))
                elif event_name == "done":
                    done_ms = now_ms
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    return {
        "answer": "".join(answer_parts),
        "sources": sources,
        "evidence": evidence,
        "usage": usage,
        "timing_ms": {"ttft": ttft_ms, "ttlt": ttlt_ms, "done": done_ms},
    }


def score_sources(expected_sources: list[str], returned_sources: list[dict[str, str]]) -> dict[str, Any]:
    expected = {normalize_url(url) for url in expected_sources}
    returned = {normalize_url(str(source.get("url", ""))) for source in returned_sources}
    missing = sorted(expected - returned)
    unexpected = sorted(returned - expected)
    return {
        "source_correct": not missing,
        "missing_sources": missing,
        "unexpected_sources": unexpected,
    }


def judge_answer(case: dict[str, Any], answer: str, model_name: str = DEFAULT_CHAT_MODEL) -> dict[str, Any]:
    judge = ChatOpenAI(model=model_name)
    answer_requirements = "\n".join(
        f"{index}. {requirement}"
        for index, requirement in enumerate(case["answer_requirements"], start=1)
    )
    response = judge.invoke(
        [
            SystemMessage(
                content=(
                    "Evaluate how well an answer satisfies the answer requirements for a "
                    "website question. Return only valid JSON with exactly these keys: "
                    "score (number from 0 to 1) and reason (short string). Give 1.0 only "
                    "when all requirements are satisfied. Lower the score for missing "
                    "requirements, violated restrictions, or materially misleading "
                    "content. Judge only the answer text against the listed requirements; "
                    "do not judge the separate source list. Harmless connective wording "
                    "and clear paraphrases are allowed unless a requirement says otherwise."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{case['question']}\n\n"
                    f"Answer requirements:\n{answer_requirements}\n\n"
                    f"Actual answer:\n{answer}"
                )
            ),
        ]
    )
    raw_content = str(response.content).strip()
    if raw_content.startswith("```"):
        raw_content = raw_content.strip("`").removeprefix("json").strip()
    try:
        judgment = json.loads(raw_content)
    except json.JSONDecodeError:
        judgment = {"score": None, "reason": f"Judge returned non-JSON output: {raw_content}"}
    score = judgment.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        judgment["score"] = None
        judgment["reason"] = judgment.get("reason") or "Judge returned an invalid score."
    else:
        judgment["score"] = float(score)
    judgment["usage"] = normalize_usage(getattr(response, "usage_metadata", None))
    return judgment


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def git_revision(project_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main() -> int:
    base_url = os.getenv("EVAL_API_URL", "http://localhost:8000")
    timeout = environment_float("EVAL_TIMEOUT_SECONDS") or 180.0
    chat_prices = {
        "input": CHAT_INPUT_COST_PER_1M,
        "output": CHAT_OUTPUT_COST_PER_1M,
    }
    embedding_prices = {
        "input": EMBEDDING_INPUT_COST_PER_1M,
        "output": None,
    }
    judge_prices = {
        "input": CHAT_INPUT_COST_PER_1M,
        "output": CHAT_OUTPUT_COST_PER_1M,
    }

    cases = load_dataset(DEFAULT_DATASET)
    if not cases:
        raise ValueError("No evaluation cases selected")

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"Running {case['id']}...", flush=True)
        try:
            result = call_streaming_api(base_url, case["question"], timeout)
            result.update(score_sources(case["expected_sources"], result["sources"]))
            result["id"] = case["id"]
            result["question"] = case["question"]
            result["expected_sources"] = case["expected_sources"]
            operations = result.get("usage", {}).get("operations", {})
            result["total_cost_usd"] = result.get("usage", {}).get("total_cost_usd")
            if result["total_cost_usd"] is None and isinstance(operations, dict):
                result["total_cost_usd"] = calculate_cost(
                    operations,
                    {
                        "chat": chat_prices,
                        "embedding": embedding_prices,
                    },
                )
            judgment = judge_answer(case, result["answer"])
            result["answer_score"] = judgment.get("score")
            result["answer_judge_reason"] = judgment.get("reason")
            result["judge_usage"] = judgment.get("usage")
            result["judge_cost_usd"] = calculate_cost(
                {"chat": judgment.get("usage")},
                {"chat": judge_prices},
            )
            result["status"] = "ok"
        except Exception as error:
            result = {
                "id": case["id"],
                "question": case["question"],
                "status": "error",
                "error": str(error),
            }
        results.append(result)

    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = DEFAULT_RESULTS_DIR / f"{run_name}.json"
    result_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    successful = [result for result in results if result.get("status") == "ok"]
    timings = [result["timing_ms"] for result in successful if result.get("timing_ms")]
    costs = [result["total_cost_usd"] for result in successful if result.get("total_cost_usd") is not None]
    judge_costs = [result["judge_cost_usd"] for result in successful if result.get("judge_cost_usd") is not None]
    summary = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(Path(__file__).parents[1]),
        "dataset": str(DEFAULT_DATASET),
        "base_url": base_url,
        "case_count": len(results),
        "successful_count": len(successful),
        "error_count": len(results) - len(successful),
        "source_correctness_rate": (
            sum(bool(result.get("source_correct")) for result in successful) / len(successful)
            if successful
            else None
        ),
        "answer_score_average": (
            sum(result["answer_score"] for result in successful if result.get("answer_score") is not None)
            / sum(result.get("answer_score") is not None for result in successful)
            if any(result.get("answer_score") is not None for result in successful)
            else None
        ),
        "total_application_cost_usd": sum(costs) if costs else None,
        "total_judge_cost_usd": sum(judge_costs) if judge_costs else None,
        "latency_ms": {
            "ttft_median": median([timing["ttft"] for timing in timings if timing.get("ttft") is not None])
            if any(timing.get("ttft") is not None for timing in timings)
            else None,
            "ttft_p95": percentile([timing["ttft"] for timing in timings if timing.get("ttft") is not None], 0.95),
            "ttlt_median": median([timing["ttlt"] for timing in timings if timing.get("ttlt") is not None])
            if any(timing.get("ttlt") is not None for timing in timings)
            else None,
            "ttlt_p95": percentile([timing["ttlt"] for timing in timings if timing.get("ttlt") is not None], 0.95),
        },
    }
    summary_path = DEFAULT_RESULTS_DIR / f"{run_name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(result_path), "summary": str(summary_path), **summary}, ensure_ascii=False))
    return 0 if summary["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
