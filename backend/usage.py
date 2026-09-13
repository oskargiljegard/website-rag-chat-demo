"""Token usage normalization and optional cost calculation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def normalize_usage(value: Any) -> dict[str, int] | None:
    """Normalize common provider/LangChain usage field names."""

    raw = _as_mapping(value)
    if not raw:
        return None

    input_tokens = raw.get("input_tokens", raw.get("prompt_tokens"))
    output_tokens = raw.get("output_tokens", raw.get("completion_tokens"))
    total_tokens = raw.get("total_tokens")
    normalized: dict[str, int] = {}

    if input_tokens is not None:
        normalized["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        normalized["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        normalized["total_tokens"] = int(total_tokens)
    elif "input_tokens" in normalized or "output_tokens" in normalized:
        normalized["total_tokens"] = normalized.get("input_tokens", 0) + normalized.get(
            "output_tokens", 0
        )

    return normalized or None


def merge_usage(*usages: dict[str, int] | None) -> dict[str, int] | None:
    """Add usage records from multiple model calls."""

    totals: dict[str, int] = {}
    for usage in usages:
        if not usage:
            continue
        for key, value in usage.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals or None


def calculate_cost(
    usage_by_operation: Mapping[str, dict[str, int] | None],
    prices_by_operation: Mapping[str, Mapping[str, float | None]],
) -> float | None:
    """Calculate total USD cost when a price is configured for every used token type."""

    total = 0.0
    saw_usage = False
    for operation, usage in usage_by_operation.items():
        if not usage:
            continue
        prices = prices_by_operation.get(operation, {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        if input_tokens:
            saw_usage = True
            input_price = prices.get("input")
            if input_price is None:
                return None
            total += input_tokens * input_price / 1_000_000
        if output_tokens:
            saw_usage = True
            output_price = prices.get("output")
            if output_price is None:
                return None
            total += output_tokens * output_price / 1_000_000
    return total if saw_usage else None
