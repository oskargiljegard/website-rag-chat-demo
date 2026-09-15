"""FastAPI HTTP API for the Compileit streaming RAG chat."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import BaseModel, Field

from rag import (
    CHAT_INPUT_COST_PER_1M,
    CHAT_OUTPUT_COST_PER_1M,
    DEFAULT_INDEX_DIR,
    EMBEDDING_INPUT_COST_PER_1M,
    build_graph,
)
from usage import calculate_cost, merge_usage


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)


app = FastAPI(title="Compileit RAG API", version="0.1.0")
frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_graph():
    return build_graph(DEFAULT_INDEX_DIR)


def to_langchain_messages(messages: list[ChatMessage]) -> list[HumanMessage | AIMessage]:
    return [
        HumanMessage(content=message.content)
        if message.role == "user"
        else AIMessage(content=message.content)
        for message in messages
    ]


def text_from_message_chunk(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        return "".join(text_parts)
    return ""


def sse_event(event: str, data: object) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


def record_operation_usage(
    usage_by_operation: dict[str, dict[str, int] | None],
    update_usage: object,
) -> None:
    """Accumulate usage because an agent may call the model more than once."""

    if not isinstance(update_usage, dict):
        return
    for operation, usage in update_usage.items():
        if usage is None or isinstance(usage, dict):
            usage_by_operation[operation] = merge_usage(
                usage_by_operation.get(operation), usage
            )


async def chat_stream(request: Request, payload: ChatRequest) -> AsyncIterator[str]:
    try:
        messages = to_langchain_messages(payload.messages)
        sources: list[dict[str, str]] = []
        evidence: list[dict[str, object]] = []
        usage_by_operation: dict[str, dict[str, int] | None] = {}
        yield sse_event("status", {"state": "searching"})

        async for part in get_graph().astream(
            {"messages": messages},
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if await request.is_disconnected():
                return

            if part["type"] == "updates":
                for node_name, update in part["data"].items():
                    if node_name == "retrieve_documents":
                        sources = list(update.get("sources", []))
                        evidence = list(update.get("evidence", []))
                        record_operation_usage(usage_by_operation, update.get("usage", {}))
                        yield sse_event("status", {"state": "generating"})
                    elif node_name == "generate_answer":
                        record_operation_usage(usage_by_operation, update.get("usage", {}))
                    elif node_name == "agent":
                        record_operation_usage(usage_by_operation, update.get("usage", {}))
                        messages = update.get("messages", [])
                        last_message = messages[-1] if messages else None
                        state = (
                            "searching"
                            if isinstance(last_message, AIMessage) and last_message.tool_calls
                            else "generating"
                        )
                        yield sse_event("status", {"state": state})
                    elif node_name == "finalize":
                        sources = list(update.get("sources", []))
                        evidence = list(update.get("evidence", []))
                        record_operation_usage(usage_by_operation, update.get("usage", {}))
            elif part["type"] == "messages":
                message_chunk, metadata = part["data"]
                if metadata.get("langgraph_node") not in {"generate_answer", "agent"}:
                    continue
                # LangGraph emits incremental AIMessageChunks and then a final
                # complete AIMessage for the node. Only forward the chunks;
                # forwarding the final message would duplicate the answer.
                if not isinstance(message_chunk, AIMessageChunk):
                    continue
                text = text_from_message_chunk(message_chunk.content)
                if text:
                    yield sse_event("token", {"text": text})

        total_usage = merge_usage(*usage_by_operation.values())
        total_cost = calculate_cost(
            usage_by_operation,
            {
                "chat": {
                    "input": CHAT_INPUT_COST_PER_1M,
                    "output": CHAT_OUTPUT_COST_PER_1M,
                },
                "embedding": {
                    "input": EMBEDDING_INPUT_COST_PER_1M,
                    "output": None,
                },
            },
        )
        yield sse_event("sources", {"sources": sources})
        yield sse_event("evidence", {"evidence": evidence})
        yield sse_event(
            "usage",
            {
                "operations": usage_by_operation,
                "total": total_usage,
                "total_cost_usd": total_cost,
            },
        )
        yield sse_event("done", {})
    except Exception as error:
        yield sse_event("error", {"message": str(error)})


@app.get("/health")
def health() -> dict[str, object]:
    manifest_path = DEFAULT_INDEX_DIR / "index-manifest.json"
    return {
        "status": "ok",
        "index_available": manifest_path.is_file(),
    }


@app.post("/chat/stream")
async def stream_chat(request: Request, payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        chat_stream(request, payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
