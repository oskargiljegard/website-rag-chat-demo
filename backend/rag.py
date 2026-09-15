"""Shared configuration and utilities for the Compileit RAG graphs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import MessagesState
from pydantic import Field

from usage import merge_usage, normalize_usage

load_dotenv()


DEFAULT_INDEX_DIR = Path(__file__).parent / "scraping" / "results" / "compileit_index"
DEFAULT_COLLECTION_NAME = "compileit_chunks"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_INPUT_COST_PER_1M = 0.02
DEFAULT_CHAT_MODEL = "gpt-5.6-luna"
CHAT_INPUT_COST_PER_1M = 0.20
CHAT_OUTPUT_COST_PER_1M = 1.20
RETRIEVAL_COUNT = 5
DEFAULT_GRAPH = "agent"


class RAGState(MessagesState):
    """Common state shape for callers that import the legacy state name."""

    retrieved_context: str
    sources: list[dict[str, str]]
    evidence: list[dict[str, object]]
    usage: dict[str, object]


class UsageTrackingOpenAIEmbeddings(OpenAIEmbeddings):
    """Track usage for query embeddings without changing index construction."""

    last_query_usage: dict[str, int] | None = Field(default=None, exclude=True)

    def embed_query(self, text: str, **kwargs: object) -> list[float]:
        self._ensure_sync_client_available()
        response = self.client.create(
            input=[text],
            **{**self._invocation_params, **kwargs},
        )
        if not isinstance(response, dict):
            response = response.model_dump()
        self.last_query_usage = normalize_usage(response.get("usage"))
        return list(response["data"][0]["embedding"])

    def consume_query_usage(self) -> dict[str, int] | None:
        usage = self.last_query_usage
        self.last_query_usage = None
        return usage


def load_index_manifest(index_dir: Path) -> dict[str, object]:
    manifest_path = index_dir / "index-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No index found at {index_dir}. Run scraping/index_chunks.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_retriever(index_dir: Path = DEFAULT_INDEX_DIR):
    """Load the persistent index and return its retriever plus usage tracker."""

    manifest = load_index_manifest(index_dir)
    embedding_model = str(manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    collection_name = str(manifest.get("collection_name", DEFAULT_COLLECTION_NAME))
    embeddings = UsageTrackingOpenAIEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(index_dir / "chroma"),
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_COUNT},
    )
    return retriever, embeddings


def build_chat_model(**kwargs: Any) -> ChatOpenAI:
    return ChatOpenAI(model=DEFAULT_CHAT_MODEL, **kwargs)


def chat_usage(response: AIMessage) -> dict[str, int] | None:
    response_metadata = getattr(response, "response_metadata", {})
    token_usage = (
        response_metadata.get("token_usage", {})
        if isinstance(response_metadata, dict)
        else {}
    )
    usage = normalize_usage(getattr(response, "usage_metadata", None))
    return usage if usage is not None else normalize_usage(token_usage)


def latest_question(state: dict[str, Any]) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("The graph requires at least one human question")


def conversation_for_prompt(state: dict[str, Any], max_messages: int = 8) -> str:
    """Format recent chat history without including the current question twice."""

    messages = list(state["messages"])
    if messages and isinstance(messages[-1], HumanMessage):
        messages = messages[:-1]
    messages = messages[-max_messages:]
    if not messages:
        return "(no previous messages)"

    lines: list[str] = []
    for message in messages:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


def documents_payload(
    documents: Sequence[Document], embedding_usage: dict[str, int] | None
) -> dict[str, object]:
    """Turn retrieved documents into the common context/source/evidence shape."""

    context_parts: list[str] = []
    sources: list[dict[str, str]] = []
    evidence: list[dict[str, object]] = []
    seen_source_urls: set[str] = set()
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        evidence.append(
            {
                "chunk_id": str(metadata.get("chunk_id", "")),
                "source_url": str(metadata.get("source_url", "")),
                "page_title": str(metadata.get("page_title", "")),
                "heading_path": str(metadata.get("heading_path", "")),
                "chunk_index": int(metadata.get("chunk_index", index)),
                "excerpt": str(document.page_content)[:2_000],
            }
        )
        context_parts.append(
            "\n".join(
                [
                    f"[Retrieved chunk {index}]",
                    f"Title: {metadata.get('page_title', '')}",
                    f"URL: {metadata.get('source_url', '')}",
                    f"Section: {metadata.get('heading_path', '')}",
                    str(document.page_content),
                ]
            )
        )

        source_url = str(metadata.get("source_url", ""))
        if source_url and source_url not in seen_source_urls:
            seen_source_urls.add(source_url)
            sources.append(
                {
                    "title": str(metadata.get("page_title", "Compileit")),
                    "url": source_url,
                }
            )

    return {
        "retrieved_context": "\n\n".join(context_parts),
        "sources": sources,
        "evidence": evidence,
        "usage": {"embedding": embedding_usage},
    }


def aggregate_documents_payloads(payloads: Sequence[dict[str, object]]) -> dict[str, object]:
    """Combine all tool calls while deduplicating source and evidence metadata."""

    context_parts: list[str] = []
    sources: list[dict[str, str]] = []
    evidence: list[dict[str, object]] = []
    seen_source_urls: set[str] = set()
    seen_chunk_ids: set[str] = set()
    embedding_usages: list[dict[str, int] | None] = []

    for payload in payloads:
        context = payload.get("retrieved_context")
        if isinstance(context, str) and context:
            context_parts.append(context)

        payload_sources = payload.get("sources", [])
        if isinstance(payload_sources, list):
            for source in payload_sources:
                if not isinstance(source, dict):
                    continue
                url = str(source.get("url", ""))
                if url and url not in seen_source_urls:
                    seen_source_urls.add(url)
                    sources.append(
                        {"title": str(source.get("title", "Compileit")), "url": url}
                    )

        payload_evidence = payload.get("evidence", [])
        if isinstance(payload_evidence, list):
            for item in payload_evidence:
                if not isinstance(item, dict):
                    continue
                chunk_id = str(item.get("chunk_id", ""))
                if chunk_id and chunk_id in seen_chunk_ids:
                    continue
                if chunk_id:
                    seen_chunk_ids.add(chunk_id)
                evidence.append(dict(item))

        usage = payload.get("usage", {})
        if isinstance(usage, dict):
            embedding_usages.append(usage.get("embedding"))

    return {
        "retrieved_context": "\n\n".join(context_parts),
        "sources": sources,
        "evidence": evidence,
        "usage": {"embedding": merge_usage(*embedding_usages)},
    }


def build_graph(index_dir: Path = DEFAULT_INDEX_DIR):
    """Build the configured graph; set ``RAG_GRAPH=simple`` to use the legacy one."""

    graph_name = os.getenv("RAG_GRAPH", DEFAULT_GRAPH).strip().lower()
    if graph_name == "simple":
        from simple_graph import build_simple_graph

        return build_simple_graph(index_dir)
    if graph_name in {"agent", "agentic"}:
        from agent_graph import build_agent_graph

        return build_agent_graph(index_dir)
    raise ValueError("RAG_GRAPH must be either 'agent' or 'simple'")
