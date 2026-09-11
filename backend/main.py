"""Run a simple retrieval-augmented chat over the indexed Compileit pages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()


DEFAULT_INDEX_DIR = Path(__file__).parent / "scraping" / "results" / "compileit_index"
DEFAULT_COLLECTION_NAME = "compileit_chunks"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-5.6-luna"
RETRIEVAL_COUNT = 5


class RAGState(MessagesState):
    retrieved_context: str
    sources: list[dict[str, str]]


def load_index_manifest(index_dir: Path) -> dict[str, object]:
    manifest_path = index_dir / "index-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No index found at {index_dir}. Run scraping/index_chunks.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_graph(index_dir: Path = DEFAULT_INDEX_DIR):
    """Load the persistent index and build the retrieve-then-answer graph."""

    manifest = load_index_manifest(index_dir)
    embedding_model = str(manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    collection_name = str(manifest.get("collection_name", DEFAULT_COLLECTION_NAME))
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(index_dir / "chroma"),
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_COUNT},
    )
    llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL))

    def retrieve_documents(state: RAGState) -> dict[str, object]:
        question = latest_question(state)
        documents = retriever.invoke(question)

        context_parts: list[str] = []
        sources: list[dict[str, str]] = []
        seen_source_urls: set[str] = set()
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
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
        }

    def generate_answer(state: RAGState) -> dict[str, object]:
        question = latest_question(state)
        system_prompt = (
            "You answer questions about the Compileit website. Use only the retrieved "
            "context provided to answer. If the context does not contain enough "
            "information, say so clearly instead of guessing. Answer in the same "
            "language as the user's question."
        )
        context_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{state.get('retrieved_context', '(none)')}"
        )
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context_prompt),
            ]
        )

        source_lines = [
            f"- [{source['title']}]({source['url']})" for source in state.get("sources", [])
        ]
        answer = str(response.content)
        if source_lines:
            answer += "\n\nSources:\n" + "\n".join(source_lines)

        return {"messages": [AIMessage(content=answer)]}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


def latest_question(state: RAGState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("The graph requires at least one human question")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to ask about Compileit")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    arguments = parse_args()
    question = arguments.question or input("Question: ")
    graph = build_graph(arguments.index_dir)
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
