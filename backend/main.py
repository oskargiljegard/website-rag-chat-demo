"""Ask a question about Compileit from the command line."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage

from api import app  # Re-export the FastAPI app for `uv run fastapi dev`.
from rag import DEFAULT_INDEX_DIR, build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to ask about Compileit")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    return parser.parse_args()


async def answer_question(question: str, index_dir: Path):
    graph = build_graph(index_dir)
    return await graph.ainvoke({"messages": [HumanMessage(content=question)]})


def main() -> int:
    arguments = parse_args()
    question = arguments.question or input("Question: ")
    result = asyncio.run(answer_question(question, arguments.index_dir))
    print(result["messages"][-1].content)
    sources = result.get("sources", [])
    if sources:
        print("\nSources:")
        for source in sources:
            print(f"- {source['title']}: {source['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
