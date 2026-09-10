"""Extract and chunk the HTML pages produced by the Compileit crawler."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Union


DEFAULT_CRAWL_DIR = Path(__file__).parent / "results" / "compileit_crawl"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "compileit_chunks"
DEFAULT_MAX_WORDS = 350

NodeChild = Union["Node", str]


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[NodeChild] = field(default_factory=list)


class DocumentParser(HTMLParser):
    """Build a small DOM tree using only the Python standard library."""

    VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack: list[Node] = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {key.lower(): value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if node.tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {key.lower(): value or "" for key, value in attrs})
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].children.append(data)


@dataclass
class Block:
    tag: str
    text: str


@dataclass
class PageExtraction:
    title: str
    canonical_url: str | None
    blocks: list[Block]
    duplicate_blocks_removed: int
    warnings: list[str]


SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "nav",
    "header",
    "footer",
    "form",
}

SKIP_MARKERS = {
    "cookie",
    "consent",
    "modal",
    "overlay",
    "popup",
    "site-header",
    "site-footer",
    "newsletter-popup",
    "navigation",
    "navbar",
    "mobile-menu",
}

EXPLICIT_BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "blockquote",
    "pre",
    "td",
    "th",
    "figcaption",
    "dt",
    "dd",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def is_skipped(node: Node) -> bool:
    if node.tag in SKIP_TAGS:
        return True
    if node.attrs.get("hidden") is not None:
        return True
    if node.attrs.get("aria-hidden", "").casefold() == "true":
        return True

    class_and_id = f"{node.attrs.get('class', '')} {node.attrs.get('id', '')}".casefold()
    return any(marker in class_and_id for marker in SKIP_MARKERS)


def text_content(node: Node, *, ignore_nested_blocks: bool = False) -> str:
    if node.tag == "img":
        alt = normalize_text(node.attrs.get("alt", ""))
        if alt and alt.casefold() not in {"image", "logo", "icon"}:
            return f"Image: {alt}"
        return ""

    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
            continue
        if is_skipped(child):
            continue
        if child.tag == "br":
            parts.append("\n")
            continue
        if ignore_nested_blocks and (
            child.tag in EXPLICIT_BLOCK_TAGS or child.tag in {"ul", "ol", "table"}
        ):
            continue
        parts.append(text_content(child, ignore_nested_blocks=ignore_nested_blocks))
    return normalize_text(" ".join(parts))


def find_first(node: Node, tag: str) -> Node | None:
    if node.tag == tag:
        return node
    for child in node.children:
        if isinstance(child, Node):
            result = find_first(child, tag)
            if result is not None:
                return result
    return None


def find_all(node: Node, tag: str) -> list[Node]:
    matches: list[Node] = []
    if node.tag == tag:
        matches.append(node)
    for child in node.children:
        if isinstance(child, Node):
            matches.extend(find_all(child, tag))
    return matches


def has_block_descendant(node: Node) -> bool:
    for child in node.children:
        if not isinstance(child, Node) or is_skipped(child):
            continue
        if child.tag in EXPLICIT_BLOCK_TAGS or child.tag == "table":
            return True
        if has_block_descendant(child):
            return True
    return False


def table_text(node: Node) -> str:
    rows: list[str] = []
    for row in find_all(node, "tr"):
        cells = [child for child in row.children if isinstance(child, Node) and child.tag in {"td", "th"}]
        values = [text_content(cell) for cell in cells]
        values = [value for value in values if value]
        if values:
            rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def extract_blocks(node: Node, *, is_root: bool = False) -> list[Block]:
    if is_skipped(node):
        return []

    if node.tag == "table":
        text = table_text(node)
        return [Block("table", text)] if text else []

    # Carousel cards and small article cards often contain a heading plus a
    # date or CTA in a non-block span. Keep the card together for extraction.
    is_card = node.attrs.get("role", "").casefold() == "group"
    is_small_article = (
        not is_root
        and node.tag == "article"
        and bool(find_all(node, "h1") or find_all(node, "h2") or find_all(node, "h3"))
        and not find_all(node, "p")
    )
    if is_card or is_small_article:
        text = text_content(node)
        return [Block("card", text)] if text else []

    if node.tag in EXPLICIT_BLOCK_TAGS:
        text = text_content(node, ignore_nested_blocks=True)
        return [Block(node.tag, text)] if text else []

    # Capture text-only cards and labels without turning every container into
    # a duplicate block.
    if node.tag in {"div", "section", "aside"} and not has_block_descendant(node):
        text = text_content(node)
        return [Block("container", text)] if text else []

    blocks: list[Block] = []
    for child in node.children:
        if isinstance(child, Node):
            blocks.extend(extract_blocks(child))
    return blocks


def semantically_duplicate_card(block: Block, previous: list[Block]) -> bool:
    if block.tag != "card":
        return False
    current_tokens = text_tokens(block.text)
    if len(current_tokens) < 12:
        return False

    recent = previous[-10:]
    for window_size in range(1, min(5, len(recent) + 1)):
        for start in range(len(recent) - window_size + 1):
            earlier_tokens = set().union(
                *(text_tokens(item.text) for item in recent[start : start + window_size])
            )
            if len(current_tokens & earlier_tokens) / len(current_tokens) >= 0.8:
                return True
    return False


def deduplicate_blocks(blocks: list[Block]) -> tuple[list[Block], int]:
    seen: set[str] = set()
    result: list[Block] = []
    removed = 0
    for block in blocks:
        key = normalize_text(block.text).casefold()
        if not key or key in seen or semantically_duplicate_card(block, result):
            removed += 1
            continue
        seen.add(key)
        result.append(block)
    return result, removed


def heading_level(tag: str) -> int | None:
    if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
        return int(tag[1])
    return None


def split_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def make_chunks(blocks: list[Block], max_words: int) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    heading_path: list[str] = []
    current_path: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current_parts, current_words
        if not current_parts:
            return
        text = "\n\n".join(part for part in current_parts if part)
        chunks.append(
            {
                "heading_path": current_path.copy(),
                "text": text,
                "word_count": len(text.split()),
            }
        )
        current_parts = []
        current_words = 0

    for block in blocks:
        level = heading_level(block.tag)
        if level is not None:
            flush()
            heading_path = heading_path[: level - 1]
            heading_path.append(block.text)
            current_path = heading_path.copy()
            continue

        pieces = split_words(block.text, max_words)
        for piece in pieces:
            piece_words = len(piece.split())
            if current_parts and current_words + piece_words > max_words:
                flush()
                current_path = heading_path.copy()
            current_parts.append(piece)
            current_words += piece_words
    flush()
    return chunks


def page_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]


def parse_page(html: bytes, charset: str | None) -> PageExtraction:
    warnings: list[str] = []
    try:
        decoded = html.decode(charset or "utf-8", errors="replace")
    except LookupError:
        warnings.append(f"unknown_charset:{charset}")
        decoded = html.decode("utf-8", errors="replace")

    parser = DocumentParser()
    parser.feed(decoded)
    root = parser.root

    title_node = find_first(root, "title")
    title = text_content(title_node) if title_node else ""
    canonical_node = next(
        (
            node
            for node in find_all(root, "link")
            if "canonical" in node.attrs.get("rel", "").casefold() and node.attrs.get("href")
        ),
        None,
    )
    canonical_url = canonical_node.attrs["href"] if canonical_node else None

    candidates = [
        (find_first(root, "main"), "main"),
        (find_first(root, "article"), "article"),
        (find_first(root, "body"), "body"),
        (root, "document"),
    ]
    blocks: list[Block] = []
    selected_root_name = "document"
    for candidate, candidate_name in candidates:
        if candidate is None:
            continue
        blocks = extract_blocks(candidate, is_root=True)
        if blocks:
            selected_root_name = candidate_name
            break

    if selected_root_name != "main":
        warnings.append(f"content_root:{selected_root_name}")
    if not blocks:
        warnings.append("no_content_blocks")

    blocks, duplicate_blocks_removed = deduplicate_blocks(blocks)
    if duplicate_blocks_removed:
        warnings.append(f"duplicate_blocks_removed:{duplicate_blocks_removed}")
    return PageExtraction(title, canonical_url, blocks, duplicate_blocks_removed, warnings)


def markdown_for_chunk(
    chunk: dict[str, object],
    index: int,
    total: int,
    url: str,
    title: str,
) -> str:
    heading_path = chunk["heading_path"]
    section = " > ".join(heading_path) if heading_path else "Unsectioned content"
    return (
        f"# Chunk {index:03d}\n\n"
        f"**Title:** {title or 'Untitled'}  \n"
        f"**URL:** {url}  \n"
        f"**Chunk:** {index} of {total}  \n"
        f"**Section:** {section}  \n\n"
        f"{chunk['text']}\n"
    )


def load_page_rows(crawl_dir: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(crawl_dir / "crawl.db")
    connection.row_factory = sqlite3.Row
    rows = list(
        connection.execute(
            """
            SELECT * FROM pages
            WHERE status = 'fetched' AND content_type IN ('text/html', 'application/xhtml+xml')
            ORDER BY depth, url
            """
        )
    )
    connection.close()
    return rows


def process_pages(crawl_dir: Path, output_dir: Path, max_words: int) -> dict[str, object]:
    rows = load_page_rows(crawl_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_output_dir = output_dir / "pages"
    pages_output_dir.mkdir(parents=True, exist_ok=True)
    chunks_jsonl_path = output_dir / "chunks.jsonl"

    total_chunks = 0
    total_blocks = 0
    total_duplicates = 0
    processed = 0
    empty_pages = 0
    failed_pages = 0
    warning_counts: dict[str, int] = {}
    page_summaries: list[dict[str, object]] = []

    with chunks_jsonl_path.open("w", encoding="utf-8") as chunks_jsonl:
        for row in rows:
            url = row["url"]
            source_path = crawl_dir / row["raw_path"]
            current_page_id = page_id(url)
            page_output_dir = pages_output_dir / current_page_id
            page_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                extraction = parse_page(source_path.read_bytes(), row["charset"])
                chunks = make_chunks(extraction.blocks, max_words)
            except (OSError, UnicodeError, ValueError) as error:
                failed_pages += 1
                page_summaries.append({"url": url, "status": "failed", "error": str(error)})
                continue

            processed += 1
            total_blocks += len(extraction.blocks)
            total_duplicates += extraction.duplicate_blocks_removed
            total_chunks += len(chunks)
            if not chunks:
                empty_pages += 1

            for warning in extraction.warnings:
                warning_type = warning.split(":", 1)[0]
                warning_counts[warning_type] = warning_counts.get(warning_type, 0) + 1

            chunk_summaries: list[dict[str, object]] = []
            for index, chunk in enumerate(chunks, start=1):
                filename = f"chunk_{index:03d}.md"
                (page_output_dir / filename).write_text(
                    markdown_for_chunk(chunk, index, len(chunks), url, extraction.title),
                    encoding="utf-8",
                )
                record = {
                    "chunk_id": f"{current_page_id}-{index:03d}",
                    "source_url": url,
                    "canonical_url": extraction.canonical_url,
                    "page_title": extraction.title,
                    "heading_path": chunk["heading_path"],
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "word_count": chunk["word_count"],
                    "source_content_hash": row["sha256"],
                    "text": chunk["text"],
                }
                chunks_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_summaries.append(
                    {
                        "file": filename,
                        "heading_path": chunk["heading_path"],
                        "word_count": chunk["word_count"],
                    }
                )

            manifest = {
                "url": url,
                "canonical_url": extraction.canonical_url,
                "title": extraction.title,
                "content_hash": row["sha256"],
                "raw_path": row["raw_path"],
                "chunk_count": len(chunks),
                "extracted_block_count": len(extraction.blocks),
                "duplicate_blocks_removed": extraction.duplicate_blocks_removed,
                "warnings": extraction.warnings,
                "chunks": chunk_summaries,
            }
            (page_output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            page_summaries.append(
                {
                    "url": url,
                    "page_id": current_page_id,
                    "title": extraction.title,
                    "chunk_count": len(chunks),
                    "block_count": len(extraction.blocks),
                    "warnings": extraction.warnings,
                }
            )

    report = {
        "input": {
            "crawl_dir": str(crawl_dir),
            "fetched_html_pages": len(rows),
        },
        "configuration": {"max_words": max_words, "overlap_words": 0},
        "results": {
            "pages_processed": processed,
            "pages_failed": failed_pages,
            "pages_without_chunks": empty_pages,
            "total_blocks": total_blocks,
            "duplicate_blocks_removed": total_duplicates,
            "total_chunks": total_chunks,
            "warning_counts": warning_counts,
        },
        "output": {
            "chunks_jsonl": "chunks.jsonl",
            "pages_directory": "pages/",
        },
        "pages": page_summaries,
    }
    (output_dir / "chunk-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-dir", type=Path, default=DEFAULT_CRAWL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.max_words < 1:
        raise SystemExit("max-words must be positive")
    if not (arguments.crawl_dir / "crawl.db").exists():
        raise SystemExit(f"Missing crawl database: {arguments.crawl_dir / 'crawl.db'}")

    report = process_pages(arguments.crawl_dir, arguments.output_dir, arguments.max_words)
    print(json.dumps(report["results"], indent=2, ensure_ascii=False))
    return 0 if report["results"]["pages_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
