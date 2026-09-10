"""Extract structured text from the saved homepage and create reviewable chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Union


HTML_PATH = Path(__file__).parent / "results" / "compileit_homepage.html"
OUTPUT_DIR = Path(__file__).parent / "results" / "homepage_chunks"
MAX_WORDS = 350

NodeChild = Union["Node", str]


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[NodeChild] = field(default_factory=list)


class DocumentParser(HTMLParser):
    """Build a small DOM tree using only Python's standard library."""

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
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag.lower():
            self.stack.pop()

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

NOISE_TEXTS = {
    "previous slide next slide",
    "next slide previous slide",
    "stoppa",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_noise_text(text: str) -> bool:
    return normalize_text(text).casefold() in NOISE_TEXTS


def is_skipped(node: Node) -> bool:
    if node.tag in SKIP_TAGS:
        return True

    class_and_id = f"{node.attrs.get('class', '')} {node.attrs.get('id', '')}".lower()
    markers = {marker for marker in SKIP_MARKERS if marker in class_and_id}
    return bool(markers) or node.attrs.get("aria-hidden", "").lower() == "true"


def text_content(node: Node, *, ignore_nested_blocks: bool = False) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
            continue
        if is_skipped(child):
            continue
        if ignore_nested_blocks and (
            child.tag in EXPLICIT_BLOCK_TAGS or child.tag in {"ul", "ol"}
        ):
            continue
        parts.append(text_content(child, ignore_nested_blocks=ignore_nested_blocks))
    return normalize_text(" ".join(parts))


def has_block_descendant(node: Node) -> bool:
    for child in node.children:
        if not isinstance(child, Node) or is_skipped(child):
            continue
        if child.tag in EXPLICIT_BLOCK_TAGS:
            return True
        if has_block_descendant(child):
            return True
    return False


def find_first(node: Node, tag: str) -> Node | None:
    if node.tag == tag:
        return node
    for child in node.children:
        if isinstance(child, Node):
            result = find_first(child, tag)
            if result is not None:
                return result
    return None


def extract_blocks(node: Node) -> list[Block]:
    if is_skipped(node):
        return []

    # News cards on the homepage use carousel slide containers. Keep the
    # headline and date together instead of dropping the date-only span.
    if node.attrs.get("role") == "group":
        text = text_content(node)
        return [Block("article", text)] if text and not is_noise_text(text) else []

    if node.tag in EXPLICIT_BLOCK_TAGS:
        text = text_content(node, ignore_nested_blocks=True)
        return [Block(node.tag, text)] if text and not is_noise_text(text) else []

    # Some cards use only div/span elements. Treat a text-only container as one
    # block, while recursively processing containers that have real structure.
    if node.tag in {"div", "article", "section", "aside"} and not has_block_descendant(node):
        text = text_content(node)
        return [Block(node.tag, text)] if text and not is_noise_text(text) else []

    blocks: list[Block] = []
    for child in node.children:
        if isinstance(child, Node):
            blocks.extend(extract_blocks(child))
    return blocks


def deduplicate_blocks(blocks: list[Block]) -> tuple[list[Block], int]:
    seen: set[tuple[str, str]] = set()
    result: list[Block] = []
    removed = 0
    for block in blocks:
        key = (block.tag, normalize_text(block.text).casefold())
        if key in seen or is_semantically_duplicate_article(block, result):
            removed += 1
            continue
        seen.add(key)
        result.append(block)
    return result, removed


def comparison_tokens(text: str) -> set[str]:
    """Return content words used for a tolerant duplicate comparison."""
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def is_semantically_duplicate_article(block: Block, previous: list[Block]) -> bool:
    """Detect responsive/card copies whose wording already appeared above."""
    if block.tag != "article":
        return False

    block_tokens = comparison_tokens(block.text)
    # Short news cards should not be removed just because they share a word
    # such as "AI" with an earlier service section.
    if len(block_tokens) < 12:
        return False

    recent = previous[-10:]
    for window_size in range(1, min(5, len(recent) + 1)):
        for start in range(len(recent) - window_size + 1):
            earlier_tokens = set().union(
                *(comparison_tokens(item.text) for item in recent[start : start + window_size])
            )
            overlap = len(block_tokens & earlier_tokens) / len(block_tokens)
            if overlap >= 0.8:
                return True
    return False


def heading_level(tag: str) -> int | None:
    if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
        return int(tag[1])
    return None


def format_block(block: Block) -> str:
    if block.tag == "li":
        return f"- {block.text}"
    if block.tag in {"pre", "td", "th"}:
        return block.text
    return block.text


def make_chunks(blocks: list[Block]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    heading_path: list[str] = []
    current_blocks: list[Block] = []
    current_path: list[str] = []

    def current_word_count() -> int:
        return sum(len(block.text.split()) for block in current_blocks)

    def flush() -> None:
        nonlocal current_blocks
        if not current_blocks:
            return
        text_parts = [format_block(block) for block in current_blocks]
        text = "\n\n".join(part for part in text_parts if part)
        if text.strip():
            chunks.append(
                {
                    "heading_path": current_path.copy(),
                    "text": text,
                    "word_count": len(text.split()),
                }
            )
        current_blocks = []

    for block in blocks:
        level = heading_level(block.tag)
        if level is not None:
            flush()
            heading_path = heading_path[: level - 1]
            heading_path.append(block.text)
            current_path = heading_path.copy()
            continue

        if current_blocks and current_word_count() + len(block.text.split()) > MAX_WORDS:
            flush()
            current_path = heading_path.copy()
        current_blocks.append(block)

    flush()
    return chunks


def write_chunks(chunks: list[dict[str, object]], metadata: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in OUTPUT_DIR.glob("chunk_*.md"):
        path.unlink()

    for index, chunk in enumerate(chunks, start=1):
        path = OUTPUT_DIR / f"chunk_{index:03d}.md"
        heading_path = chunk["heading_path"]
        heading = " > ".join(heading_path) if heading_path else "Homepage"
        content = (
            f"# Chunk {index:03d}\n\n"
            f"**Section:** {heading}\n\n"
            f"{chunk['text']}\n"
        )
        path.write_text(content, encoding="utf-8")

    manifest = {
        **metadata,
        "max_words": MAX_WORDS,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "file": f"chunk_{index:03d}.md",
                "heading_path": chunk["heading_path"],
                "word_count": chunk["word_count"],
                "text": chunk["text"],
            }
            for index, chunk in enumerate(chunks, start=1)
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if not HTML_PATH.exists():
        raise SystemExit(f"Missing input HTML: {HTML_PATH}")

    parser = DocumentParser()
    parser.feed(HTML_PATH.read_text(encoding="utf-8", errors="replace"))

    content_root = find_first(parser.root, "main")
    if content_root is None:
        content_root = find_first(parser.root, "body") or parser.root

    blocks = extract_blocks(content_root)
    blocks, duplicates_removed = deduplicate_blocks(blocks)
    chunks = make_chunks(blocks)

    metadata = {
        "input_file": str(HTML_PATH),
        "extracted_block_count": len(blocks),
        "duplicate_blocks_removed": duplicates_removed,
        "extraction": "standard-library HTML parser; main content preferred",
    }
    write_chunks(chunks, metadata)

    print(f"Extracted {len(blocks)} content blocks")
    print(f"Removed {duplicates_removed} duplicate blocks")
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
