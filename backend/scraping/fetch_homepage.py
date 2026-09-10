"""Fetch the Compileit homepage and save the raw response locally."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


URL = "https://compileit.com/"
RESULTS_DIR = Path(__file__).parent / "results"
HTML_PATH = RESULTS_DIR / "compileit_homepage.html"
METADATA_PATH = RESULTS_DIR / "compileit_homepage.json"


def fetch_homepage() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    request = Request(
        URL,
        headers={
            "User-Agent": "website-rag-chat-demo/0.1 (homepage fetch experiment)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            body = response.read()
            final_url = response.geturl()
            status_code = response.status
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
    except HTTPError as error:
        print(f"HTTP error {error.code}: {error.reason}", file=sys.stderr)
        return 1
    except URLError as error:
        print(f"Network error: {error.reason}", file=sys.stderr)
        return 1

    HTML_PATH.write_bytes(body)
    metadata = {
        "requested_url": URL,
        "final_url": final_url,
        "status_code": status_code,
        "content_type": content_type,
        "charset": charset,
        "bytes": len(body),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "html_path": str(HTML_PATH.relative_to(Path.cwd())),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Fetched {final_url}")
    print(f"Saved {len(body):,} bytes to {HTML_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(fetch_homepage())
