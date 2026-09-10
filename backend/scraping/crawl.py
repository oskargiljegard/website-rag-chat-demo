"""Crawl Compileit HTML pages and save the raw responses locally.

This crawler intentionally does not download non-HTML resources or URLs with
query strings. Both are logged for later review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import urllib.robotparser
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_START_URL = "https://compileit.com/"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "compileit_crawl"
USER_AGENT = "website-rag-chat-demo/0.1 (authorized local crawl experiment)"
ALLOWED_HOSTS = {"compileit.com", "www.compileit.com"}
HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
DEFAULT_MAX_PAGES = 1000
DEFAULT_DELAY_SECONDS = 0.5
DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_ATTEMPTS = 2

NON_HTML_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bmp",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".svg",
    ".tar",
    ".tgz",
    ".tif",
    ".tiff",
    ".txt",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def host_is_allowed(host: str | None) -> bool:
    return (host or "").lower() in ALLOWED_HOSTS


def normalized_url(url: str, *, keep_query: bool = False) -> str:
    parts = urlsplit(url)
    hostname = (parts.hostname or "").lower()
    if not hostname:
        return url

    port = parts.port
    netloc = hostname
    if port and not ((parts.scheme == "http" and port == 80) or (parts.scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"

    return urlunsplit(
        (
            parts.scheme.lower(),
            netloc,
            parts.path or "/",
            parts.query if keep_query else "",
            "",
        )
    )


@dataclass(frozen=True)
class URLDecision:
    url: str
    reason: str | None = None
    query_parameters: tuple[str, ...] = ()


def classify_url(raw_url: str, source_url: str | None = None) -> URLDecision:
    absolute_url = urljoin(source_url or DEFAULT_START_URL, raw_url)
    parts = urlsplit(absolute_url)

    if parts.scheme.lower() not in {"http", "https"}:
        return URLDecision(absolute_url, "unsupported_scheme")
    if parts.username or parts.password:
        return URLDecision(absolute_url, "credentials_in_url")
    if not host_is_allowed(parts.hostname):
        return URLDecision(absolute_url, "out_of_scope_host")

    without_fragment = normalized_url(absolute_url, keep_query=True)
    query_parameters = tuple(sorted({key for key, _ in parse_qsl(parts.query, keep_blank_values=True)}))
    if parts.query:
        return URLDecision(without_fragment, "query_url", query_parameters)

    path_suffix = Path(parts.path.lower()).suffix
    if path_suffix in NON_HTML_EXTENSIONS:
        return URLDecision(normalized_url(absolute_url), "non_html_extension")

    return URLDecision(normalized_url(absolute_url))


class RedirectRejected(Exception):
    def __init__(self, target_url: str, reason: str) -> None:
        super().__init__(f"Redirect rejected: {target_url} ({reason})")
        self.target_url = target_url
        self.reason = reason


class SameSiteRedirectHandler(HTTPRedirectHandler):
    """Prevent urllib from following redirects outside the crawl scope."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        target_url = urljoin(request.full_url, newurl)
        decision = classify_url(target_url, request.full_url)
        if decision.reason:
            raise RedirectRejected(decision.url, decision.reason)
        return super().redirect_request(request, fp, code, msg, headers, decision.url)


class LinkParser(HTMLParser):
    """Extract ordinary links and canonical links from an HTML response."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.canonical_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() in {"a", "area"} and attributes.get("href"):
            self.hrefs.append(attributes["href"])
        if tag.lower() == "link" and "canonical" in attributes.get("rel", "").lower():
            if attributes.get("href"):
                self.canonical_hrefs.append(attributes["href"])


class CrawlDatabase:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                depth INTEGER NOT NULL,
                discovered_from TEXT,
                discovered_at TEXT NOT NULL,
                fetched_at TEXT,
                final_url TEXT,
                status_code INTEGER,
                content_type TEXT,
                charset TEXT,
                bytes INTEGER,
                sha256 TEXT,
                raw_path TEXT,
                metadata_path TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS links (
                source_url TEXT NOT NULL,
                target_url TEXT NOT NULL,
                raw_href TEXT NOT NULL,
                discovered_at TEXT NOT NULL,
                UNIQUE(source_url, target_url, raw_href)
            );

            CREATE TABLE IF NOT EXISTS skips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reason TEXT NOT NULL,
                url TEXT NOT NULL,
                source_url TEXT,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.execute("UPDATE pages SET status = 'queued' WHERE status = 'fetching'")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def add_page(self, url: str, depth: int, discovered_from: str | None) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO pages
                (url, status, depth, discovered_from, discovered_at)
            VALUES (?, 'queued', ?, ?, ?)
            """,
            (url, depth, discovered_from, utc_now()),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def requeue_retryable_failures(self, max_attempts: int) -> None:
        self.connection.execute(
            """
            UPDATE pages
            SET status = 'queued', error = 'retrying after a previous run'
            WHERE status = 'failed' AND attempts < ?
            """,
            (max_attempts,),
        )
        self.connection.commit()

    def add_link(self, source_url: str, target_url: str, raw_href: str) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO links (source_url, target_url, raw_href, discovered_at)
            VALUES (?, ?, ?, ?)
            """,
            (source_url, target_url, raw_href, utc_now()),
        )
        self.connection.commit()

    def add_skip(self, reason: str, url: str, source_url: str | None, details: dict[str, object]) -> None:
        self.connection.execute(
            """
            INSERT INTO skips (reason, url, source_url, details, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (reason, url, source_url, json.dumps(details, ensure_ascii=False), utc_now()),
        )
        self.connection.commit()

    def next_queued_page(self) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM pages WHERE status = 'queued' ORDER BY depth, discovered_at LIMIT 1"
        ).fetchone()

    def mark_fetching(self, url: str) -> int:
        self.connection.execute(
            "UPDATE pages SET status = 'fetching', attempts = attempts + 1 WHERE url = ?",
            (url,),
        )
        self.connection.commit()
        row = self.connection.execute("SELECT attempts FROM pages WHERE url = ?", (url,)).fetchone()
        return int(row["attempts"])

    def mark_queued(self, url: str, error: str) -> None:
        self.connection.execute(
            "UPDATE pages SET status = 'queued', error = ? WHERE url = ?",
            (error, url),
        )
        self.connection.commit()

    def update_page(self, url: str, **values: object) -> None:
        assignments = ", ".join(f"{key} = ?" for key in values)
        self.connection.execute(
            f"UPDATE pages SET {assignments} WHERE url = ?",
            (*values.values(), url),
        )
        self.connection.commit()

    def count(self, where: str = "1 = 1") -> int:
        row = self.connection.execute(f"SELECT COUNT(*) AS count FROM pages WHERE {where}").fetchone()
        return int(row["count"])

    def skip_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT reason, COUNT(*) AS count FROM skips GROUP BY reason ORDER BY count DESC"
        ).fetchall()
        return {row["reason"]: int(row["count"]) for row in rows}

    def query_parameter_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        rows = self.connection.execute("SELECT details FROM skips WHERE reason = 'query_url'").fetchall()
        for row in rows:
            details = json.loads(row["details"])
            for parameter in details.get("query_parameters", []):
                counts[parameter] = counts.get(parameter, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    charset: str | None
    body: bytes | None
    reason: str | None = None


class Crawler:
    def __init__(
        self,
        start_url: str,
        output_dir: Path,
        max_pages: int,
        delay_seconds: float,
        max_response_bytes: int,
        max_attempts: int,
    ) -> None:
        decision = classify_url(start_url)
        if decision.reason:
            raise ValueError(f"Invalid start URL: {start_url} ({decision.reason})")

        self.start_url = decision.url
        self.output_dir = output_dir
        self.pages_dir = output_dir / "pages"
        self.metadata_dir = output_dir / "metadata"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.db = CrawlDatabase(output_dir / "crawl.db")
        self.skip_log_path = output_dir / "skipped_links.jsonl"
        self.max_pages = max_pages
        self.delay_seconds = delay_seconds
        self.max_response_bytes = max_response_bytes
        self.max_attempts = max_attempts
        self.db.requeue_retryable_failures(max_attempts)
        self.opener = build_opener(SameSiteRedirectHandler())
        self.robots: urllib.robotparser.RobotFileParser | None = None
        self.robots_sitemaps: list[str] = []
        self.last_request_at = 0.0

    def close(self) -> None:
        self.db.close()

    def wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self.last_request_at = time.monotonic()

    def request(self, url: str, accept: str) -> Any:
        self.wait_for_rate_limit()
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
        return self.opener.open(request, timeout=30)

    def load_robots(self) -> None:
        origin = f"{urlsplit(self.start_url).scheme}://compileit.com"
        robots_url = f"{origin}/robots.txt"
        try:
            with self.request(robots_url, "text/plain") as response:
                body = response.read(512 * 1024)
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(body.decode("utf-8", errors="replace").splitlines())
            self.robots = parser
            self.robots_sitemaps = extract_sitemap_urls(body.decode("utf-8", errors="replace"))
            print(f"Loaded robots.txt ({len(self.robots_sitemaps)} sitemap reference(s))")
        except (HTTPError, URLError, OSError) as error:
            print(f"Warning: could not load robots.txt: {error}", file=sys.stderr)
            self.robots = None

    def robots_allows(self, url: str) -> bool:
        return self.robots is None or self.robots.can_fetch(USER_AGENT, url)

    def record_skip(
        self,
        reason: str,
        url: str,
        source_url: str | None,
        **details: object,
    ) -> None:
        payload = {
            "reason": reason,
            "url": url,
            "source_url": source_url,
            "details": details,
            "created_at": utc_now(),
        }
        with self.skip_log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.db.add_skip(reason, url, source_url, details)

    def discover(self, raw_url: str, source_url: str | None, depth: int, raw_href: str = "") -> None:
        decision = classify_url(raw_url, source_url)
        target_url = decision.url
        if source_url:
            self.db.add_link(source_url, target_url, raw_href or raw_url)

        if decision.reason:
            self.record_skip(
                decision.reason,
                target_url,
                source_url,
                raw_href=raw_href or raw_url,
                query_parameters=list(decision.query_parameters),
            )
            return

        self.db.add_page(target_url, depth, source_url)

    def load_sitemap(self, sitemap_url: str, depth: int = 0) -> None:
        if depth > 3:
            return
        decision = classify_url(sitemap_url)
        if decision.reason and decision.reason != "non_html_extension":
            return
        try:
            with self.request(decision.url, "application/xml,text/xml") as response:
                body = response.read(2 * 1024 * 1024)
        except (HTTPError, URLError, OSError, RedirectRejected) as error:
            print(f"Could not load sitemap {sitemap_url}: {error}", file=sys.stderr)
            return

        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as error:
            print(f"Could not parse sitemap {sitemap_url}: {error}", file=sys.stderr)
            return

        root_name = root.tag.rsplit("}", 1)[-1].lower()
        locations = [
            (element.text or "").strip()
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1].lower() == "loc" and (element.text or "").strip()
        ]
        if root_name == "sitemapindex":
            for location in locations:
                nested = classify_url(location, sitemap_url)
                if not nested.reason:
                    self.load_sitemap(nested.url, depth + 1)
        else:
            for location in locations:
                self.discover(location, sitemap_url, 0, raw_href=location)

    def fetch_html(self, url: str) -> FetchResult:
        with self.request(url, "text/html,application/xhtml+xml") as response:
            final_decision = classify_url(response.geturl(), url)
            if final_decision.reason:
                raise RedirectRejected(final_decision.url, final_decision.reason)

            content_type = response.headers.get_content_type().lower()
            charset = response.headers.get_content_charset()
            status_code = int(getattr(response, "status", 200))
            if content_type not in HTML_CONTENT_TYPES:
                return FetchResult(
                    url,
                    final_decision.url,
                    status_code,
                    content_type,
                    charset,
                    None,
                    "non_html_response",
                )

            body = response.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            return FetchResult(
                url,
                final_decision.url,
                status_code,
                content_type,
                charset,
                None,
                "response_too_large",
            )
        return FetchResult(url, final_decision.url, status_code, content_type, charset, body)

    def save_page(self, page: sqlite3.Row, result: FetchResult) -> None:
        page_id = hashlib.sha256(page["url"].encode("utf-8")).hexdigest()[:20]
        html_path = self.pages_dir / f"{page_id}.html"
        metadata_path = self.metadata_dir / f"{page_id}.json"
        assert result.body is not None
        html_path.write_bytes(result.body)

        content_hash = hashlib.sha256(result.body).hexdigest()
        metadata = {
            "requested_url": page["url"],
            "final_url": result.final_url,
            "status_code": result.status_code,
            "content_type": result.content_type,
            "charset": result.charset,
            "bytes": len(result.body),
            "sha256": content_hash,
            "depth": page["depth"],
            "discovered_from": page["discovered_from"],
            "fetched_at": utc_now(),
            "raw_path": str(html_path.relative_to(self.output_dir)),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.db.update_page(
            page["url"],
            status="fetched",
            fetched_at=metadata["fetched_at"],
            final_url=result.final_url,
            status_code=result.status_code,
            content_type=result.content_type,
            charset=result.charset,
            bytes=len(result.body),
            sha256=content_hash,
            raw_path=str(html_path.relative_to(self.output_dir)),
            metadata_path=str(metadata_path.relative_to(self.output_dir)),
            error=None,
        )

        parser = LinkParser()
        parser.feed(result.body.decode(result.charset or "utf-8", errors="replace"))
        for href in parser.hrefs:
            self.discover(href, result.final_url, int(page["depth"]) + 1, raw_href=href)
        for href in parser.canonical_hrefs:
            self.discover(href, result.final_url, int(page["depth"]), raw_href=href)

    def process_page(self, page: sqlite3.Row) -> None:
        url = page["url"]
        if not self.robots_allows(url):
            self.db.update_page(url, status="skipped", error="robots_denied")
            self.record_skip("robots_denied", url, page["discovered_from"])
            return

        attempts = self.db.mark_fetching(url)
        try:
            result = self.fetch_html(url)
        except RedirectRejected as error:
            self.db.update_page(url, status="skipped", error=error.reason, final_url=error.target_url)
            self.record_skip(error.reason, error.target_url, url, redirected_from=url)
            return
        except HTTPError as error:
            message = f"HTTP {error.code}: {error.reason}"
            if error.code in {408, 425, 429, 500, 502, 503, 504} and attempts < self.max_attempts:
                self.db.mark_queued(url, message)
                return
            self.db.update_page(url, status="failed", status_code=error.code, error=message)
            return
        except (URLError, TimeoutError, OSError) as error:
            message = str(error)
            if attempts < self.max_attempts:
                self.db.mark_queued(url, message)
                return
            self.db.update_page(url, status="failed", error=message)
            return

        if result.reason:
            self.db.update_page(
                url,
                status="skipped",
                fetched_at=utc_now(),
                final_url=result.final_url,
                status_code=result.status_code,
                content_type=result.content_type,
                charset=result.charset,
                error=result.reason,
            )
            self.record_skip(
                result.reason,
                result.final_url,
                url,
                status_code=result.status_code,
                content_type=result.content_type,
            )
            return

        self.save_page(page, result)

    def write_report(self, started_at: str) -> None:
        finished_at = utc_now()
        report = {
            "started_at": started_at,
            "finished_at": finished_at,
            "start_url": self.start_url,
            "allowed_hosts": sorted(ALLOWED_HOSTS),
            "robots_loaded": self.robots is not None,
            "counts": {
                "queued": self.db.count("status = 'queued'"),
                "fetched": self.db.count("status = 'fetched'"),
                "skipped": self.db.count("status = 'skipped'"),
                "failed": self.db.count("status = 'failed'"),
                "discovered_total": self.db.count(),
            },
            "skips_by_reason": self.db.skip_counts(),
            "query_parameter_names": self.db.query_parameter_summary(),
            "output": {
                "database": "crawl.db",
                "pages": "pages/",
                "metadata": "metadata/",
                "skipped_links": "skipped_links.jsonl",
            },
        }
        (self.output_dir / "crawl-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def run(self) -> None:
        started_at = utc_now()
        self.load_robots()
        self.discover(self.start_url, None, 0, raw_href=self.start_url)

        for sitemap_url in self.robots_sitemaps or ["https://compileit.com/sitemap.xml"]:
            self.load_sitemap(sitemap_url)

        while self.db.count("status = 'fetched'") < self.max_pages:
            page = self.db.next_queued_page()
            if page is None:
                break
            self.process_page(page)
            fetched = self.db.count("status = 'fetched'")
            if fetched and fetched % 25 == 0:
                queued = self.db.count("status = 'queued'")
                print(f"Fetched {fetched} pages; queued {queued}")

        self.write_report(started_at)
        print(json.dumps(self.read_report_summary(), indent=2, ensure_ascii=False))

    def read_report_summary(self) -> dict[str, object]:
        return {
            "fetched": self.db.count("status = 'fetched'"),
            "queued": self.db.count("status = 'queued'"),
            "skipped": self.db.count("status = 'skipped'"),
            "failed": self.db.count("status = 'failed'"),
            "skips_by_reason": self.db.skip_counts(),
            "query_parameter_names": self.db.query_parameter_summary(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--max-response-mb", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.max_pages < 1 or arguments.delay < 0 or arguments.max_attempts < 1:
        raise SystemExit("max-pages must be positive; delay cannot be negative; max-attempts must be positive")

    crawler = Crawler(
        start_url=arguments.start_url,
        output_dir=arguments.output_dir,
        max_pages=arguments.max_pages,
        delay_seconds=arguments.delay,
        max_response_bytes=int(arguments.max_response_mb * 1024 * 1024),
        max_attempts=arguments.max_attempts,
    )
    try:
        crawler.run()
    finally:
        crawler.close()
    return 0


def extract_sitemap_urls(robots_text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"^\s*sitemap:\s*(\S+)\s*$", robots_text, flags=re.IGNORECASE | re.MULTILINE)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
