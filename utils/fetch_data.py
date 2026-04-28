"""Fetch GeeksforGeeks tutorial pages and save raw documents as JSONL.

Usage:
    python utils/fetch_data.py
    python utils/fetch_data.py --topics strings,list,tuples --max-linked-pages 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests import Response
from requests.exceptions import SSLError

DEFAULT_URL = "https://www.geeksforgeeks.org/python/python-programming-language-tutorial/"
DEFAULT_TOPICS = "strings,list,tuples,dictionary,sets,arrays,list comprehension"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
CONTENT_SELECTORS = [
    "article",
    "div.article--viewer_content",
    "div.text",
    "div.entry-content",
    "div.content",
    "main",
]
BLOCKED_TAGS = {"script", "style", "noscript", "iframe", "form", "button", "svg", "aside", "footer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch GeeksforGeeks pages into raw JSONL documents.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Seed GeeksforGeeks URL.")
    parser.add_argument("--topics", default=DEFAULT_TOPICS, help="Comma-separated keywords to follow linked pages.")
    parser.add_argument("--max-linked-pages", type=int, default=10, help="Maximum linked pages to fetch.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")
    parser.add_argument("--output", default="data/raw_documents.jsonl", help="Output JSONL path.")
    return parser.parse_args()


def parse_topics(raw_topics: str) -> list[str]:
    if raw_topics.strip().lower() in {"", "none", "no", "off"}:
        return []
    return [item.strip().lower() for item in raw_topics.split(",") if item.strip()]


def clean_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def validate_gfg_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.netloc or "geeksforgeeks.org" not in parsed.netloc:
        raise ValueError("Only geeksforgeeks.org URLs are supported.")
    return url


def fetch_url(url: str, timeout: int) -> Response:
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
    except SSLError:
        if os.getenv("DISABLE_INSECURE_FALLBACK", "0").strip() == "1":
            raise
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.raise_for_status()
        return response


def choose_content_root(soup: BeautifulSoup):
    for selector in CONTENT_SELECTORS:
        root = soup.select_one(selector)
        if root and clean_text(root.get_text(" ", strip=True)):
            return root
    return soup.body or soup


def remove_noise(root) -> None:
    for tag in root.find_all(BLOCKED_TAGS):
        tag.decompose()

    noisy_selectors = [
        ".article_bottom_text",
        ".social-share",
        ".read-more-container",
        ".also-read",
        ".recommended",
        ".comments-area",
        ".advertisement",
        "[class*='advert']",
        "[id*='advert']",
    ]
    for selector in noisy_selectors:
        for node in root.select(selector):
            node.decompose()


def extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    root = choose_content_root(soup)
    remove_noise(root)

    title_node = soup.find("h1") or soup.find("title")
    title = clean_text(title_node.get_text(" ", strip=True)) if title_node else "Untitled"

    parts: list[str] = []
    for node in root.find_all(["h1", "h2", "h3", "p", "li", "pre"]):
        text = clean_text(node.get_text(" ", strip=True))
        if text:
            parts.append(text)

    text = "\n\n".join(parts).strip()
    return title, text


def extract_topic_links(base_url: str, html: str, topic_keywords: list[str]) -> list[str]:
    if not topic_keywords:
        return []

    soup = BeautifulSoup(html, "html.parser")
    root = choose_content_root(soup)

    links: list[str] = []
    seen: set[str] = set()

    for anchor in root.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href:
            continue

        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if "geeksforgeeks.org" not in parsed.netloc:
            continue

        label = clean_text(anchor.get_text(" ", strip=True)).lower()
        slug_text = parsed.path.replace("-", " ").replace("/", " ").lower()
        searchable = f"{label} {slug_text}"

        if any(keyword in searchable for keyword in topic_keywords):
            normalized = parsed._replace(query="", fragment="").geturl()
            if normalized != base_url and normalized not in seen:
                seen.add(normalized)
                links.append(normalized)

    return links


def make_doc_id(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def make_topic(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "general"
    return path.split("/")[-1].replace("-", " ")


def write_jsonl(path: str, rows: list[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()

    try:
        seed_url = validate_gfg_url(args.url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    topics = parse_topics(args.topics)
    rows: list[dict] = []

    try:
        response = fetch_url(seed_url, args.timeout)
        html = response.text
        title, text = extract_main_text(html)
        rows.append(
            {
                "doc_id": make_doc_id(seed_url),
                "source_url": seed_url,
                "title": title,
                "topic": make_topic(seed_url),
                "text": text,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        linked = extract_topic_links(seed_url, html, topics)[: max(0, args.max_linked_pages)]
        for url in linked:
            try:
                linked_html = fetch_url(url, args.timeout).text
                linked_title, linked_text = extract_main_text(linked_html)
                rows.append(
                    {
                        "doc_id": make_doc_id(url),
                        "source_url": url,
                        "title": linked_title,
                        "topic": make_topic(url),
                        "text": linked_text,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except Exception as exc:
                print(f"Warning: failed linked page {url}: {exc}", file=sys.stderr)

    except Exception as exc:
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1

    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} raw document(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
