"""Clean and normalize fetched documents for chunking.

Usage:
    python utils/preprocessing.py
"""

from __future__ import annotations

import argparse
import json
import os
import re

NOISE_PATTERNS = [
    r"^quiz[:\s]",
    r"^coding problems$",
    r"^complete tutorial",
    r"^share this",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw JSONL documents.")
    parser.add_argument("--input", default="data/raw_documents.jsonl", help="Input raw JSONL path.")
    parser.add_argument("--output", default="data/clean_documents.jsonl", help="Output cleaned JSONL path.")
    parser.add_argument("--min-words", type=int, default=30, help="Drop docs shorter than this word count.")
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    test = line.strip().lower()
    if not test:
        return False
    return any(re.search(pattern, test) for pattern in NOISE_PATTERNS)


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    lines = [ln.strip() for ln in text.split("\n")]

    filtered: list[str] = []
    seen_recent: set[str] = set()
    for line in lines:
        if not line:
            filtered.append("")
            continue
        if is_noise_line(line):
            continue

        dedupe_key = line.lower()
        if dedupe_key in seen_recent and len(line.split()) <= 4:
            continue
        seen_recent.add(dedupe_key)
        if len(seen_recent) > 1000:
            seen_recent.clear()

        filtered.append(line)

    cleaned = "\n".join(filtered)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def main() -> int:
    args = parse_args()

    rows = read_jsonl(args.input)
    out_rows: list[dict] = []

    for row in rows:
        text = clean_text(str(row.get("text", "")))
        word_count = len(text.split())
        if word_count < args.min_words:
            continue

        row["text"] = text
        row["word_count"] = word_count
        row["char_count"] = len(text)
        out_rows.append(row)

    write_jsonl(args.output, out_rows)
    print(f"Saved {len(out_rows)} cleaned document(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
