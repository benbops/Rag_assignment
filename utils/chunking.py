"""Chunk cleaned documents into RAG-ready JSONL chunks.

Usage:
    python utils/chunking.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk cleaned JSONL documents.")
    parser.add_argument("--input", default="data/clean_documents.jsonl", help="Input cleaned JSONL path.")
    parser.add_argument("--output", default="data/chunks.jsonl", help="Output chunk JSONL path.")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max tokens per chunk.")
    parser.add_argument("--overlap", type=int, default=120, help="Token overlap between chunks.")
    parser.add_argument("--min-tokens", type=int, default=80, help="Minimum tokens per chunk.")
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


def approx_token_count(text: str) -> int:
    return len(text.split())


def is_heading(paragraph: str) -> bool:
    line = paragraph.strip()
    if not line:
        return False
    if len(line) > 80:
        return False
    if line.endswith(":"):
        return True
    if re.search(r"[.!?]$", line):
        return False
    words = line.split()
    if len(words) <= 10:
        return True
    return False


def split_sections(text: str) -> list[tuple[str, str]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections: list[tuple[str, list[str]]] = []

    current_title = "Introduction"
    current_parts: list[str] = []

    for para in paragraphs:
        if is_heading(para):
            if current_parts:
                sections.append((current_title, current_parts))
            current_title = para
            current_parts = []
        else:
            current_parts.append(para)

    if current_parts:
        sections.append((current_title, current_parts))

    return [(title, "\n\n".join(parts).strip()) for title, parts in sections if parts]


def sliding_windows(words: list[str], max_tokens: int, overlap: int, min_tokens: int) -> Iterator[str]:
    if not words:
        return

    step = max(1, max_tokens - overlap)
    i = 0
    while i < len(words):
        chunk_words = words[i : i + max_tokens]
        if len(chunk_words) < min_tokens and i != 0:
            break
        yield " ".join(chunk_words).strip()
        i += step


def chunk_section(section_text: str, max_tokens: int, overlap: int, min_tokens: int) -> list[str]:
    words = section_text.split()
    return [chunk for chunk in sliding_windows(words, max_tokens, overlap, min_tokens) if chunk]


def main() -> int:
    args = parse_args()

    docs = read_jsonl(args.input)
    chunks: list[dict] = []

    for doc in docs:
        doc_id = str(doc.get("doc_id", "unknown"))
        source_url = str(doc.get("source_url", ""))
        title = str(doc.get("title", "Untitled"))
        topic = str(doc.get("topic", "general"))
        text = str(doc.get("text", "")).strip()

        chunk_index = 0
        for section_title, section_text in split_sections(text):
            for chunk_text in chunk_section(section_text, args.max_tokens, args.overlap, args.min_tokens):
                chunk_id = f"{doc_id}-{chunk_index:04d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "source_url": source_url,
                        "title": title,
                        "topic": topic,
                        "section_title": section_title,
                        "text": chunk_text,
                        "token_count": approx_token_count(chunk_text),
                    }
                )
                chunk_index += 1

    write_jsonl(args.output, chunks)
    print(f"Saved {len(chunks)} chunk(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
