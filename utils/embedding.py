"""Create embeddings for chunks using nomic-embed-text via Ollama.

Usage:
    python utils/embedding.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import requests
from requests import Response
from requests.exceptions import HTTPError

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunks with Ollama nomic-embed-text.")
    parser.add_argument("--input", default="data/chunks.jsonl", help="Input chunk JSONL path.")
    parser.add_argument("--output", default="data/chunks_embedded.jsonl", help="Output embeddings JSONL path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Base Ollama URL.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between embedding calls.")
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


def parse_error_detail(resp: Response) -> str:
    try:
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
    except Exception:
        pass
    return resp.text.strip()[:400] if resp.text else "No error body returned"


def call_embed_api(endpoint: str, payload: dict) -> dict:
    resp = requests.post(endpoint, json=payload, timeout=180)
    try:
        resp.raise_for_status()
    except HTTPError as exc:
        detail = parse_error_detail(resp)
        raise RuntimeError(f"Ollama request failed at {endpoint}: {detail}") from exc
    return resp.json()


def get_embedding(text: str, model: str, ollama_url: str) -> list[float]:
    base = ollama_url.rstrip("/")

    # Newer Ollama releases support /api/embed with input list.
    embed_endpoint = base + "/api/embed"
    embed_payload = {"model": model, "input": [text]}
    try:
        data = call_embed_api(embed_endpoint, embed_payload)
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            vector = embeddings[0]
            if vector:
                return vector
    except RuntimeError:
        # Fall back to legacy endpoint below.
        pass

    legacy_endpoint = base + "/api/embeddings"
    legacy_payload = {"model": model, "prompt": text}
    data = call_embed_api(legacy_endpoint, legacy_payload)
    vector = data.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise RuntimeError(f"Invalid embedding response: {data}")
    return vector


def main() -> int:
    args = parse_args()

    chunks = read_jsonl(args.input)
    rows: list[dict] = []

    for idx, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        vector = get_embedding(text, args.model, args.ollama_url)
        row = dict(chunk)
        row["embedding_model"] = args.model
        row["vector"] = vector
        rows.append(row)

        if args.sleep > 0:
            time.sleep(args.sleep)

        if idx % 25 == 0:
            print(f"Embedded {idx}/{len(chunks)} chunks")

    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} embedded chunk(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
