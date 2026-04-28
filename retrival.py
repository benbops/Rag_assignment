"""Retrieve relevant chunks from Qdrant and answer questions with Ollama.

Usage:
    python retrival.py --question "What is Python used for?"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
import yaml
from qdrant_client import QdrantClient
from requests import Response
from requests.exceptions import HTTPError, ReadTimeout, RequestException


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args(cfg: dict) -> argparse.Namespace:
    qdrant = cfg.get("qdrant", {})
    ollama = cfg.get("ollama", {})

    qdrant_host = qdrant.get("host", "localhost")
    qdrant_port = qdrant.get("port", 6333)
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

    parser = argparse.ArgumentParser(description="Retrieve chunks from Qdrant and answer with Ollama.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("question", nargs="?", help="User question (positional).")
    parser.add_argument("--question", dest="question_flag", default=None, help="User question.")
    parser.add_argument("--qdrant-url", default=qdrant_url, help="Qdrant URL.")
    parser.add_argument("--collection", default=qdrant.get("collection_name", "gfg_rag_docs"), help="Qdrant collection.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k chunks to retrieve.")
    parser.add_argument("--embedding-model", default=ollama.get("embedding_model", "nomic-embed-text"), help="Ollama embedding model.")
    parser.add_argument("--llm-model", default=ollama.get("llm_model", "llama3.2"), help="Ollama LLM model.")
    parser.add_argument("--ollama-url", default=ollama.get("base_url", "http://localhost:11434"), help="Base Ollama URL.")
    parser.add_argument("--embed-timeout", type=int, default=int(ollama.get("embed_timeout", 120)), help="Timeout (seconds) for embedding call.")
    parser.add_argument("--generate-timeout", type=int, default=int(ollama.get("generate_timeout", 600)), help="Timeout (seconds) for answer generation call.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks before final answer.")
    return parser.parse_args()


def parse_error_detail(resp: Response) -> str:
    try:
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
    except Exception:
        pass
    return resp.text.strip()[:400] if resp.text else "No error body returned"


def call_ollama(endpoint: str, payload: dict, timeout_sec: int, step_name: str) -> dict:
    print(f"Calling Ollama: {endpoint}", flush=True)
    try:
        resp = requests.post(endpoint, json=payload, timeout=max(1, timeout_sec))
    except ReadTimeout as exc:
        raise RuntimeError(
            f"Ollama timeout during {step_name} after {timeout_sec}s at {endpoint}. "
            "Model may still be loading. Try again with a higher timeout or run: ollama pull llama3.2"
        ) from exc
    except RequestException as exc:
        raise RuntimeError(f"Failed to connect to Ollama during {step_name} at {endpoint}: {exc}") from exc

    try:
        resp.raise_for_status()
    except HTTPError as exc:
        detail = parse_error_detail(resp)
        raise RuntimeError(f"Ollama request failed at {endpoint}: {detail}") from exc
    return resp.json()


def get_embedding(text: str, model: str, ollama_url: str, timeout_sec: int) -> list[float]:
    base = ollama_url.rstrip("/")

    embed_endpoint = base + "/api/embed"
    embed_payload = {"model": model, "input": [text]}
    try:
        data = call_ollama(embed_endpoint, embed_payload, timeout_sec=timeout_sec, step_name="embedding")
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            vector = embeddings[0]
            if vector:
                return vector
    except RuntimeError:
        pass

    legacy_endpoint = base + "/api/embeddings"
    legacy_payload = {"model": model, "prompt": text}
    data = call_ollama(legacy_endpoint, legacy_payload, timeout_sec=timeout_sec, step_name="embedding")
    vector = data.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise RuntimeError(f"Invalid embedding response: {data}")
    return vector


def retrieve_chunks(
    qdrant_url: str,
    collection: str,
    query_vector: list[float],
    top_k: int,
) -> list[dict]:
    client = QdrantClient(url=qdrant_url)
    limit = max(1, top_k)

    # qdrant-client has different query methods across versions.
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        hits = result.points if hasattr(result, "points") else result
    else:
        hits = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )

    rows: list[dict] = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        rows.append(
            {
                "score": float(getattr(hit, "score", 0.0)),
                "chunk_id": payload.get("chunk_id"),
                "source_url": payload.get("source_url"),
                "title": payload.get("title"),
                "section_title": payload.get("section_title"),
                "text": payload.get("text", ""),
            }
        )
    return rows


def build_context(retrieved_rows: list[dict]) -> str:
    blocks: list[str] = []
    for i, row in enumerate(retrieved_rows, start=1):
        title = row.get("title") or "Untitled"
        section = row.get("section_title") or "Unknown"
        source = row.get("source_url") or "Unknown"
        text = str(row.get("text") or "").strip()

        blocks.append(
            f"[Chunk {i}]\\n"
            f"Title: {title}\\n"
            f"Section: {section}\\n"
            f"Source: {source}\\n"
            f"Content: {text}"
        )
    return "\\n\\n".join(blocks)


def ask_llm(question: str, context: str, model: str, ollama_url: str, timeout_sec: int) -> str:
    prompt = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the context is not enough, say what is missing.\\n\\n"
        f"Context:\\n{context}\\n\\n"
        f"Question: {question}\\n"
        "Answer:"
    )

    endpoint = ollama_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    data = call_ollama(endpoint, payload, timeout_sec=timeout_sec, step_name="generation")
    answer = data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"Invalid generate response: {data}")
    return answer.strip()


def main() -> int:
    cfg = load_config()
    args = parse_args(cfg)

    raw_question = args.question_flag if args.question_flag is not None else args.question
    question = str(raw_question or "").strip()
    if not question:
        raise RuntimeError("Question cannot be empty. Pass --question or a positional question.")

    start = time.time()
    print("Step 1/4: Embedding question...", flush=True)
    query_vector = get_embedding(question, args.embedding_model, args.ollama_url, args.embed_timeout)
    print("Step 1/4 complete.", flush=True)

    print("Step 2/4: Retrieving chunks from Qdrant...", flush=True)
    retrieved_rows = retrieve_chunks(
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        query_vector=query_vector,
        top_k=args.top_k,
    )
    print(f"Step 2/4 complete. Retrieved {len(retrieved_rows)} chunk(s).", flush=True)

    if not retrieved_rows:
        print("No chunks were retrieved from Qdrant.")
        return 0

    print("Step 3/4: Building context...", flush=True)
    context = build_context(retrieved_rows)

    print("Step 4/4: Generating final answer with Ollama LLM...", flush=True)
    answer = ask_llm(question, context, args.llm_model, args.ollama_url, args.generate_timeout)

    print("\n=== Retrieved Data ===")
    print(json.dumps(retrieved_rows, ensure_ascii=False, indent=2))

    if args.show_context:
        print("\n=== Prompt Context ===")
        print(context)

    print("\n=== Final Answer ===")
    print(answer)
    print(f"\nDone in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
