"""Run the full 5-stage RAG ingestion pipeline.

Stages:
1) utils/fetch_data.py
2) utils/preprocessing.py
3) utils/chunking.py
4) utils/embedding.py
5) utils/vectorstore.py

Usage:
    python Ingestion.py
    python Ingestion.py --url https://www.geeksforgeeks.org/python/python-programming-language-tutorial/
    python Ingestion.py --topics strings,list,tuples --collection gfg_rag_docs --recreate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args(cfg: dict) -> argparse.Namespace:
    qdrant = cfg.get("qdrant", {})
    ollama = cfg.get("ollama", {})
    data = cfg.get("data", {})

    qdrant_host = qdrant.get("host", "localhost")
    qdrant_port = qdrant.get("port", 6333)
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

    parser = argparse.ArgumentParser(description="Run all 5 ingestion stages end-to-end.")

    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project root containing the utils folder. Default: current directory.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")

    # Shared data paths
    parser.add_argument("--raw-output", default=data.get("raw_output", "data/raw_documents.jsonl"), help="Output path for fetch stage.")
    parser.add_argument("--clean-output", default=data.get("clean_output", "data/clean_documents.jsonl"), help="Output path for preprocessing stage.")
    parser.add_argument("--chunks-output", default=data.get("chunks_output", "data/chunks.jsonl"), help="Output path for chunking stage.")
    parser.add_argument("--embedded-output", default=data.get("embedded_output", "data/chunks_embedded.jsonl"), help="Output path for embedding stage.")

    # Stage 1: fetch
    parser.add_argument(
        "--url",
        default="https://www.geeksforgeeks.org/python/python-programming-language-tutorial/",
        help="Seed GeeksforGeeks URL.",
    )
    parser.add_argument(
        "--topics",
        default="strings,list,tuples,dictionary,sets,arrays,list comprehension",
        help="Comma-separated topic keywords used by fetch stage.",
    )
    parser.add_argument("--max-linked-pages", type=int, default=10, help="Max linked pages to fetch.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout for fetch stage.")

    # Stage 2: preprocessing
    parser.add_argument("--min-words", type=int, default=30, help="Minimum words kept in preprocessing.")

    # Stage 3: chunking
    parser.add_argument("--max-tokens", type=int, default=800, help="Max tokens per chunk.")
    parser.add_argument("--overlap", type=int, default=120, help="Token overlap between chunks.")
    parser.add_argument("--min-tokens", type=int, default=80, help="Minimum tokens per chunk.")

    # Stage 4: embedding
    parser.add_argument("--model", default=ollama.get("embedding_model", "nomic-embed-text"), help="Embedding model name.")
    parser.add_argument("--ollama-url", default=ollama.get("base_url", "http://localhost:11434"), help="Base URL of Ollama server.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pause between embedding calls.")

    # Stage 5: vectorstore
    parser.add_argument("--qdrant-url", default=qdrant_url, help="Qdrant URL.")
    parser.add_argument("--api-key", default=None, help="Optional Qdrant API key.")
    parser.add_argument("--collection", default=qdrant.get("collection_name", "gfg_rag_docs"), help="Qdrant collection name.")
    parser.add_argument("--batch-size", type=int, default=qdrant.get("batch_size", 64), help="Qdrant upsert batch size.")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection before upsert.")

    return parser.parse_args()


def run_stage(name: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n=== Running stage: {name} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Stage '{name}' failed with exit code {result.returncode}.")


def main() -> int:
    cfg = load_config()
    args = parse_args(cfg)
    base_dir = Path(args.base_dir).resolve()
    utils_dir = base_dir / "utils"

    if not utils_dir.exists():
        print(f"utils folder not found at: {utils_dir}", file=sys.stderr)
        return 1

    py = sys.executable

    fetch_cmd = [
        py,
        str(utils_dir / "fetch_data.py"),
        "--url",
        args.url,
        "--topics",
        args.topics,
        "--max-linked-pages",
        str(args.max_linked_pages),
        "--timeout",
        str(args.timeout),
        "--output",
        args.raw_output,
    ]

    preprocess_cmd = [
        py,
        str(utils_dir / "preprocessing.py"),
        "--input",
        args.raw_output,
        "--output",
        args.clean_output,
        "--min-words",
        str(args.min_words),
    ]

    chunk_cmd = [
        py,
        str(utils_dir / "chunking.py"),
        "--input",
        args.clean_output,
        "--output",
        args.chunks_output,
        "--max-tokens",
        str(args.max_tokens),
        "--overlap",
        str(args.overlap),
        "--min-tokens",
        str(args.min_tokens),
    ]

    embed_cmd = [
        py,
        str(utils_dir / "embedding.py"),
        "--input",
        args.chunks_output,
        "--output",
        args.embedded_output,
        "--model",
        args.model,
        "--ollama-url",
        args.ollama_url,
        "--sleep",
        str(args.sleep),
    ]

    vector_cmd = [
        py,
        str(utils_dir / "vectorstore.py"),
        "--input",
        args.embedded_output,
        "--qdrant-url",
        args.qdrant_url,
        "--collection",
        args.collection,
        "--batch-size",
        str(args.batch_size),
    ]

    if args.api_key:
        vector_cmd.extend(["--api-key", args.api_key])
    if args.recreate:
        vector_cmd.append("--recreate")

    try:
        run_stage("fetch", fetch_cmd, base_dir)
        run_stage("preprocess", preprocess_cmd, base_dir)
        run_stage("chunk", chunk_cmd, base_dir)
        run_stage("embed", embed_cmd, base_dir)
        run_stage("vectorstore", vector_cmd, base_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("\nPipeline completed successfully.")
    print(f"Collection: {args.collection}")
    print(f"Embedded file: {args.embedded_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
