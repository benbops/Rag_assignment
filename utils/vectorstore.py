"""Upsert embedded chunks into Qdrant.

Usage:
    python utils/vectorstore.py
"""

from __future__ import annotations

import argparse
import json
import uuid
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load embedded JSONL chunks into Qdrant.")
    parser.add_argument("--input", default="data/chunks_embedded.jsonl", help="Input embedded JSONL path.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL.")
    parser.add_argument("--api-key", default=None, help="Optional Qdrant API key.")
    parser.add_argument("--collection", default="gfg_rag_docs", help="Qdrant collection name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for upsert.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection.")
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def normalize_point_id(value: object) -> int | str:
    # Qdrant accepts only unsigned integers or UUID strings as point IDs.
    if isinstance(value, int) and value >= 0:
        return value

    text = str(value).strip()
    if text.isdigit():
        return int(text)

    try:
        return str(uuid.UUID(text))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


def build_points(rows: list[dict]) -> list[PointStruct]:
    points: list[PointStruct] = []
    for row in rows:
        vector = row.get("vector")
        chunk_id_raw = row.get("chunk_id")
        if not isinstance(vector, list) or not vector or chunk_id_raw is None:
            continue

        point_id = normalize_point_id(chunk_id_raw)
        payload = {k: v for k, v in row.items() if k != "vector"}
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
    return points


def ensure_collection(client: QdrantClient, collection: str, vector_size: int, recreate: bool) -> None:
    if recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    collections = client.get_collections().collections
    existing = {item.name for item in collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def main() -> int:
    args = parse_args()

    rows = read_jsonl(args.input)
    points = build_points(rows)
    if not points:
        raise RuntimeError("No valid embedded points found in input file.")

    vector_size = len(points[0].vector)
    client = QdrantClient(url=args.qdrant_url, api_key=args.api_key)

    ensure_collection(client, args.collection, vector_size, args.recreate)

    total = 0
    for batch in batched(points, max(1, args.batch_size)):
        client.upsert(collection_name=args.collection, points=batch)
        total += len(batch)
        print(f"Upserted {total}/{len(points)} points")

    print(f"Done. Collection '{args.collection}' now has {len(points)} point(s) upserted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
