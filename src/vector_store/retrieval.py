"""FAISS index loading and vector retrieval utilities for semantic search."""

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_faiss_index(index_path: str, metadata_path: str):
    """Load the faiss index file and metadata file.

    Args:
        index_path (str): The filepath for the index file.
        metadata_path (str): The filepath for the metadata file.

    Returns:
        index: A faiss index file
        metadata: A json file of metadata matching the faiss index file
    """
    index = faiss.read_index(index_path)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def retrieve(query: str, model: SentenceTransformer, index, metadata, k: int = 5):
    """Retrieve top-k chunks from FAISS based on a query.

    Args:
        query (str): User query text.
        model: Embedding model.
        index: FAISS index.
        metadata (list): Metadata entries.
        k (int): Number of chunks to return.

    Returns:
        list: Retrieved chunks with scores.
    """
    # 1. Embed query
    q_emb = model.encode(query)
    q_emb = q_emb / np.linalg.norm(q_emb)  # normalize for cosine similarity
    q_emb = np.array([q_emb], dtype="float32")

    # 2. Search FAISS
    distances, ids = index.search(q_emb, k)

    # 3. Fetch metadata for each result
    results = []
    for score, idx in zip(distances[0], ids[0], strict=True):
        results.append(
            {
                "score": float(score),
                "chunk_id": metadata[idx]["chunk_id"],
                "parent_id": metadata[idx]["parent_id"],
                "title": metadata[idx]["title"],
                "text": metadata[idx]["text"],
                "urls": metadata[idx]["urls"],
                "time_utc": metadata[idx]["time_utc"],
                "upvote": metadata[idx]["upvote"],
                "num_comments": metadata[idx]["num_comments"],
                "source": metadata[idx]["source"],
            }
        )

    return results


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index, metadata = load_faiss_index(
    index_path="data/vector_index/all_vectors.faiss",
    metadata_path="data/vector_index/all_metadata.jsonl",
)
results = retrieve(
    "I got IPL to fix the sun damage on my arms, this is what my experience was like.",
    model=model,
    index=index,
    metadata=metadata,
)
print(results)
