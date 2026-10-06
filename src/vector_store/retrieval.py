"""FAISS index loading and vector retrieval utilities for semantic search."""

import json
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_faiss_index(index_file: str, metadata_file: str):
    """Load the faiss index file and metadata file.

    Args:
        index_file (str): The filepath for the index file.
        metadata_file (str): The filepath for the metadata file.

    Returns:
        index: A faiss index file
        metadata: A list of metadata
    """
    index = faiss.read_index(index_file)

    with open(metadata_file, encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]

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
    similarities, ids = index.search(q_emb, k)
    print(similarities, ids)
    # 3. Fetch metadata for each result
    results = []

    for score, idx in zip(similarities[0], ids[0], strict=True):
        print("score", score, "index", idx)
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


def filter_results(results: list[dict[str, Any]], score_threshold: float):
    """Filter top-k chunks so only good scores are passed to LLM.

    Args:
        results (list[dict[str, Any]]): the top_k results that match the query
        score_threshold (float): A number between 0 and 1 that determines which results
            are useful to return.
    """
    return [result for result in results if result["score"] >= score_threshold]
