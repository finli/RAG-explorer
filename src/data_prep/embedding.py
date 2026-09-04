"""Embed data, store in FAISS index, store metadata in .json file."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_vector_index(chunked_jsonl: str, embedding_model: SentenceTransformer):
    """Build a FAISS index and metadata file from chunked JSONL input.

    This function loads each text chunk, generates embeddings, stores them in a
    FAISS index, and writes a corresponding metadata JSON file containing vector
    IDs and chunk attributes.

    Args:
        chunked_jsonl (str): Path to the chunked JSONL file.
        embedding_model (SentenceTransformer): Model used to generate embeddings.

    Returns:
        None
    """
    # --- Prepare output paths
    path = Path(chunked_jsonl)
    out_dir = Path("data/vector_index")
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / f"{path.stem}.faiss"
    metadata_path = out_dir / f"{path.stem}_metadata.json"

    # --- Create FAISS index (L2 or cosine)
    dim = embedding_model.get_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    metadata = []

    # --- Load chunked JSONL
    with open(chunked_jsonl) as f:
        for vector_id, json_str in enumerate(f):
            line = json.loads(json_str)

            text = line.get("post_chunk", "").strip()
            if not text:
                continue

            # --- Embed
            emb = embedding_model.encode(text)
            emb = emb / np.linalg.norm(emb)  # normalize for cosine similarity
            emb = np.array([emb], dtype="float32")

            # --- Add to FAISS
            index.add(emb)

            # --- Store metadata
            metadata.append(
                {
                    "vector_id": vector_id,
                    "chunk_id": line.get("chunk_id"),
                    "parent_id": line.get("parent_id"),
                    "title": line.get("title"),
                    "time_utc": line.get("time_utc"),
                    "upvote": line.get("upvote"),
                    "num_comments": line.get("num_comments"),
                    "source": line.get("source"),
                    "urls": line.get("urls"),
                    "text": text,
                }
            )

    # --- Save FAISS + metadata
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as out:
        json.dump(metadata, out, ensure_ascii=False, indent=2)
