"""Embed data, store in FAISS index, store metadata in .json file."""

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_vector_index(
    in_file: str,
    out_metadata: str,
    out_embedding: str,
    embedding_model: SentenceTransformer,
):
    """Build a FAISS index and metadata file from chunked JSONL input.

    This function loads each text chunk, generates embeddings, stores them in a
    FAISS index, and writes a corresponding metadata JSON file containing vector
    IDs and chunk attributes.

    Args:
        in_file (str): Path to the chunked JSONL file.
        out_metadata (str): Filename of the JSONL medata output file
        out_embedding (str): Filename of the embedding output file.
        embedding_model (SentenceTransformer): Model used to generate embeddings.
    """
    # --- Create FAISS index (L2 or cosine)
    dim = embedding_model.get_embedding_dimension()

    index = faiss.IndexFlatL2(dim)

    metadata = []

    # --- Load chunked JSONL
    with open(in_file) as f:
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
    faiss.write_index(index, out_embedding)
    with open(out_metadata, "w", encoding="utf-8") as out:
        for doc in metadata:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
