"""Embed data, store in FAISS index, store metadata in .json file."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_explorer.data_prep.schema import VECTOR_FIELDS, validate

from .jsonl import read_jsonl, write_jsonl


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
    index = faiss.IndexFlatIP(dim)
    metadata = []

    # --- Load chunked JSONL
    for line in read_jsonl(in_file):
        text = line.get("text", "").strip()
        if not text:
            continue

        # --- Embed
        emb = embedding_model.encode(text)
        emb = emb / np.linalg.norm(emb)  # normalize for cosine similarity
        emb = np.array([emb], dtype="float32")

        # --- Store metadata; vector_id is the row of this vector in the FAISS index
        record = {
            "vector_id": index.ntotal,
            "parent_id": line.get("parent_id"),
            "title": line.get("title"),
            "chunk_id": line.get("chunk_id"),
            "text": text,
            "source": line.get("source"),
            "links": line.get("links"),
            "doc_type": line.get("doc_type"),
            "source_url": line.get("source_url"),
            "time_utc": line.get("time_utc"),
            "upvote": line.get("upvote"),
            "num_comments": line.get("num_comments"),
        }
        validate(record=record, fields=VECTOR_FIELDS)
        metadata.append(record)

        # --- Add to FAISS
        index.add(emb)

    if not metadata:
        raise ValueError(
            f"No text chunks were embedded from {in_file}: the file is empty "
            "or every record has an empty 'text' field."
        )

    # --- Save FAISS + metadata
    faiss.write_index(index, out_embedding)
    write_jsonl(metadata, out_metadata)
