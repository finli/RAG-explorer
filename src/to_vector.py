"""
Chunk the text, embed, and then store in a vector database.
"""

import os
from pathlib import Path

import faiss
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer


def to_chunk(file: str):
    """
    Break data into chunks with overlap, save chunks.
    """

    # Output file name
    path = Path(file)
    out_file = Path("data/chunks") / f"{path.stem}_chunks.jsonl"

    with open(file, "r") as json_file, open(out_file, "w", encoding="utf-8") as out:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=80, separators=["\n", ""]
        )

        for json_str in json_file:
            line = json.loads(json_str)
            post = line.get("post", "")

            # Skip empty posts
            if not post.strip():
                continue

            chunks = splitter.split_text(post)
            # Save each chunk with metadata
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "chunk_id": f"{line.get("id")}_chunk_{i}",
                    "parent_id": line.get("id"),
                    "title": line.get("title", ""),
                    "post_chunk": chunk,
                    "time_utc": line.get("time_utc"),
                    "upvote": line.get("upvote"),
                    "num_comments": line.get("num_comments"),
                    "source": line.get("source"),
                    "urls": line.get("urls", []),
                }

                out.write(json.dumps(chunk_doc, ensure_ascii=False) + "\n")


def build_vector_index(chunked_jsonl: str, embedding_model: SentenceTransformer):
    """
    Build a FAISS index from chunked JSONL.
    Steps:
        1. Load chunked JSONL
        2. Embed each chunk
        3. Add vector + metadata to FAISS
        4. Save FAISS index + metadata JSON
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
    with open(chunked_jsonl, "r") as f:
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


obj = os.scandir("data/clean")
for entry in obj:
    if entry.is_file():
        to_chunk(entry)

# Create the model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

obj = os.scandir("data/chunks")
for entry in obj:
    if entry.is_file():
        build_vector_index(entry, model)
