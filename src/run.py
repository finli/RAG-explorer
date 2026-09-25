"""Orchestrates data prep: cleaning, chunking, embedding and combining outputs."""

from pathlib import Path

from sentence_transformers import SentenceTransformer

from data_prep import (
    build_vector_index,
    clean,
    remove_duplicates,
    to_chunk,
)

# from vector_store import retrieve, load_faiss_index


def prepare_data(
    model: SentenceTransformer, chunk_size: int = 400, chunk_overlap: int = 80
):
    """Prepare the dataset by cleaning, chunking, embedding, and combining outputs.

    This function checks whether intermediate folders are empty and performs
    the necessary processing steps only when required.
    """
    # Deduplicate documents
    if not Path("data/deduped.json").exists():
        remove_duplicates(input_dir="data/raw/", output_file="data/deduped.jsonl")

    # Clean documents
    if not Path("data/clean.jsonl").exists():
        clean(input_file="data/deduped.jsonl", output_file="data/clean.jsonl")

    # Break documents into chunks
    if not Path("data/chunks.jsonl").exists():
        to_chunk(
            in_file="data/clean.jsonl",
            out_file="data/chunks.jsonl",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Embed, save as vector, and metadata if either file isn't written
    metadata_file = Path("data/vector_metadata.jsonl")
    faiss_file = Path("data/vector.faiss")

    if not metadata_file.exists() or not faiss_file.exists():
        build_vector_index(
            in_file="data/chunks.jsonl",
            out_embedding="data/vector.faiss",
            out_metadata="data/vector_metadata.jsonl",
            embedding_model=model,
        )


# Create the model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Clean, chunk, and embed data
prepare_data(model=model)

"""
# Get answers
index, metadata = load_faiss_index(
    index_path="data/vector_index/all_vectors.faiss",
    metadata_path="data/vector_index/all_metadata.jsonl",
)

results = retrieve(
    "Which product is the best for my acne?",
    model=model,
    index=index,
    metadata=metadata,
)
#print(results)
"""
