"""Orchestrates data prep: cleaning, chunking, embedding and combining outputs."""

from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag_explorer.data_prep import (
    build_vector_index,
    chunk_all,
)
from rag_explorer.data_prep.ingest import ingest_guides, ingest_posts, remove_duplicates
from rag_explorer.rag import get_response, return_prompt
from rag_explorer.vector_store import filter_results, load_faiss_index, retrieve


def prepare_data(
    model: SentenceTransformer, chunk_size: int = 400, chunk_overlap: int = 80
):
    """Prepare the dataset by cleaning, chunking, embedding, and combining outputs.

    This function checks whether intermediate folders are empty and performs
    the necessary processing steps only when required.
    """
    # Ingest posts
    if not Path("data/posts_deduped.json").exists():
        remove_duplicates(
            input_dir="data/raw/posts", output_file="data/posts_deduped.jsonl"
        )
    if not Path("data/posts_clean.jsonl").exists():
        ingest_posts(
            input_file="data/posts_deduped.jsonl", output_file="data/posts_clean.jsonl"
        )

    # ingest guides
    if not Path("data/guides_clean.jsonl").exists():
        ingest_guides(input_dir="data/raw/wiki", output_file="data/guides_clean.jsonl")

    # Break documents into chunks
    if not Path("data/chunks.jsonl").exists():
        chunk_all(
            input_files=["data/posts_clean.jsonl", "data/guides_clean.jsonl"],
            output_file="data/chunks.jsonl",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Embed, save vector and metadata if either file isn't written
    metadata_file = Path("data/vector_metadata.jsonl")
    faiss_file = Path("data/vector.faiss")

    if not metadata_file.exists() or not faiss_file.exists():
        build_vector_index(
            in_file="data/chunks.jsonl",
            out_embedding=str(faiss_file),
            out_metadata=str(metadata_file),
            embedding_model=model,
        )


# Create the model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Clean, chunk, and embed data
prepare_data(model=model)


# Get answers
index, metadata = load_faiss_index(
    index_file="data/vector.faiss",
    metadata_file="data/vector_metadata.jsonl",
)

question = "acne treatment products"
results = retrieve(
    query=question,
    model=model,
    index=index,
    metadata=metadata,
)
filtered_results = filter_results(results, score_threshold=0.4)

print("filtered_results", filtered_results)

prompt = return_prompt(filtered_results, query=question)

print(get_response(prompt))
