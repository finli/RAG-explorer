"""Orchestrates data prep: cleaning, chunking, embedding and combining outputs."""

import os

from sentence_transformers import SentenceTransformer

from data_prep import (
    build_vector_index,
    combine_faiss_indexes,
    combine_metadata_files,
    csv_to_jsonl,
    to_chunk,
)


def prepare_data(
    model: SentenceTransformer, chunk_size: int = 400, chunk_overlap: int = 80
):
    """Prepare the dataset by cleaning, chunking, embedding, and combining outputs.

    This function checks whether intermediate folders are empty and performs
    the necessary processing steps only when required.
    """
    # Clean docs, change to jsonl format
    if len(os.listdir("data/clean")) == 0:  # data/clean is empty
        print("Cleaning docs")
        obj = os.scandir("data/raw")
        for entry in obj:
            if entry.is_file():
                csv_to_jsonl(entry)

    # break docs into chunks
    if len(os.listdir("data/chunks")) == 0:
        print("Break in to chunks.")
        obj = os.scandir("data/clean")
        for entry in obj:
            if entry.is_file():
                to_chunk(entry, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # embed, save as vector
    if len(os.listdir("data/vector_files")) == 0:
        print("Embed")
        obj = os.scandir("data/chunks")
        for entry in obj:
            if entry.is_file():
                build_vector_index(entry, model)

    # concatinate vector files
    if len(os.listdir("data/vector_index")) == 0:
        print("Combine metadata")
        combine_metadata_files(
            input_dir="data/vector_files",
            output_file="data/vector_index/all_metadata.jsonl",
        )

        print("Combine FAISS indexes")
        combine_faiss_indexes(
            input_dir="data/vector_files",
            output_file="data/vector_index/all_vectors.faiss",
        )


# Create the model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# if data vector_index folder doeosn't exist
prepare_data(model=model)
