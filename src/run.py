import json
import os
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from to_doc import csv_to_jsonl
from to_vector import build_vector_index, to_chunk


def combine_metadata_files(input_dir: str, output_file: str):
    input_dir = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = []
    global_id = 0

    for entry in os.scandir(input_dir):
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue

        print(f"Reading JSON array: {entry.name}")

        with open(entry.path, "r", encoding="utf-8") as f:
            data = json.load(f)  # load entire JSON array

            for obj in data:
                obj["vector_id"] = global_id
                global_id += 1
                combined.append(obj)

    # Write combined JSON array
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(combined, out, ensure_ascii=False, indent=2)

    print(f"Combined metadata written to {output_path}")
    print(f"Total metadata vectors: {global_id}")


def combine_faiss_indexes(input_dir: str, output_file: str):
    input_dir = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_index = None

    for entry in os.scandir(input_dir):
        if not entry.is_file() or not entry.name.endswith(".faiss"):
            continue

        idx = faiss.read_index(entry.path)

        if main_index is None:
            main_index = idx
        else:
            main_index.merge_from(idx)

    faiss.write_index(main_index, str(output_path))
    print("Total faiss vectors:", main_index.ntotal)


def prepare_data(
    model: SentenceTransformer, chunk_size: int = 400, chunk_overlap: int = 80
):
    """
    Check if work has been done (folder is empty).
    If work undone, prepare the data.
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
