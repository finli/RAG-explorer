"""Combine metadata JSON files and FAISS index files into unified outputs."""

import json
import os
from pathlib import Path

import faiss


def combine_metadata_files(input_dir: str, output_file: str):
    """Append all JSON metadata files into a single combined JSON file.

    This function loads each `.json` metadata file in the directory, assigns
    sequential `vector_id` values, and writes a unified JSON array to disk.

    Args:
        input_dir (str): Directory containing metadata JSON files.
        output_file (str): Path to the combined output JSON file.

    Returns:
        None
    """
    input_dir = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = []
    global_id = 0

    for entry in os.scandir(input_dir):
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue

        print(f"Reading JSON array: {entry.name}")

        with open(entry.path, encoding="utf-8") as f:
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
    """Combine multiple FAISS index files into a single unified index.

    All FAISS index files in the directory must share the same dimensionality.
    The merged index is written to the specified output file.

    Args:
        input_dir (str): Directory containing `.faiss` index files.
        output_file (str): Path to the combined FAISS index file.

    Returns:
        None
    """
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
