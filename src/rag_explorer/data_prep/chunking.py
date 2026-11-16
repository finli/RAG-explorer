"""Break the text into chunks with Langchain."""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ingest import common
from .schema import CHUNK_FIELDS, validate


def chunk_doc(line: dict, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Break data into chunks with overlap, save chunks.

    Args:
        line: A dictionary with a reddit post and metadata.
        chunk_size (int): The number of characters in a chunk.
        chunk_overlap (int): The number of characters by which chunks should overlap.
    """
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", ""]
    )
    chunks = splitter.split_text(line["post"])

    # Save each chunk with metadata
    for i, chunk in enumerate(chunks):
        chunk_doc = {
            "parent_id": line.get("id"),
            "title": line.get("title", ""),
            "chunk_id": f"{line.get('id')}_chunk_{i}",
            "text": chunk,
            "source": line.get("source"),
            "links": line.get("links", []),
            "doc_type": line.get("doc_type"),
            "source_url": line.get("source_url"),
            "time_utc": line.get("time_utc"),
            "upvote": line.get("upvote"),
            "num_comments": line.get("num_comments"),
        }
        validate(record=chunk_doc, fields=CHUNK_FIELDS)
        docs.append(chunk_doc)
    return docs


def chunk_all(
    input_files: list[str], output_file: str, chunk_size: int, chunk_overlap: str
):
    """Chunk every document in the cleaned jsonl files and write one chunk file.

    Each line of each input file is a cleaned doc (see ``DOC_FIELDS``). Docs are
    split with ``chunk_doc`` and all chunks are written to a single jsonl file.

    Args:
        input_files (list[str]): Paths of the cleaned jsonl files to read.
        output_file (str): Path of the jsonl file to write.
        chunk_size (int): The maximum number of characters in a chunk.
        chunk_overlap (int): The number of characters shared by adjacent chunks.
    """
    chunks = []
    for path in input_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                chunks.extend(
                    chunk_doc(
                        line=json.loads(line),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
    common.write_jsonl(chunks, output_file)
