"""Break the text into chunks with Langchain."""

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def to_chunk(in_file: str, chunk_size: int, chunk_overlap: int):
    """Break data into chunks with overlap, save chunks.

    Args:
        in_file (str): Name of filepath
        chunk_size (int): The number of characters in a chunk.
        chunk_overlap (int): The number of chars that should overlap between chunks.
    """
    path = Path(in_file)
    out_file = Path("data/chunks") / f"{path.stem}_chunks.jsonl"

    with open(in_file) as json_file, open(out_file, "w", encoding="utf-8") as out:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", ""]
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
                    "chunk_id": f"{line.get('id')}_chunk_{i}",
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
