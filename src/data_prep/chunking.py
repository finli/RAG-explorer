"""Break the text into chunks with Langchain."""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


def to_chunk(in_file: str, out_file: str, chunk_size: int, chunk_overlap: int):
    """Break data into chunks with overlap, save chunks.

    Args:
        in_file (str): Name of clean jsonl file.
        out_file (str): Name of the jsonl file to write.
        chunk_size (int): The number of characters in a chunk.

        chunk_overlap (int): The number of chars that should overlap between chunks.
    """
    with open(in_file) as json_file:
        docs = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", ""]
        )

        for json_str in json_file:
            line = json.loads(json_str)
            post = line.get("post", "")

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
                docs.append(chunk_doc)

    with open(out_file, "w", encoding="utf-8") as out:
        for doc in docs:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
