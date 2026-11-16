"""Read and write JSON Lines files for the data prep pipeline."""

import json
from collections.abc import Iterator
from pathlib import Path


def write_jsonl(docs: list[dict], output_file: str):
    """Write docs to a JSON Lines file, one JSON object per line.

    Args:
        docs (list[dict]): The documents to write.
        output_file (str): The path of the jsonl file to write.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        for doc in docs:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")


def read_jsonl(input_file: str) -> Iterator[dict]:
    """Yield one dict per line of a JSON Lines file."""
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
