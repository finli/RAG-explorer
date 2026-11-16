"""Read guide files (URL on line 1, body after) into the cleaned schema."""

import re
from pathlib import Path

from . import common


def ingest_guides(input_dir: str, output_file: str):
    """Read raw Reddit guide files, clean them, and write a jsonl file.

    Each ``*.txt`` file has the source URL on its first line and the body after.
    Files with an empty body are skipped. The title is the filename.
    The source is the subreddit from the URL.

    Args:
        input_dir (str): The directory containing the guide ``.txt`` files.
        output_file (str): The path of the jsonl file to write.
    """
    docs = []
    for f in sorted(Path(input_dir).glob("*.txt")):
        print(Path(f).stem)
        url, _, body = f.read_text(encoding="utf-8").partition("\n")

        url, body = url.strip(), body.strip()

        if not body:
            continue

        sub = re.search(r"reddit\.com/r/([^/]+)", url)
        print("sub", sub.group(1) if sub else "unknown")

        docs.append(
            common.make_doc(
                title=Path(f).stem,
                post=body,
                source=sub.group(1) if sub else "unknown",
                doc_type="guide",
                source_url=url,
            ),
        )

    common.write_jsonl(docs, output_file)
