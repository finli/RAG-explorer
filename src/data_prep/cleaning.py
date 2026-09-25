"""Clean json reddit files."""

import json
import re
import unicodedata


def clean_field(field: str):
    """Clean reddit field by replacing unicode, newlines, and removing urls.

    Args:
        field (str): The field from Reddit.

    Returns:
        str: Cleaned reddit field.
    """
    # --- Unicode normalization (safe, no escape corruption)
    field = unicodedata.normalize("NFKC", field)

    # Replace reddit unicode
    field = (
        field.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2022", "-")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("&#x200B;", "")
    )

    # Collapse multiple newlines → single newline
    field = re.sub(r"\n\s*\n+", "\n", field)

    # take urls out of field
    field = re.sub(r"https?://\S+", "[URL]", field)

    return field


def clean(input_file: str, output_file: str):
    """Clean reddit document.

    Args:
      input_file (str): The name of the jsonl file to clean.
      output_file (str): The name of the jsonl file to write.
    """
    docs = []

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            jsonl = json.loads(line)

            post = jsonl["post"]

            # if field is very short, skip appending
            if len(str(post).strip()) < 2:
                continue

            # Take urls out of field and add to "urls "
            urls = re.findall(r"https?://\S+", post)

            clean_post = clean_field(post)
            clean_title = clean_field(jsonl["title"])

            doc = {
                "id": f"{clean_title[:20]}_{jsonl['time_utc']}",
                "title": clean_title,
                "post": clean_post,
                "time_utc": float(jsonl["time_utc"]),
                "upvote": float(jsonl["upvote"]),
                "num_comments": int(jsonl["num_comments"]),
                "urls": urls,
                "source": jsonl["source"],
            }
            docs.append(doc)

        with open(output_file, "w", encoding="utf-8") as out:
            for doc in docs:
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
