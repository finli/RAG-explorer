import csv, json
import os
from pathlib import Path
import re
import unicodedata


def wash_post(post: str):

    # --- Unicode normalization (safe, no escape corruption)
    post = unicodedata.normalize("NFKC", post)

    # Replace reddit unicode
    post = (
        post.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2022", "-")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("&#x200B;", "")
    )

    # Collapse multiple newlines → single newline
    post = re.sub(r"\n\s*\n+", "\n", post)

    # Strip markdown (or convert to plain text)

    # take urls out of post
    post = re.sub(r"https?://\S+", "[URL]", post)

    return post


def csv_to_jsonl(csv_file: str):
    """
    Convert csv file to jsonl for LangChain.
    """

    # get filename without extension or folder
    filename = Path(csv_file)
    out_name = filename.stem  # filename without extension
    _, source = str(out_name).split("_", 1)

    # Create documents for LangChain
    docs = []

    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:

            post = row["post"]

            # Remove repeated boilerplate
            boilerplate = [
                "Today is Casual Friday",
                "If you're new to SkincareAddiction: welcome",
                "Asian Beauty isn’t ALL about skincare",
                "Post all of your deals, memes, gifs, hauls, sheet mask selfies, and other fluff.", 
                "Hello and welcome to the Daily Help Thread", 
                "Have a rant about your routine or beauty products", 
                "It’s the Weekly Random Chat Post! ",
                "Frustrated and need to rant",
                "seen several comments about the same old same", 
                "kick off the month with a thread to keep purchases in check", 
                "Welcome to our WITW Shopping Guide series",
                "discuss the anti-haul, where more",
            ]

            # if post contains daily post, skip appending this doc
            if any(b in post for b in boilerplate) or len(post) < 2:
                continue

            # Take urls out of post and add to "urls "
            urls = re.findall(r"https?://\S+", post)

            clean_post = wash_post(post)
            clean_title = wash_post(row["title"])

            doc = {
                "id": f"{clean_title[:20]}_{row['time_utc']}",
                "title": clean_title,
                "post": clean_post,
                "time_utc": float(row["time_utc"]),
                "upvote": float(row["upvote"]),
                "num_comments": int(row["num_comments"]),
                "urls": urls,
                "source": "reddit_" + source,
            }
            docs.append(doc)

    out_path = Path("data/clean") / f"{out_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for d in docs:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")

obj = os.scandir("data/raw")
for entry in obj:
    if entry.is_file():
        csv_to_jsonl(entry)

    
