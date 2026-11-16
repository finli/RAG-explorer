"""Combine raw Reddit post csv files, deduplicate, and clean them into jsonl."""

import json
import os
from pathlib import Path

import pandas as pd

from . import common

RENAME = {
    "self_text": "post",
    "selftext": "post",
    "created_utc": "time_utc",
    "ups": "upvote",
    "source url": "source_url",
    "permalink": "source_url",
    "subreddit": "source",
}
CANON = ["title", "post", "time_utc", "upvote", "num_comments", "source_url", "source"]


def remove_duplicates(input_dir: str, output_file: str):
    """Combine raw Reddit csv files and drop duplicate and empty posts.

    Column names are mapped to the canonical set in ``CANON``. Rows are
    deduplicated on (title, post), and rows with no post are removed. The
    source is taken from the filename, which must look like
    ``something_redditforum.csv``. The result is written as jsonl.

    Args:
        input_dir (str): The directory containing the raw Reddit csv files.
        output_file (str): The path of the jsonl file to write. Parent
            directories are created if needed.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_df = pd.DataFrame()

    # Combine all files into one
    for entry in os.scandir(input_dir):
        this_df = pd.read_csv(entry.path).rename(columns=RENAME).reindex(columns=CANON)

        # get the forum name from the filename (always same format of date_name.csv)
        this_df["source"] = Path(entry.name).stem.split("_")[-1]
        combined_df = pd.concat([combined_df, this_df], ignore_index=True)

    combined_df = combined_df.drop_duplicates(subset=["title", "post"])

    # remove empty posts
    combined_df = combined_df.dropna(subset=["post"])

    combined_df.to_json(output_path, index=False, orient="records", lines=True)


boilerplate = [
    "What are you favourite multipurpose products",
    "[META] Hello, new subscribers",
    "Today is Casual Friday",
    "If you're new to SkincareAddiction: welcome",
    "Asian Beauty isn’t ALL about skincare",
    "Post all of your deals, memes, gifs, hauls, sheet mask selfies,",
    "Hello and welcome to the Daily Help Thread",
    "Have a rant about your routine or beauty products",
    "It’s the Weekly Random Chat Post! ",
    "Frustrated and need to rant",
    "seen several comments about the same old same",
    "kick off the month with a thread to keep purchases in check",
    "Welcome to our WITW Shopping Guide series",
    "discuss the anti-haul, where more",
]


def ingest_posts(input_file: str, output_file: str):
    """Read a jsonl file of Reddit posts, clean them, and write a jsonl file.

    Posts that are nearly empty or contain known boilerplate text (daily
    threads, welcome posts, etc.) are skipped.

    Args:
        input_file (str): The path of the jsonl file to read, as produced by
            ``remove_duplicates``.
        output_file (str): The path of the jsonl file to write.
    """
    docs = []

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            post = row["post"]

            # if field is very short, skip appending
            # if post contains daily post, skip appending
            if len(str(post).strip()) < 2 or any(b in post for b in boilerplate):
                continue

            doc = common.make_doc(
                title=row["title"],
                post=post,
                source=row["source"],
                doc_type="post",
                source_url=row["source_url"],
                time_utc=float(row["time_utc"]),
                upvote=row["upvote"],
                num_comments=row["num_comments"],
            )

            docs.append(doc)

        common.write_jsonl(docs, output_file)
