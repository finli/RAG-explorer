"""Read in all files for all reddit forums, then deduplicate."""

import os
from pathlib import Path

import pandas as pd


def remove_duplicates(input_dir: str, output_file: str):
    """Deduplicate based on (title, post) then store all data in one jsonl file.

    Assumes that filename format is date_redditforum.csv.

    Args:
        input_dir: The directory with the raw reddit csv.
        output_file: The name of the jsonl file to write.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_df = pd.DataFrame()

    # Combine all files into one
    for entry in os.scandir(input_dir):
        this_df = pd.read_csv(entry.path)

        # get the forum name from the filename (always same format of date_name.csv)
        source = Path(entry.name).stem.split("_")[-1]
        # add forum name to df
        this_df["source"] = source
        combined_df = pd.concat([combined_df, this_df], ignore_index=True)

    combined_df = combined_df.drop_duplicates(subset=["title", "post"])

    # remove empty posts
    combined_df = combined_df.dropna(subset=["post"])

    combined_df.to_json(output_path, index=False, orient="records", lines=True)


remove_duplicates(input_dir="data/raw/", output_file="data/deduped.jsonl")
