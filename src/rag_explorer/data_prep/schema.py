# src/rag_explorer/data_prep/schema.py
"""Field definitions shared across the data prep pipeline."""

# Unified schema. Every cleaned doc has the same keys
DOC_FIELDS = [
    "id",
    "title",
    "post",
    "source",
    "links",
    "doc_type",
    "source_url",
    "time_utc",
    "upvote",
    "num_comments",
]
CHUNK_FIELDS = [
    "parent_id",
    "title",
    "chunk_id",
    "text",
    "source",
    "links",
    "doc_type",
    "source_url",
    "time_utc",
    "upvote",
    "num_comments",
]


def validate(record: dict, fields: list[str]) -> None:
    """Raise ValueError if record's keys don't match fields exactly, order matters."""
    if list(record) != fields:
        missing = set(fields) - set(record)
        extra = set(record) - set(fields)
        raise ValueError(f"Schema mismatch: missing={missing}, extra={extra}")
