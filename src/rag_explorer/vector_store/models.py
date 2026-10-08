"""Data models for vector store retrieval results."""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """A single retrieved text chunk and its associated metadata."""

    score: float
    chunk_id: str
    parent_id: str
    title: str
    text: str
    urls: list[str]
    time_utc: float
    upvote: float
    num_comments: int
    source: str
