"""Unit test rag/prompt.py."""

from rag_explorer.rag.prompt import return_prompt
from rag_explorer.vector_store.models import RetrievalResult


def test_prompt_contains_context():
    """Verify that the generated prompt contains the query and context."""
    result = RetrievalResult(
        score=0.5,
        chunk_id="1234",
        parent_id="54321",
        title="title",
        text="This is a text",
        urls=["http://www.wikipedia.com"],
        time_utc=1234.5,
        upvote=0,
        num_comments=0,
        source="source",
    )

    prompt = return_prompt(
        [result],
        "Will paying for skincare make me look younger?",
    )

    assert "This is a text" in prompt
    assert "Will paying for skincare make me look younger?" in prompt
