"""Create a RAG prompt for an LLM."""

from rag_explorer.vector_store.models import RetrievalResult


def return_prompt(context_list: list[RetrievalResult], query: str) -> str:
    """Generate a RAG prompt from retrieved Reddit discussions.

    Builds a prompt containing retrieved context and the user's query.
    The prompt instructs the LLM to prioritize Reddit evidence when
    available and to clearly distinguish between Reddit-derived information
    and general knowledge.

    Args:
        context_list: Retrieved Reddit discussion chunks.
        query: User question.

    Returns:
        A formatted prompt ready to be sent to the LLM.
    """
    # Change list to str
    context = "\n\n".join(result.text for result in context_list)
    return f"""You are a skincare discussion assistant.

    Use retrieved Reddit discussions as your primary source whenever they are available
    and relevant.
    If Reddit context is available:
    - Clearly label information derived from Reddit as [Reddit].

    If the context is insufficient, you may use your general knowledge, but clearly 
    indicate which parts come from the retrieved Reddit discussions and which parts 
    are general knowledge.

    If the context is empty, ignore the missing context and answer the question using 
    general knowledge.

    Context:
    {context}

    Question:
    {query}"""
