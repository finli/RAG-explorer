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
    context = ""

    for result in context_list:
        context += f"""
        SOURCE: reddit/{result.source}
        
        TITLE:
        {result.title}
        
        POST:
        {result.text}
        ---
        """
    return f"""You are a skincare discussion assistant.

    Answer the question directly. 

   If the Reddit discussions contain useful information:
        - Provide a concise answer.
        - Mention products that appear in the discussions.
        - Cite Reddit evidence using [Reddit].

    If the Reddit discussions do not contain enough information:
        - Say so briefly.
        - Then provide a short answer based on general skincare knowledge.

    Context:
    {context}

    Question:
    {query}
    
    Answer:"""
