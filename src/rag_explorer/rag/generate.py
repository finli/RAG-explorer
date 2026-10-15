"""Generate responses from a local LLM using Ollama."""

from ollama import chat


def get_response(prompt: str):
    """Generate a response from the configured Ollama model.

    Sends a prompt to a locally hosted LLM and returns the model's
    response object.

    Args:
        prompt: Prompt to send to the model.

    Returns:
        The response returned by the Ollama chat API.
    """
    message = [{"role": "user", "content": prompt}]
    return chat(
        model="qwen3:4b", messages=message, think=False, options={"num_predict": 300}
    )
