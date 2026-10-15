"""Utilities for sending info to llm: prompt, response."""

from .generate import get_response
from .prompt import return_prompt

__all__ = ["get_response", "return_prompt"]
