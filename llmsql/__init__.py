from .llm import LLM
from .pandas import LLMQuery


def init(llm: LLM):
    """Configure LLM at package level"""
    LLMQuery._llm = llm
