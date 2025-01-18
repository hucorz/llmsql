from .llm import LLM
from .pandas import LLMQuery, BatchLLMQuery


def init(llm: LLM):
    """Configure LLM at package level"""
    LLMQuery._llm = llm
    BatchLLMQuery._llm = llm
