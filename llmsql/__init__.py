from .llm import LLM
from .pandas import LLMQuery

REGISTERED_MODEL = None


def init(llm: LLM):
    """Configure LLM at package level"""
    global REGISTERED_MODEL
    REGISTERED_MODEL = llm
    LLMQuery._llm = llm
