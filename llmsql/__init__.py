from .llm import LLM, LLMEntryPoint
from .pandas import LLMQuery

GlobalEntryPoint = None
VECTORIZE: bool = False
VECTORIZATION_STRIDE = 1


def init(llm: LLM = None):
    """Configure LLM at package level"""
    global GlobalEntryPoint
    GlobalEntryPoint = LLMEntryPoint()
    LLMQuery._llm = llm
