import pandas as pd

from ..llm import LLM
from .stratified_agg import StratifiedLLMEstimator


@pd.api.extensions.register_series_accessor("llm")
class LLMQuery:
    """Pandas accessor for LLM operations"""

    _llm: LLM = None

    def __init__(self, pandas_obj):
        if self._llm is None:
            raise ValueError(
                "Please configure LLM estimator first using llmsql.setup()"
            )

        self._obj = pandas_obj
        self._estimator = None

    def create_index(
        self,
        n_clusters: int = 5,
        threshold: float = 1,
        sample_ratio: float = 0.2,
    ):
        """Configure LLM estimator"""
        self._estimator = StratifiedLLMEstimator(
            llm=self._llm,
            texts=self._obj,
            sample_ratio=sample_ratio,
            threshold=threshold,
            n_clusters=n_clusters,
        )

    def map(self, query: str) -> pd.Series | pd.DataFrame:
        """
        Map each item in series using LLM query

        Args:
            query: Query string for LLM (e.g., "{sentiment}" or "{score: float, sentiment: str}")

        Returns:
            pd.Series | pd.DataFrame: Series for single field query, DataFrame for multiple fields
        """
        results = [self._llm(query=query, context=text) for text in self._obj]

        # if len(results[0]) == 1:
        return pd.Series(results, index=self._obj.index)

        # return pd.DataFrame(results, index=self._obj.index)

    def sum(
        self, query: str, approx: bool = False, adjust: bool = True
    ) -> float | dict[str, float]:
        """
        Calculate sum using either approximation or full query

        Args:
            query: Query string for LLM (e.g., "{score: float}" or "{score: float, confidence: float}")
            approx: If True, use stratified sampling; if False, query all items

        Returns:
            float | dict[str, float]: Sum of numeric fields. Single float for one field,
                                     dict of sums for multiple fields
        """
        if approx:
            estimate = self._estimator.estimate(query, adjust)
            return estimate * len(self._obj)

        return self._query_all(query)

    def _query_all(self, query: str) -> dict[str, float]:
        """Query all items without sampling"""
        return sum(self._llm(query=query, context=text) for text in self._obj)
