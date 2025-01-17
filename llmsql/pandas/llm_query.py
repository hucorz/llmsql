from typing import Optional

import pandas as pd

from ..llm import LLM
from .stratified_agg import StratifiedLLMEstimator


@pd.api.extensions.register_series_accessor("llm")
class LLMQuery:
    """Pandas accessor for LLM operations"""

    _llm: Optional[LLM] = None

    def __init__(self, pandas_obj):
        if self._llm is None:
            raise ValueError(
                "Please configure LLM estimator first using llmsql.setup()"
            )

        self._obj = pandas_obj
        self._estimator = None

    def create_index(
        self,
        sample_ratio: float = 0.2,
    ):
        """Configure LLM estimator"""
        self._estimator = StratifiedLLMEstimator(
            llm=self._llm,
            texts=self._obj,
            sample_ratio=sample_ratio,
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

        if len(results[0]) == 1:
            key = next(iter(results[0]))  # get the unique key
            return pd.Series([r[key] for r in results], index=self._obj.index, name=key)

        return pd.DataFrame(results, index=self._obj.index)

    def sum(self, query: str, approx: bool = False) -> float | dict[str, float]:
        """
        Calculate sum using either approximation or full query

        Args:
            query: Query string for LLM (e.g., "{score: float}" or "{score: float, confidence: float}")
            approx: If True, use stratified sampling; if False, query all items

        Returns:
            float | dict[str, float]: Sum of numeric fields. Single float for one field,
                                     dict of sums for multiple fields
        """
        # Parse query to check field types
        fields, _ = self._llm._parse_template(query)

        for field_name, field_type in fields.items():
            if field_type not in ("float", "int", "bool"):
                raise ValueError(
                    f"Field '{field_name}' of type '{field_type}' is not summable. "
                    f"Only numeric types (float, int, bool) can be summed."
                )

        if approx:
            if self._estimator is None:
                raise ValueError(
                    "Please configure LLM estimator first using .configure()"
                )
            estimate = self._estimator.estimate(query, fields)
            result = {field: estimate[field] * len(self._obj) for field in fields}
        else:
            result = self._query_all(query, fields)

        # Return single value if only one field
        if len(fields) == 1:
            key = next(iter(result))
            return result[key]
        return result

    def _query_all(self, query: str, fields: list[str]) -> dict[str, float]:
        """Query all items without sampling"""
        results = [self._llm(query=query, context=text) for text in self._obj]

        sums = {field: sum(r[field] for r in results) for field in fields}

        return sums
