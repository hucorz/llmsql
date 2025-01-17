import numpy as np
import pandas as pd

from ..index import TextIndex
from ..llm import LLM


class StratifiedLLMEstimator:
    """Stratified LLM Estimator for text analysis"""

    def __init__(
        self,
        llm: LLM,
        texts: pd.Series,
        sample_ratio: float = 0.2,
    ):
        self.llm = llm
        self.texts = texts
        self.sample_size = int(sample_ratio * len(texts))
        self.text_index = TextIndex(texts)

    def estimate(self, query: str, fields: list[str]) -> dict[str, float]:
        """Execute stratified estimation

        Args:
            query: Query template string
            fields: List of field names to estimate

        Returns:
            dict[str, float]: Estimated values for each numeric field
        """
        total_size = len(self.texts)
        estimates = {field: 0.0 for field in fields}

        for _, stratum_indices in self.text_index:
            stratum_weight = len(stratum_indices) / total_size
            stratum_samples = int(self.sample_size * stratum_weight)
            sampled_indices = np.random.choice(
                stratum_indices, size=stratum_samples, replace=False
            )

            stratum_results = {field: [] for field in fields}
            for idx in sampled_indices:
                result = self.llm(query=query, context=self.texts.iloc[idx])
                for field in fields:
                    stratum_results[field].append(result[field])

            # Calculate stratum statistics for each field
            for field in fields:
                stratum_mean = np.mean(stratum_results[field])
                estimates[field] += stratum_mean * stratum_weight

        return estimates
