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
        n_clusters: int = 5,
        sample_ratio: float = 0.2,
        cv_threshold: float = 1,
    ):
        self.llm = llm
        self.texts = texts
        self.text_index = TextIndex(texts, n_clusters=n_clusters)
        self.sample_size = int(sample_ratio * len(texts))
        self.cv_threshold = cv_threshold

    def estimate(self, query: str, fields: list[str], adjust: bool = True) -> dict[str, float]:
        """Execute stratified estimation

        Args:
            query: Query template string
            fields: List of field names to estimate

        Returns:
            dict[str, float]: Estimated values for each numeric field
        """
        total_size = len(self.texts)
        estimates = {field: 0.0 for field in fields}

        problematic_clusters = set()

        for cluster_id, stratum_indices in self.text_index:
            stratum_weight = len(stratum_indices) / total_size
            stratum_samples = int(self.sample_size * stratum_weight) + 1
            sampled_indices = np.random.choice(
                stratum_indices, size=stratum_samples, replace=False
            )

            stratum_results = {field: [] for field in fields}
            for idx in sampled_indices:
                result = self.llm(query=query, context=self.texts.iloc[idx])
                for field in fields:
                    stratum_results[field].append(result[field])

            if adjust:
                for field_values in stratum_results.values():
                    if len(field_values) > 1:
                        mean = np.mean(field_values)
                        if mean != 0:
                            cv = np.std(field_values) / abs(mean)
                            print(f"cluster_id: {cluster_id}, cv: {cv}")
                            if cv > self.cv_threshold:
                                problematic_clusters.add(cluster_id)
                                break

            for field in fields:
                stratum_mean = np.mean(stratum_results[field])
                estimates[field] += stratum_mean * stratum_weight

        if problematic_clusters:
            for cluster_id in problematic_clusters:
                self.text_index.adjust(cluster_id)

        return estimates
