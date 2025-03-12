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
        threshold: float = 0.8,
    ):
        self.llm = llm
        self.texts = texts
        self.text_index = TextIndex(texts, n_clusters=n_clusters)
        self.sample_ratio = sample_ratio
        self.threshold = threshold

    def estimate(self, query: str, adjust: bool = True) -> dict[str, float]:
        """Execute stratified estimation"""
        total_size = len(self.texts)
        total_result = 0
        problematic_results = {}

        for cluster_id, indices in self.text_index:
            n_samples, weight = self._get_stratum_samples(len(indices), total_size)

            sampled_indices = np.random.choice(indices, size=n_samples, replace=False)
            sampled_results = self._get_sampled_results(query, sampled_indices)

            # 计算均值并加权
            mean = np.mean(sampled_results)
            total_result += mean * weight

            if adjust and self._check(mean):
                # 记录问题样本以及其查询结果
                problematic_results[cluster_id] = (
                    sampled_indices,
                    sampled_results,
                    mean,
                )

        if problematic_results:
            for cluster_id, (indices, results, mean) in problematic_results.items():
                self.text_index.adjust(cluster_id, indices, results, mean)

        return total_result

    def _get_stratum_samples(self, cluster_size: int, total_size: int):
        weight = cluster_size / total_size
        n_samples = max(1, int(np.ceil(self.sample_ratio * cluster_size)))
        return n_samples, weight

    def _get_sampled_results(self, query: str, sampled_indices: list[int]):
        return [
            self.llm(query=query, context=self.texts.iloc[idx])
            for idx in sampled_indices
        ]

    def _check(self, mean: float):
        return 1 - self.threshold < mean < self.threshold
