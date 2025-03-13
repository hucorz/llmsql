import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class TextIndex:
    def __init__(
        self,
        texts: list[str],
        n_clusters: int,
        sim_threshold: float,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(
            embedding_model,
            cache_folder="/fs/fast/share/pingtai_cc/models/huggingface/",
        )
        self.embeddings = self.model.encode(texts)
        self.clusters = self._init_cluster(n_clusters)
        self.sim_threshold = sim_threshold

    def adjust(
        self,
        cluster_id: int,
        sampled_indices: list[int],
        sampled_results: list,
        mean: float,
    ):
        """调整有问题的聚类，将少数结果及其相似样本分离出来"""
        print(f"\n[Cluster {cluster_id}]")
        print(
            f"样本: {len(sampled_indices)}/{len(self.clusters[cluster_id])} (抽样/总数)"
        )
        print(f"均值: {mean:.4f}")

        cluster_indices = self.clusters[cluster_id]
        is_true_minority = mean < 0.5

        # 从原集群中取出少数样本的嵌入向量
        minor_indices = [
            idx
            for idx, res in zip(sampled_indices, sampled_results)
            if res == is_true_minority
        ]
        major_indices = [idx for idx in cluster_indices if idx not in minor_indices]
        print(f"少数派: {len(minor_indices)}/{len(sampled_indices)} (少数派/样本数)")

        # 为每个少数样本找出相似的多数样本
        similar_major_indices = self._find_similar_indices(minor_indices, major_indices)
        new_cluster_indices = minor_indices + similar_major_indices

        # 更新聚类
        self.clusters[cluster_id] = [
            idx for idx in cluster_indices if idx not in new_cluster_indices
        ]
        self.clusters.append(new_cluster_indices)

        print(
            f"分裂: {len(cluster_indices)} -> {len(self.clusters[cluster_id])} + {len(new_cluster_indices)}"
        )

    def _find_similar_indices(self, minor_indices: list[int], major_indices: list[int]):
        """使用Faiss加速相似度搜索"""
        minor_embeddings = self.embeddings[minor_indices]
        major_embeddings = self.embeddings[major_indices]

        # 归一化向量，这样内积就等于余弦相似度
        faiss.normalize_L2(major_embeddings)
        faiss.normalize_L2(minor_embeddings)

        # 创建Faiss索引
        dimension = major_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        index.add(np.ascontiguousarray(major_embeddings))

        D, I = index.search(
            np.ascontiguousarray(minor_embeddings),
            len(major_indices),
        )

        # 收集相似度大于阈值的索引
        similar_majority_indices = []
        for distances, indices in zip(D, I):
            similar_idx = [
                major_indices[i]
                for i, d in zip(indices, distances)
                if d > self.sim_threshold
            ]
            similar_majority_indices.extend(similar_idx)

        return list(set(similar_majority_indices))

    def _init_cluster(self, n_clusters: int = 5) -> list[list[int]]:
        """Create initial clustering"""
        dimension = self.embeddings.shape[1]
        kmeans = faiss.Kmeans(dimension, n_clusters, verbose=True)

        embeddings_array = np.ascontiguousarray(self.embeddings.astype("float32"))
        kmeans.train(embeddings_array)

        _, labels = kmeans.index.search(embeddings_array, 1)

        return [np.where(labels == i)[0].tolist() for i in range(n_clusters)]

    def __iter__(self):
        """Iterate over stratum indices"""
        for indices in self.clusters:
            yield indices
