import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class TextIndex:
    def __init__(
        self, texts, n_clusters: int = 5, embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(
            embedding_model,
            cache_folder="/fs/fast/share/pingtai_cc/models/huggingface/",
        )
        self.embeddings = self.model.encode(texts)
        self.clusters: dict[int, list[int]] = None
        self._create_index(n_clusters)

    def adjust(self, cluster_id: int) -> None:
        """Split target cluster into two sub-clusters

        Args:
            cluster_id: ID of the cluster to split
        """
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} not found")

        cluster_indices = self.clusters[cluster_id]
        if len(cluster_indices) < 2:
            return

        labels = self._cluster(self.embeddings[cluster_indices])

        new_clusters = self.clusters.copy()
        next_cluster_id = max(self.clusters.keys()) + 1

        new_clusters[cluster_id] = [
            idx for i, idx in enumerate(cluster_indices) if labels[i] == 0
        ]
        new_clusters[next_cluster_id] = [
            idx for i, idx in enumerate(cluster_indices) if labels[i] == 1
        ]

        self.clusters = new_clusters

    def _create_index(self, n_clusters: int = 5) -> None:
        """Create initial clustering"""

        labels = self._cluster(self.embeddings, n_clusters)
        self.clusters = {
            i: np.where(labels == i)[0].tolist() for i in range(n_clusters)
        }

    def _cluster(self, embeddings: np.ndarray, n_clusters: int = 2) -> np.ndarray:
        """Cluster embeddings into n groups using Faiss KMeans

        Args:
            embeddings: Input embeddings array
            n_clusters: Number of clusters (default: 2)

        Returns:
            np.ndarray: Cluster labels for each embedding
        """
        dimension = embeddings.shape[1]
        kmeans = faiss.Kmeans(dimension, n_clusters, niter=20, verbose=False)
        embeddings_array = np.ascontiguousarray(embeddings.astype("float32"))
        kmeans.train(embeddings_array)

        _, labels = kmeans.index.search(embeddings_array, 1)
        return labels.flatten()

    def __iter__(self):
        """Iterate over stratum indices"""
        if self.clusters is None:
            raise ValueError("Index not created yet")

        for cls, stratum_indices in self.clusters.items():
            yield cls, stratum_indices

    def __getitem__(self, key: int):
        return self.clusters[key]
