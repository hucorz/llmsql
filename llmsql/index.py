from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class TextIndex:
    def __init__(self, tests, embedding_model: str = "all-MiniLM-L6-v2", delta=0.1):
        self.model = SentenceTransformer(
            embedding_model,
            cache_folder="/fs/fast/share/pingtai_cc/models/huggingface/",
        )
        self.embeddings = None
        self.clusters: Optional[dict[int, list[int]]] = None
        self.index = None
        self.delta = delta
        self._create_index(tests)

    def _create_index(self, texts, initial_clusters: int = 5):
        self.embeddings = self.model.encode(texts)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        kmeans = faiss.Kmeans(dimension, initial_clusters, niter=20, verbose=False)

        embeddings_array = np.ascontiguousarray(self.embeddings.astype("float32"))
        kmeans.train(embeddings_array)

        _, cluster_labels = kmeans.index.search(embeddings_array, 1)

        cluster_indices = {}
        for i in range(initial_clusters):
            cluster_indices[i] = np.where(cluster_labels == i)[0].tolist()

        self.clusters = cluster_indices

        self.index.add(embeddings_array)

    def __iter__(self):
        """Iterate over stratum indices"""
        if self.clusters is None:
            raise ValueError("Index not created yet")

        for cls, stratum_indices in self.clusters.items():
            yield cls, stratum_indices
