import numpy as np


class HybridFusionRetriever:

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        alpha=0.7,
        candidate_k=100
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.candidate_k = candidate_k

    def _minmax_normalize(self, scores):

        if not scores:
            return {}

        min_s = min(scores.values())
        max_s = max(scores.values())

        if max_s == min_s:
            return {k: 1.0 for k in scores}

        return {
            doc_id: (score - min_s) / (max_s - min_s)
            for doc_id, score in scores.items()
        }

    def search(self, query, k=10):

        bm25_results = self.bm25.search_with_scores(
            query,
            k=self.candidate_k
        )

        dense_results = self.dense.search_with_scores(
            query,
            k=self.candidate_k
        )

        bm25_scores = self._minmax_normalize(
            dict(bm25_results)
        )

        dense_scores = self._minmax_normalize(
            dict(dense_results)
        )

        candidate_docs = set(bm25_scores) | set(dense_scores)

        fused_scores = {}

        for doc_id in candidate_docs:

            fused_scores[doc_id] = (
                self.alpha * bm25_scores.get(doc_id, 0.0)
                + (1 - self.alpha) * dense_scores.get(doc_id, 0.0)
            )

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc_id for doc_id, _ in ranked[:k]]

    def retrieve_with_scores(self, query, k=10):

        bm25_results = self.bm25.search_with_scores(
            query,
            k=self.candidate_k
        )

        dense_results = self.dense.search_with_scores(
            query,
            k=self.candidate_k
        )

        bm25_scores = self._minmax_normalize(
            dict(bm25_results)
        )

        dense_scores = self._minmax_normalize(
            dict(dense_results)
        )

        candidate_docs = set(bm25_scores) | set(dense_scores)

        fused_scores = {}

        for doc_id in candidate_docs:

            fused_scores[doc_id] = (
                self.alpha * bm25_scores.get(doc_id, 0.0)
                + (1 - self.alpha) * dense_scores.get(doc_id, 0.0)
            )

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:k]