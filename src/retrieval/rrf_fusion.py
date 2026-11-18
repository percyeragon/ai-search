class RRFFusion:

    def __init__(self, k=60):
        self.k = k

    def _rrf_scores(self, results):

        scores = {}

        for rank, doc_id in enumerate(results):
            scores[doc_id] = 1 / (self.k + rank + 1)

        return scores

    def fuse(self, bm25_results, dense_results):

        scores = {}

        for doc_id, score in self._rrf_scores(bm25_results).items():
            scores[doc_id] = scores.get(doc_id, 0) + score

        for doc_id, score in self._rrf_scores(dense_results).items():
            scores[doc_id] = scores.get(doc_id, 0) + score

        return scores