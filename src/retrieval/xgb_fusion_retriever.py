import numpy as np
from src.rerank.features import FeatureExtractor


class XGBFusionRetriever:

    def __init__(self, bm25, dense, model, corpus, kg=None, candidate_k=100):

        self.bm25 = bm25
        self.dense = dense
        self.model = model
        self.candidate_k = candidate_k
        self.fe = FeatureExtractor(corpus, kg)
        self.corpus = corpus

    def search(self, query, k=10):

        bm25_res = dict(self.bm25.search_with_scores(query, self.candidate_k))
        dense_res = dict(self.dense.search_with_scores(query, self.candidate_k))

        docs = list(set(bm25_res) | set(dense_res))

        X = []

        for doc_id in docs:

            bm25_score = bm25_res.get(doc_id, 0.0)
            dense_score = dense_res.get(doc_id, 0.0)

            features = self.fe.extract(
                query,
                doc_id,
                bm25_score,
                dense_score
            )

            X.append(features)

        X = np.array(X)
        scores = self.model.predict(X)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:k]]