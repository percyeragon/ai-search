from rank_bm25 import BM25Okapi
import numpy as np


class BM25Retriever:

    def __init__(self, corpus):

        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.texts = [corpus[cid]["text"] for cid in self.doc_ids]

        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def get_scores(self, query):

        tokenized_query = query.lower().split()
        return self.bm25.get_scores(tokenized_query)

    def search(self, query, k=10):

        scores = self.get_scores(query)

        top_k_idx = np.argsort(scores)[::-1][:k]

        return [
            self.doc_ids[i]
            for i in top_k_idx
        ]

    def search_with_scores(self, query, k=10):

        scores = self.get_scores(query)

        top_k_idx = np.argsort(scores)[::-1][:k]

        return [
            (self.doc_ids[i], float(scores[i]))
            for i in top_k_idx
        ]