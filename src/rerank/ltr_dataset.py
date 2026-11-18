import numpy as np
from tqdm import tqdm
from src.rerank.features import FeatureExtractor


class LTRDatasetBuilder:

    def __init__(self, bm25, dense, corpus, qrels, ce_model=None, kg=None):

        self.bm25 = bm25
        self.dense = dense
        self.corpus = corpus
        self.qrels = qrels
        self.ce_model = ce_model

        self.fe = FeatureExtractor(corpus, kg)

    def build(self, queries, candidate_k=50):

        X = []
        y = []
        group = []

        for qid, query_obj in tqdm(queries.items()):

            query = query_obj["text"] if isinstance(query_obj, dict) else query_obj

            bm25_res = dict(self.bm25.search_with_scores(query, candidate_k))
            dense_res = dict(self.dense.search_with_scores(query, candidate_k))

            docs = list(set(bm25_res) | set(dense_res))

            q_count = 0

            for doc_id in docs:

                bm25_score = bm25_res.get(doc_id, 0.0)
                dense_score = dense_res.get(doc_id, 0.0)

                # -----------------------
                # CE score (optional)
                # -----------------------
                ce_score = 0.0
                if self.ce_model:
                    ce_score = self.ce_model.predict(query, self.corpus[doc_id]["text"])

                features = self.fe.extract(
                    query,
                    doc_id,
                    bm25_score,
                    dense_score,
                    ce_score
                )

                label = self.qrels.get(qid, {}).get(doc_id, 0)

                X.append(features)
                y.append(label)

                q_count += 1

            group.append(q_count)

        return np.array(X), np.array(y), group