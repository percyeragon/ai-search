import numpy as np


class FeatureExtractor:

    def __init__(self, corpus, kg=None):
        self.corpus = corpus
        self.kg = kg or {}

    def extract(self, query, doc_id, bm25_score, dense_score, ce_score=None):

        doc = self.corpus[doc_id]["text"]

        # -----------------------------
        # 1. Basic retrieval signals
        # -----------------------------
        features = [
            bm25_score,
            dense_score
        ]

        # -----------------------------
        # 2. Cross-Encoder signal
        # -----------------------------
        if ce_score is None:
            ce_score = 0.0

        features.append(ce_score)

        # -----------------------------
        # 3. Query features
        # -----------------------------
        q_len = len(query.split())
        features.append(q_len)

        # term overlap
        q_terms = set(query.lower().split())
        d_terms = set(doc.lower().split())

        overlap = len(q_terms & d_terms) / (len(q_terms) + 1e-6)
        features.append(overlap)

        # -----------------------------
        # 4. Document features
        # -----------------------------
        doc_len = len(doc.split())
        features.append(doc_len)

        # -----------------------------
        # 5. KG features (optional but strong)
        # -----------------------------
        kg_score = 0

        if self.kg:
            q_entities = self.kg.get("query_entities", {}).get(query, set())
            d_entities = self.kg.get("doc_entities", {}).get(doc_id, set())

            if q_entities and d_entities:
                kg_score = len(q_entities & d_entities) / len(q_entities)

        features.append(kg_score)

        return np.array(features, dtype=np.float32)