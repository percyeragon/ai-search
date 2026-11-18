from collections import Counter, defaultdict
import math
import re
import numpy as np


class QueryExpander:

    def __init__(
        self,
        bm25,
        corpus,
        doc_ids,
        top_k_docs=10,
        top_k_terms=6,
        min_df=2,
        alpha_query=0.7,
        alpha_docs=0.3
    ):
        self.bm25 = bm25
        self.corpus = corpus
        self.doc_ids = doc_ids

        self.top_k_docs = top_k_docs
        self.top_k_terms = top_k_terms
        self.min_df = min_df

        self.alpha_query = alpha_query
        self.alpha_docs = alpha_docs

        self.num_docs = len(doc_ids)
        self.doc_freq = self._compute_doc_freq()

    # ==================================================
    # TOKENIZATION
    # ==================================================

    def _tokenize(self, text):
        return re.findall(r"\b[a-zA-ZäöüßA-Z0-9+#./-]{2,}\b", str(text).lower())

    def _normalize(self, tokens):
        return [t.strip() for t in tokens if len(t.strip()) > 1]

    # ==================================================
    # DF STATISTICS
    # ==================================================

    def _compute_doc_freq(self):
        df = defaultdict(int)

        for doc_id in self.doc_ids:
            doc = self.corpus[doc_id]
            tokens = set(self._tokenize(doc.get("text", "")))

            for t in tokens:
                df[t] += 1

        return df

    def _idf(self, term):
        df = self.doc_freq.get(term, 0)
        return math.log((self.num_docs + 1) / (df + 1)) + 1

    # ==================================================
    # RETRIEVAL
    # ==================================================

    def _retrieve(self, query):
        scores = np.array(self.bm25.get_scores(query))
        top_idx = np.argsort(scores)[-self.top_k_docs:][::-1]
        return scores, top_idx

    # ==================================================
    # TERM EXTRACTION
    # ==================================================

    def _extract_terms(self, doc_text):
        tokens = self._tokenize(doc_text)

        # remove ultra common noise
        return [
            t for t in tokens
            if len(t) > 2
        ]

    # ==================================================
    # QUERY SIGNALS
    # ==================================================

    def _query_terms(self, query):
        return set(self._tokenize(query))

    # ==================================================
    # SCORING
    # ==================================================

    def _score_terms(self, query, scores, top_idx):
        query_terms = self._query_terms(query)
        max_score = np.max(scores) + 1e-6

        term_scores = defaultdict(float)

        for rank, idx in enumerate(top_idx):
            doc_id = self.doc_ids[idx]
            doc_text = self.corpus[doc_id]["text"]

            terms = self._extract_terms(doc_text)
            tf = Counter(terms)

            # retrieval weight (rank-sensitive)
            retrieval_weight = scores[idx] / max_score
            position_weight = 1.0 / (rank + 1)

            doc_weight = (
                self.alpha_docs * retrieval_weight +
                (1 - self.alpha_docs) * position_weight
            )

            for term, freq in tf.items():

                if term in query_terms:
                    continue

                if self.doc_freq.get(term, 0) < self.min_df:
                    continue

                if freq < 2:
                    continue

                tf_weight = 1 + math.log(freq)
                idf = self._idf(term)

                term_scores[term] += doc_weight * tf_weight * idf

        return term_scores

    # ==================================================
    # EXPANSION
    # ==================================================

    def expand(self, query):
        scores, top_idx = self._retrieve(query)

        term_scores = self._score_terms(query, scores, top_idx)

        ranked = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)

        expansion_terms = [t for t, _ in ranked[:self.top_k_terms]]

        if not expansion_terms:
            return query

        return query + " " + " ".join(expansion_terms)