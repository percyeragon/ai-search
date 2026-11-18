class KGReranker:

    def __init__(self, corpus, kg, weight=0.3):

        self.corpus = corpus
        self.kg = kg
        self.weight = weight

    def rerank(self, query, scores):

        # KG-based query representation
        query_terms = self.kg.extract_query_terms(query)

        # top candidates (pre-ranked)
        top_candidates = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]

        for doc_id, base_score in top_candidates:

            doc_terms = self.kg.job_terms[doc_id]

            kg_score = self._compute_kg_score(query_terms, doc_terms)

            scores[doc_id] += self.weight * kg_score

        return scores

    def _compute_kg_score(self, query_terms, doc_terms):

        if not query_terms or not doc_terms:
            return 0.0

        score = 0.0
        term_graph = self.kg.term_graph

        for qt in query_terms:

            if qt in doc_terms:
                score += 2.0

            # only walk qt's (bounded) neighbor list instead of every
            # doc term - term_graph[qt] has ~ann_top_k entries, while
            # doc_terms (unigrams + bigrams of a full job posting) can run
            # into the hundreds/thousands, so this avoids an O(query_terms
            # * doc_terms) scan per candidate.
            neighbors = term_graph.get(qt)
            if not neighbors:
                continue

            if len(neighbors) < len(doc_terms):
                for dt, sim in neighbors.items():
                    if dt != qt and dt in doc_terms:
                        score += sim
            else:
                for dt in doc_terms:
                    if dt != qt:
                        sim = neighbors.get(dt)
                        if sim is not None:
                            score += sim

        # normalize
        return min(
            score / (len(query_terms) + len(doc_terms) + 1e-9),
            0.25  # keep your original cap logic
        )