"""
Kompositionelle Retrieval-Pipeline.

Bisher gab es pro Kombination eine eigene fest verdrahtete Klasse
(HybridRRFKG, HybridRRFCE, HybridRRFKGCE, ...). Das erlaubte aber keine
freien Kombinationen wie BM25+KG, BM25+QE, Dense+CE usw.

Dieses Modul trennt die Pipeline in vier unabhängige Bausteine:

    [optional: Query Expansion] -> Base-Scorer -> [optional: KG-Rerank]
        -> [optional: Cross-Encoder-Rerank] -> Top-k

Jeder "Base-Scorer" implementiert nur eine einzige Methode:

    get_candidate_scores(query, k) -> dict[doc_id, score]

Alles, was diese Methode hat, kann beliebig mit QE/KG/CE kombiniert werden.

Platzierung: nach src/retrieval/pipeline.py kopieren.
"""

from src.retrieval.rrf_fusion import RRFFusion


def minmax_normalize(scores):
    """Skaliert ein scores-dict nach [0, 1]. Gleiche Logik wie in
    HybridFusionRetriever._minmax_normalize, hier wiederverwendbar gemacht."""

    if not scores:
        return {}

    vals = scores.values()
    lo, hi = min(vals), max(vals)

    if hi == lo:
        return {k: 1.0 for k in scores}

    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# =====================================================
# BASE-SCORER ADAPTER
# =====================================================
# Jeder Adapter macht aus einem bestehenden Retriever eine einheitliche
# Score-Quelle, ohne dessen Originalcode zu verändern.

class BM25Scorer:
    name = "bm25"

    def __init__(self, bm25):
        self.bm25 = bm25

    def get_candidate_scores(self, query, k=50):
        return dict(self.bm25.search_with_scores(query, k))


class DenseScorer:
    name = "dense"

    def __init__(self, dense):
        self.dense = dense

    def get_candidate_scores(self, query, k=50):
        return dict(self.dense.search_with_scores(query, k))


class RRFScorer:
    """Reciprocal-Rank-Fusion von BM25 + Dense, ohne Reranking.
    fuse_k=50 entspricht exakt dem bisherigen Verhalten von HybridRRF
    (dort war 50 hart codiert)."""

    name = "rrf"

    def __init__(self, bm25, dense, rrf_k=60, fuse_k=50):
        self.bm25 = bm25
        self.dense = dense
        self.fusion = RRFFusion(k=rrf_k)
        self.fuse_k = fuse_k

    def get_candidate_scores(self, query, k=50):
        bm25_res = self.bm25.search(query, self.fuse_k)
        dense_res = self.dense.search(query, self.fuse_k)
        return self.fusion.fuse(bm25_res, dense_res)


class AlphaFusionScorer:
    """Wrapper um die bestehende alpha-gewichtete Min-Max-Fusion
    (HybridFusionRetriever). Nutzt deren eigene Defaults (candidate_k=100
    intern), damit "alpha" ohne QE/KG/CE exakt die bisherigen
    Hybrid-Fusion-Resultate reproduziert."""

    name = "alpha"

    def __init__(self, hybrid_fusion_retriever):
        self.hf = hybrid_fusion_retriever

    def get_candidate_scores(self, query, k=50):
        return dict(self.hf.retrieve_with_scores(query, k=k))


class XGBScorer:
    """Wrapper um den trainierten XGB-Fusion-Retriever. Repliziert dessen
    search()-Logik bis zum Scoring-Schritt (vor dem finalen Top-k-Cut),
    damit auch hier ein scores-dict für KG/CE zur Verfügung steht."""

    name = "xgb"

    def __init__(self, xgb_fusion_retriever):
        self.xgb = xgb_fusion_retriever
        
    def get_candidate_scores(self, query, k=50):
            import numpy as np

            bm25_res = dict(self.xgb.bm25.search_with_scores(query, k))
            dense_res = dict(self.xgb.dense.search_with_scores(query, k))

            docs = list(set(bm25_res) | set(dense_res))

            X = []
            for doc_id in docs:
                features = self.xgb.fe.extract(
                    query,
                    doc_id,
                    bm25_res.get(doc_id, 0.0),
                    dense_res.get(doc_id, 0.0),
                )
                X.append(features)

            X = np.array(X)
            scores = self.xgb.model.predict(X)

            return dict(zip(docs, scores))

# =====================================================
# PIPELINE
# =====================================================

class ComponentPipeline:
    """
    Setzt Base-Scorer + optionale QE/KG/CE-Bausteine frei zusammen.

    normalize_before_kg:
        Default False, damit "rrf"+kg exakt die bisherigen
        Hybrid-RRF+KG-Zahlen reproduziert (dort wurden RRF-Scores
        unnormalisiert verwendet). BM25-Rohscores und KG-Boost (max. 0.075)
        liegen aber auf komplett unterschiedlichen Skalen — für BM25+KG
        oder Dense+KG sollte man normalize_before_kg=True setzen, sonst
        hat der KG-Boost praktisch keinen Effekt.
    """

    def __init__(
        self,
        scorer,
        corpus,
        query_expander=None,
        kg_reranker=None,
        ce_reranker=None,
        candidate_k=50,
        ce_candidate_k=20,
        normalize_before_kg=False,
    ):
        self.scorer = scorer
        self.corpus = corpus
        self.qe = query_expander
        self.kg = kg_reranker
        self.ce = ce_reranker
        self.candidate_k = candidate_k
        # Eigener, kleinerer Kandidaten-Pool fürs CE-Reranking: candidate_k
        # dient dem Recall-Funnel vor KG (braucht mehr Breite), CE profitiert
        # kaum davon, weit hinten liegende Kandidaten noch zu scoren.
        self.ce_candidate_k = ce_candidate_k
        self.normalize_before_kg = normalize_before_kg

    @property
    def name(self):
        parts = [self.scorer.name]
        if self.qe:
            parts.append("qe")
        if self.kg:
            parts.append("kg")
        if self.ce:
            parts.append("ce")
        return "+".join(parts)

    def search(self, query, k=10, allowed_ids=None, return_scores=False):

        q = self.qe.expand(query) if self.qe else query

        # Bei aktivem Filter brauchen wir genug Kandidaten, damit nach dem
        # Filtern noch sinnvoll viele übrig bleiben -> gesamten Korpus abfragen
        # statt nur candidate_k.
        effective_k = len(self.corpus) if allowed_ids is not None else self.candidate_k

        scores = self.scorer.get_candidate_scores(q, effective_k)

        if allowed_ids is not None:
            scores = {doc_id: s for doc_id, s in scores.items() if doc_id in allowed_ids}

        if self.kg:
            if self.normalize_before_kg:
                scores = minmax_normalize(scores)
            else:
                scores = dict(scores)  # KGReranker mutiert das dict in-place

            scores = self.kg.rerank(q, scores)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if self.ce:
            # min. so viele Kandidaten wie angefragte k, sonst könnte top_k=k
            # nicht erfüllt werden.
            ce_k = max(self.ce_candidate_k, k)
            candidates = [doc for doc, _ in ranked[:ce_k]]
            ce_results = self.ce.rerank(
                query=q,
                corpus=self.corpus,
                doc_ids=candidates,
                top_k=k,
            )
            if return_scores:
                return ce_results
            return [doc for doc, _ in ce_results]

        if return_scores:
            return ranked[:k]

        return [doc for doc, _ in ranked[:k]]