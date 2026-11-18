import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3",
        max_length=192,
        batch_size=64,
        use_fp16=True,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )

        # FP16 nur sinnvoll auf GPU
        if use_fp16 and device == "cuda":
            self.model.model.half()

        self.batch_size = batch_size
        self._cache = {}  # (query, doc_id) -> score

    def rerank(self, query, corpus, doc_ids, top_k=10):
        pairs = []
        uncached_idx = []
        scores = [None] * len(doc_ids)

        for i, doc_id in enumerate(doc_ids):
            key = (query, doc_id)
            if key in self._cache:
                scores[i] = self._cache[key]
            else:
                text = corpus[doc_id]["text"]
                pairs.append((query, text))
                uncached_idx.append(i)

        if pairs:
            predicted = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            for idx, score in zip(uncached_idx, predicted):
                scores[idx] = float(score)
                self._cache[(query, doc_ids[idx])] = float(score)

        ranked = sorted(
            zip(doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # WICHTIG: (doc_id, score)-Tupel zurückgeben, nicht nur doc_id.
        # ComponentPipeline.search() erwartet dieses Format sowohl für
        # return_scores=True als auch beim Entpacken via `for doc, _ in ce_results`.
        return ranked[:top_k]

    def clear_cache(self):
        self._cache.clear()