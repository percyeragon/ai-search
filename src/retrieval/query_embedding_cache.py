from collections import OrderedDict
import numpy as np


class QueryEmbeddingCache:
    """Memoizes single-query embeddings by exact query text.

    DenseRetriever and SemanticSkillKG each encode the query text
    independently, so in a request that combines a dense-family scorer with
    the KG reranker (e.g. "rrf+kg", "dense+kg"), the identical query text
    was being pushed through the (large) embedding model twice. Sharing one
    cache instance between both eliminates the second forward pass.

    Bounded via LRU eviction (maxsize) since this lives for the lifetime of
    a long-running server process and must not grow without bound as
    distinct query texts accumulate.
    """

    def __init__(self, model, maxsize: int = 2048):
        self.model = model
        self._maxsize = maxsize
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached

        emb = self.model.encode(text, normalize_embeddings=True).astype(np.float32)
        self._cache[text] = emb
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return emb

    def invalidate(self, text: str = None) -> None:
        """Drops one cached entry, or the whole cache if text is None.

        Call with no argument whenever the underlying corpus/model changes
        (e.g. on a pipeline reload) so stale embeddings can't leak into a
        rebuilt index.
        """
        if text is None:
            self._cache.clear()
        else:
            self._cache.pop(text, None)

    def __len__(self) -> int:
        return len(self._cache)
