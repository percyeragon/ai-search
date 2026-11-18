import os
import json
import hashlib
from pathlib import Path

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from src.retrieval.embedding_store import EmbeddingStore

# Speicherbedarf beim Encoden: kleine Batches und begrenzte Sequenzlänge
# halten den RAM/VRAM-Bedarf von bge-m3 bei wenigen GB (Default wäre 8192 Tokens).
DENSE_ENCODE_BATCH_SIZE = int(os.getenv("DENSE_ENCODE_BATCH_SIZE", "4"))
DENSE_MAX_SEQ_LENGTH = int(os.getenv("DENSE_MAX_SEQ_LENGTH", "512"))


def _doc_text_hash(doc: dict) -> str:
    """Hash über die für das Embedding relevanten Felder (title + text)."""
    payload = (doc.get("title", "") or "") + "\x00" + (doc.get("text", "") or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DenseRetriever:

    def __init__(self, corpus, cache_path=None, device=None, read_only=False):
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.texts = [corpus[cid]["text"] for cid in self.doc_ids]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer("BAAI/bge-m3", device=self.device)
        self.model.max_seq_length = DENSE_MAX_SEQ_LENGTH

        # Optional shared QueryEmbeddingCache, set by the caller once it
        # exists (see query_embedding_cache.py), so a query encoded here can
        # be reused by e.g. the KG reranker instead of re-encoded.
        self.query_cache = None
        self.read_only = read_only

        self.embeddings = self._load_or_encode_corpus_incremental(cache_path)
        self.dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)
        if self.device == "cuda" and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(self.embeddings)

    # =====================================================
    # Inkrementelles Laden / Encoden
    # =====================================================

    def _cache_paths(self, cache_path):
        return {"db": cache_path}

    def _load_or_encode_corpus_incremental(self, cache_path):
        if not cache_path:
            print("Kein cache_path angegeben, encode kompletten Korpus ohne Caching...")
            return self._encode_texts(self.texts)

        paths = self._cache_paths(cache_path)
        current_hashes = {cid: _doc_text_hash(self.corpus[cid]) for cid in self.doc_ids}

        old_ids, old_hashes, old_embeddings = self._load_cache(paths)

        if old_embeddings is None:
            if self.read_only:
                raise RuntimeError(
                    "Kein Embedding-Cache vorhanden. Der Master muss den Cache "
                    "zuerst aufbauen."
                )
            # Kein Cache vorhanden -> voller Encode
            print("Kein Cache gefunden, encode kompletten Korpus...")
            embeddings = self._encode_texts(self.texts)
            self._save_cache(paths, self.doc_ids, current_hashes, embeddings)
            return embeddings

        # Diff berechnen: neu / geändert / entfernt
        old_id_set = set(old_ids)
        new_id_set = set(self.doc_ids)

        added = [cid for cid in self.doc_ids if cid not in old_id_set]
        changed = [
            cid for cid in self.doc_ids
            if cid in old_id_set and old_hashes.get(cid) != current_hashes[cid]
        ]
        removed = old_id_set - new_id_set

        to_encode = added + changed

        print(
            f"Korpus-Diff: {len(added)} neu, {len(changed)} geändert, "
            f"{len(removed)} entfernt (von {len(self.doc_ids)} Docs gesamt)."
        )

        if not to_encode and not removed:
            print("Keine Änderungen erkannt, verwende bestehenden Cache.")
            lookup = dict(zip(old_ids, old_embeddings))
            embeddings = np.stack([lookup[cid] for cid in self.doc_ids])
            return embeddings

        if self.read_only:
            raise RuntimeError(
                "Der Embedding-Cache ist nicht aktuell. Bitte zuerst den Master "
                "neu laden lassen."
            )

        lookup = dict(zip(old_ids, old_embeddings))

        if to_encode:
            new_texts = [self.corpus[cid]["text"] for cid in to_encode]
            new_vecs = self._encode_texts(new_texts)
            for cid, vec in zip(to_encode, new_vecs):
                lookup[cid] = vec

        for cid in removed:
            lookup.pop(cid, None)

        embeddings = np.stack([lookup[cid] for cid in self.doc_ids])

        self._save_cache(paths, self.doc_ids, current_hashes, embeddings)
        return embeddings

    def _encode_texts(self, texts):
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=DENSE_ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def _load_cache(self, paths):
        store = EmbeddingStore(paths["db"], read_only=self.read_only)
        if store.embeddings is None:
            return None, None, None
        return store.doc_ids, store.hashes, store.embeddings

    def _save_cache(self, paths, ids, hashes, embeddings):
        store = EmbeddingStore(paths["db"], read_only=self.read_only)
        store._save(ids, [hashes[doc_id] for doc_id in ids], embeddings)
        print(f"Embedding-Datenbank aktualisiert: {paths['db']}")

    # =====================================================
    # Unverändert
    # =====================================================

    def _encode_queries(self, queries):
        return self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def search(self, query, k=10):
        return [doc_id for doc_id, _ in self.search_with_scores(query, k)]

    def search_with_scores(self, query, k=10):
        if self.query_cache is not None:
            q_emb = self.query_cache.get(query).reshape(1, -1)
        else:
            q_emb = self._encode_queries([query])
        scores, idx = self.index.search(q_emb, k)
        return [
            (self.doc_ids[i], float(score))
            for i, score in zip(idx[0], scores[0])
        ]

    def search_batch_with_scores(self, queries, k=10):
        q_emb = self._encode_queries(queries)
        scores, idx = self.index.search(q_emb, k)
        results = []
        for query_scores, query_idx in zip(scores, idx):
            results.append([
                (self.doc_ids[i], float(score))
                for i, score in zip(query_idx, query_scores)
            ])
        return results