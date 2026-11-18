import hashlib
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import pickle
from pathlib import Path
from typing import Optional
from collections import defaultdict
from tqdm.auto import tqdm
import re

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticSkillKG:

    def __init__(
        self,
        corpus,
        model_name="BAAI/bge-m3",
        sim_threshold_skill=0.62,
        sim_threshold_query=0.55,
        min_term_len=3,
        ann_backend="auto",       # "faiss", "exact", or "auto" (faiss if available & vocab large)
        ann_top_k=50,             # neighbors per term to retrieve from the ANN index
        ann_threshold_vocab=8000  # above this vocab size, "auto" switches to faiss if installed
        , cache_dir: Optional[str] = None, force_rebuild: bool = False
        , model: Optional[SentenceTransformer] = None
        , query_cache=None
        , max_query_terms: Optional[int] = 30
    ):
        # Reuse an already-loaded SentenceTransformer (e.g. from a
        # DenseRetriever on the same corpus) instead of loading a second
        # copy of the same (large) embedding model. Callers that don't pass
        # one get the previous standalone-loading behavior.
        self.model = model if model is not None else SentenceTransformer(model_name)

        # Optional QueryEmbeddingCache shared with e.g. a DenseRetriever, so
        # a query already encoded there isn't pushed through the model a
        # second time here.
        self.query_cache = query_cache

        self.sim_threshold_skill = sim_threshold_skill
        self.sim_threshold_query = sim_threshold_query
        self.min_term_len = min_term_len

        # Caps how many terms extract_query_terms()/_terms_from_embedding()
        # return per query. Without this, "alle Terme >= sim_threshold_query"
        # can be several thousand for a large vocabulary (gemessen: bis zu
        # ~5000 Terme bei einem ~1M-Term-Vokabular), was KGReranker/score_job
        # zu einer O(query_terms * candidates) Python-Schleife macht und pro
        # Query mehrere Sekunden statt Millisekunden kostet. None = alte,
        # unbegrenzte Extraktion.
        self.max_query_terms = max_query_terms

        self.ann_backend = ann_backend
        self.ann_top_k = ann_top_k
        self.ann_threshold_vocab = ann_threshold_vocab

        # graph
        self.jobs = {}                  # job_id -> text
        self.job_terms = defaultdict(set)

        self.term_vocab = []
        self.term_index = {}            # term -> row index in term matrix
        self.term_matrix = None         # (n_terms, dim)

        self.term_graph = defaultdict(dict)
        self.term_faiss_index = None    # kept around for reuse in query expansion if desired

        self.job_ids = []
        self.job_matrix = None          # (n_jobs, dim)

        # query text -> extracted terms (in-process only, not persisted to
        # cache_dir). Grid-style eval runs reuse one SemanticSkillKG instance
        # across many pipeline combinations (bases x pp_stage x qe x ce x k),
        # so the same query text is re-encoded redundantly many times over
        # without this.
        self._query_term_cache = {}

        # cache directory (optional)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        # try to load from cache if available and not forcing rebuild
        loaded = False
        if self.cache_dir and not force_rebuild:
            try:
                loaded = self._try_load_cache()
                if not loaded:
                    logger.warning(
                        "KG-Cache unter %s nicht nutzbar (siehe vorherige Log-Zeile für Grund) — baue neu.",
                        self.cache_dir,
                    )
                else:
                    logger.info(
                        "KG-Cache erfolgreich geladen aus %s (%d Dokumente, %d Terme) — gleiche jetzt mit aktuellem Korpus ab.",
                        self.cache_dir, len(self.jobs), len(self.term_vocab),
                    )
            except Exception:
                logger.exception(
                    "Unerwarteter Fehler beim Laden des KG-Cache unter %s — baue neu.", self.cache_dir
                )
                loaded = False
        elif force_rebuild:
            logger.info("force_rebuild=True gesetzt — KG-Cache wird ignoriert und neu gebaut.")

        if not loaded:
            self._build(corpus)
        else:
            # Cache vorhanden: nicht neu bauen, sondern nur die Differenz zum
            # aktuellen Korpus (neu/geändert/entfernt) inkrementell verarbeiten.
            self._reconcile(corpus)

        if self.cache_dir:
            try:
                self._save_cache()
                logger.info("KG-Cache gespeichert unter %s.", self.cache_dir)
            except Exception:
                logger.exception("Fehler beim Speichern des KG-Cache unter %s.", self.cache_dir)

    # -------------------------------------------------
    # CACHE: LADEN / SPEICHERN
    # -------------------------------------------------
    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _try_load_cache(self) -> bool:
        """Lädt Cache-Artefakte von der Platte, OHNE den aktuellen Korpus zu
        vergleichen. Der Abgleich mit dem aktuell übergebenen Korpus passiert
        separat und inkrementell in _reconcile(), damit eine Änderung an
        einzelnen Dokumenten nicht den gesamten Cache verwirft."""
        cd = self.cache_dir
        if not cd.exists() or not cd.is_dir():
            logger.info("KG-Cache-Verzeichnis %s existiert nicht — kein Cache vorhanden.", cd)
            return False

        # required files
        required = [
            cd / "jobs.json",
            cd / "job_terms.json",
            cd / "term_vocab.json",
            cd / "term_matrix.npy",
            cd / "term_index.json",
            cd / "term_graph.pkl",
            cd / "job_ids.json",
            cd / "job_matrix.npy",
        ]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            logger.info("KG-Cache unvollständig unter %s, fehlende Dateien: %s", cd, missing)
            return False

        # load
        with open(cd / "jobs.json", "r", encoding="utf-8") as f:
            self.jobs = json.load(f)

        with open(cd / "job_terms.json", "r", encoding="utf-8") as f:
            jt = json.load(f)
            # convert lists back to sets
            self.job_terms = defaultdict(set, {k: set(v) for k, v in jt.items()})

        with open(cd / "term_vocab.json", "r", encoding="utf-8") as f:
            self.term_vocab = json.load(f)

        self.term_index = json.loads((cd / "term_index.json").read_text(encoding="utf-8"))

        self.term_matrix = np.load(cd / "term_matrix.npy").astype(np.float32)

        with open(cd / "term_graph.pkl", "rb") as f:
            self.term_graph = pickle.load(f)

        with open(cd / "job_ids.json", "r", encoding="utf-8") as f:
            self.job_ids = json.load(f)

        self.job_matrix = np.load(cd / "job_matrix.npy").astype(np.float32)

        return True

    def _save_cache(self):
        cd = self.cache_dir
        cd.mkdir(parents=True, exist_ok=True)

        with open(cd / "jobs.json", "w", encoding="utf-8") as f:
            json.dump(self.jobs, f, ensure_ascii=False)

        # convert sets to lists for json
        jt = {k: list(v) for k, v in self.job_terms.items()}
        with open(cd / "job_terms.json", "w", encoding="utf-8") as f:
            json.dump(jt, f, ensure_ascii=False)

        with open(cd / "term_vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.term_vocab, f, ensure_ascii=False)

        np.save(cd / "term_matrix.npy", self.term_matrix)

        (cd / "term_index.json").write_text(json.dumps(self.term_index, ensure_ascii=False), encoding="utf-8")

        with open(cd / "term_graph.pkl", "wb") as f:
            pickle.dump(self.term_graph, f)

        with open(cd / "job_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.job_ids, f, ensure_ascii=False)

        np.save(cd / "job_matrix.npy", self.job_matrix)

    # -------------------------------------------------
    # INKREMENTELLES UPDATE (statt vollem Rebuild bei Korpusänderung)
    # -------------------------------------------------
    def _reconcile(self, corpus):
        """Vergleicht Content-Hashes der geladenen Cache-Dokumente gegen den
        aktuell übergebenen Korpus und verarbeitet nur die Differenz:
        - neue Dokumente: Terme extrahieren, neue Term-Embeddings berechnen,
          neue Terme in den Graphen einhängen, Job-Embedding berechnen
        - geänderte Dokumente: wie neu behandelt (alte Terme/Embedding verworfen)
        - entfernte Dokumente: aus jobs/job_terms/job_matrix entfernt
        - unveränderte Dokumente: bleiben komplett unangetastet
        """
        old_hashes = {doc_id: self._hash_text(text) for doc_id, text in self.jobs.items()}
        new_hashes = {doc_id: self._hash_text(doc.get("text", "")) for doc_id, doc in corpus.items()}

        old_ids, new_ids = set(old_hashes), set(new_hashes)
        added = new_ids - old_ids
        removed = old_ids - new_ids
        common = new_ids & old_ids
        changed = {d for d in common if new_hashes[d] != old_hashes[d]}
        touched = added | changed

        if not touched and not removed:
            logger.info("Korpus unverändert (%d Dokumente) — kein Rebuild nötig.", len(new_ids))
            return

        logger.info(
            "Inkrementelles KG-Update: %d neu, %d geändert, %d entfernt, %d unverändert.",
            len(added), len(changed), len(removed), len(common) - len(changed),
        )

        # 1. veraltete Jobs entfernen (removed + changed; changed wird gleich neu extrahiert)
        for doc_id in removed | changed:
            self.jobs.pop(doc_id, None)
            self.job_terms.pop(doc_id, None)

        # 2. Terme nur für neue/geänderte Docs extrahieren
        new_terms_found = set()
        for doc_id in tqdm(touched, desc="Extracting terms (incremental)", unit="job"):
            text = corpus[doc_id].get("text", "")
            self.jobs[doc_id] = text
            terms = self._extract_terms(text)
            self.job_terms[doc_id] = terms
            new_terms_found.update(terms)

        # 3. nur wirklich neue Vokabel-Terme embedden (Rest bereits im Cache vorhanden)
        unseen_terms = sorted(t for t in new_terms_found if t not in self.term_index)
        if unseen_terms:
            new_embs = self.model.encode(
                unseen_terms,
                normalize_embeddings=True,
                batch_size=256,
                show_progress_bar=True,
            ).astype(np.float32)

            start_idx = len(self.term_vocab)
            for offset, term in enumerate(unseen_terms):
                self.term_index[term] = start_idx + offset
            self.term_vocab.extend(unseen_terms)

            if self.term_matrix is None or self.term_matrix.shape[0] == 0:
                self.term_matrix = new_embs
            else:
                self.term_matrix = np.vstack([self.term_matrix, new_embs])

            self._link_new_terms(unseen_terms)
        else:
            logger.info("Keine neuen Terme im Vokabular — Term-Graph bleibt unverändert.")

        # 4. Job-Embeddings: nur touched Docs neu encoden, Rest aus altem job_matrix übernehmen
        self._rebuild_job_matrix(touched)

        logger.info(
            "Inkrementelles KG-Update abgeschlossen: %d Dokumente, %d Terme im Vokabular.",
            len(self.jobs), len(self.term_vocab),
        )

    def _link_new_terms(self, new_terms: list):
        """Verlinkt neue Terme im term_graph gegen das (gewachsene) Gesamtvokabular.
        Bereits bestehende Alt-Alt-Kanten werden nicht neu berechnet, da sich
        deren Embeddings nicht verändert haben."""
        if not new_terms:
            return
        thresh = self.sim_threshold_skill
        new_idx = [self.term_index[t] for t in new_terms]
        new_matrix = self.term_matrix[new_idx]

        backend = self._resolve_ann_backend()
        if backend == "faiss":
            # Index muss über das gewachsene Vokabular neu aufgebaut werden,
            # da FAISS-Indizes nicht inkrementell erweiterbar sind (HNSW).
            dim = self.term_matrix.shape[1]
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
            index.add(self.term_matrix)
            self.term_faiss_index = index

            k = min(self.ann_top_k + 1, len(self.term_vocab))
            sims, idxs = index.search(new_matrix, k)

            for local_i, global_i in enumerate(new_idx):
                t1 = self.term_vocab[global_i]
                for sim, global_j in zip(sims[local_i], idxs[local_i]):
                    if global_j == -1 or global_j == global_i:
                        continue
                    if sim < thresh:
                        continue
                    t2 = self.term_vocab[global_j]
                    sim = float(sim)
                    existing = self.term_graph[t1].get(t2)
                    if existing is None or sim > existing:
                        self.term_graph[t1][t2] = sim
                        self.term_graph[t2][t1] = sim
        else:
            # exact: neue Terme gegen das GESAMTE Vokabular vergleichen,
            # alt-alt Paare werden nicht neu berechnet
            sims = new_matrix @ self.term_matrix.T
            new_idx_set = set(new_idx)

            for local_i, global_i in enumerate(new_idx):
                row = sims[local_i]
                candidates = np.where(row >= thresh)[0]
                t1 = self.term_vocab[global_i]
                for global_j in candidates:
                    if global_j == global_i:
                        continue
                    if global_j in new_idx_set and global_j < global_i:
                        continue  # neu-neu Paar nicht doppelt verarbeiten
                    t2 = self.term_vocab[global_j]
                    sim = float(row[global_j])
                    self.term_graph[t1][t2] = sim
                    self.term_graph[t2][t1] = sim

        logger.info(
            "Term-Graph erweitert: %d neue Terme verlinkt (Vokabular gesamt: %d).",
            len(new_terms), len(self.term_vocab),
        )

    def _rebuild_job_matrix(self, touched_ids):
        """Encodiert nur die touched Job-Texte neu; unveränderte Zeilen werden
        1:1 aus der alten job_matrix übernommen statt neu berechnet."""
        old_id_to_row = {jid: i for i, jid in enumerate(self.job_ids)}

        touched_ids = list(touched_ids)
        touched_texts = [self.jobs[jid] for jid in touched_ids]
        touched_emb_map = {}
        if touched_texts:
            touched_embs = self.model.encode(
                touched_texts,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=True,
            ).astype(np.float32)
            touched_emb_map = dict(zip(touched_ids, touched_embs))

        new_job_ids = list(self.jobs.keys())

        if touched_texts:
            dim = touched_embs.shape[1]
        elif self.job_matrix is not None and self.job_matrix.shape[0] > 0:
            dim = self.job_matrix.shape[1]
        else:
            dim = 1

        new_matrix = np.zeros((len(new_job_ids), dim), dtype=np.float32)

        for i, jid in enumerate(new_job_ids):
            if jid in touched_emb_map:
                new_matrix[i] = touched_emb_map[jid]
            else:
                old_row = old_id_to_row.get(jid)
                if old_row is not None:
                    new_matrix[i] = self.job_matrix[old_row]

        self.job_ids = new_job_ids
        self.job_matrix = new_matrix

    # -------------------------------------------------
    # 1. TERM EXTRACTION (NO LABELS AVAILABLE)
    # -------------------------------------------------
    def _extract_terms(self, text: str):
        text = text.lower()
        tokens = re.findall(r"[a-zäöüß]+", text)

        stop = {
            "und", "oder", "für", "mit", "eine", "einer", "einem",
            "der", "die", "das", "im", "in", "am", "bei", "als",
            "ich", "suche", "stelle", "vollzeit", "teilzeit"
        }

        tokens = [t for t in tokens if t not in stop and len(t) >= self.min_term_len]
        bigrams = [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]

        return set(tokens + bigrams)

    # -------------------------------------------------
    # 2. BUILD GRAPH (voller Rebuild — nur wenn kein nutzbarer Cache vorhanden ist)
    # -------------------------------------------------
    def _build(self, corpus):

        all_terms = set()
        for doc_id, doc in tqdm(corpus.items(), desc="Extracting terms", unit="job"):
            text = doc.get("text", "")
            self.jobs[doc_id] = text

            terms = self._extract_terms(text)
            self.job_terms[doc_id] = terms      # keep as set (fast intersections)
            all_terms.update(terms)

        # -------------------------
        # TERM EMBEDDINGS (batch encode once)
        # -------------------------
        self.term_vocab = list(all_terms)
        self.term_index = {t: i for i, t in enumerate(self.term_vocab)}

        if self.term_vocab:
            self.term_matrix = self.model.encode(
                self.term_vocab,
                normalize_embeddings=True,
                batch_size=256,
                show_progress_bar=True,
            ).astype(np.float32)
        else:
            self.term_matrix = np.zeros((0, 1), dtype=np.float32)

        # -------------------------
        # TERM GRAPH
        # -------------------------
        backend = self._resolve_ann_backend()
        if backend == "faiss":
            self._build_term_graph_faiss(top_k=self.ann_top_k)
        else:
            self._build_term_graph_chunked(chunk_size=1024)

        # -------------------------
        # JOB EMBEDDINGS (batch encode once, store as matrix for fast retrieval)
        # -------------------------
        self.job_ids = list(self.jobs.keys())
        job_texts = [self.jobs[jid] for jid in self.job_ids]

        if job_texts:
            self.job_matrix = self.model.encode(
                job_texts,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=True,
            ).astype(np.float32)
        else:
            self.job_matrix = np.zeros((0, 1), dtype=np.float32)

    def _resolve_ann_backend(self):
        if self.ann_backend == "exact":
            return "exact"
        if self.ann_backend == "faiss":
            if not _FAISS_AVAILABLE:
                raise ImportError(
                    "ann_backend='faiss' requested but faiss is not installed. "
                    "Install with `pip install faiss-cpu` (or faiss-gpu)."
                )
            return "faiss"
        # auto
        if _FAISS_AVAILABLE and len(self.term_vocab) >= self.ann_threshold_vocab:
            return "faiss"
        return "exact"

    def _build_term_graph_faiss(self, top_k=50):
        """
        ANN-based term graph construction using FAISS.
        Instead of computing all n^2 pairwise similarities, builds an HNSW
        index over term embeddings and queries each term's top_k nearest
        neighbors. This scales roughly O(n * top_k * log n) instead of O(n^2),
        which matters once the vocabulary grows into the tens of thousands
        (e.g. a large multi-source German job postings corpus).

        Trade-off: HNSW is approximate, so a small fraction of true neighbors
        above sim_threshold_skill may be missed. For exact results on smaller
        vocabularies, use ann_backend="exact".
        """
        n = len(self.term_vocab)
        if n == 0:
            return

        dim = self.term_matrix.shape[1]

        # cosine similarity == inner product since embeddings are normalized
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
        index.add(self.term_matrix)
        self.term_faiss_index = index

        thresh = self.sim_threshold_skill
        # +1 because the nearest neighbor of a term is always itself
        k = min(top_k + 1, n)

        batch_size = 2048
        with tqdm(total=n, desc="Building term graph (FAISS/HNSW)", unit="term") as pbar:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = self.term_matrix[start:end]

                sims, idxs = index.search(batch, k)

                for local_i in range(end - start):
                    global_i = start + local_i
                    t1 = self.term_vocab[global_i]

                    for sim, global_j in zip(sims[local_i], idxs[local_i]):
                        if global_j == -1 or global_j == global_i:
                            continue
                        if sim < thresh:
                            continue

                        t2 = self.term_vocab[global_j]
                        sim = float(sim)

                        # symmetric edge; cheap dict overwrite if already present
                        # from the neighbor's own search (keeps the higher of
                        # the two near-identical HNSW similarity estimates)
                        existing = self.term_graph[t1].get(t2)
                        if existing is None or sim > existing:
                            self.term_graph[t1][t2] = sim
                            self.term_graph[t2][t1] = sim

                pbar.update(end - start)
                pbar.set_postfix(edges=sum(len(v) for v in self.term_graph.values()) // 2)

    def _build_term_graph_chunked(self, chunk_size=1024):
        """
        Computes pairwise term similarities in chunks via matrix multiplication
        (cosine sim = dot product, since embeddings are normalized).
        Avoids O(n^2) Python-level np.dot calls; only keeps pairs above threshold.
        """
        n = len(self.term_vocab)
        if n == 0:
            return

        thresh = self.sim_threshold_skill
        n_chunks = (n + chunk_size - 1) // chunk_size

        with tqdm(total=n_chunks, desc="Building term graph", unit="chunk") as pbar:
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = self.term_matrix[start:end]            # (c, dim)

                # similarity of this chunk against the *upper triangle* remainder
                # only compare chunk rows against terms with index >= start
                sims = chunk @ self.term_matrix[start:].T       # (c, n-start)

                # mask self-similarity and lower triangle within overlapping region
                for local_i in range(end - start):
                    global_i = start + local_i
                    row = sims[local_i]

                    # indices relative to term_matrix[start:]
                    candidate_local_idx = np.where(row >= thresh)[0]

                    for local_j in candidate_local_idx:
                        global_j = start + local_j
                        if global_j <= global_i:
                            continue  # skip self & already-handled pairs

                        sim = float(row[local_j])
                        t1 = self.term_vocab[global_i]
                        t2 = self.term_vocab[global_j]

                        self.term_graph[t1][t2] = sim
                        self.term_graph[t2][t1] = sim

                pbar.update(1)
                pbar.set_postfix(edges=sum(len(v) for v in self.term_graph.values()) // 2)

    # -------------------------------------------------
    # 3. QUERY → TERMS
    # -------------------------------------------------
    def extract_query_terms(self, query: str):
        cached = self._query_term_cache.get(query)
        if cached is not None:
            return cached

        if self.query_cache is not None:
            q_emb = self.query_cache.get(query)
        else:
            q_emb = self.model.encode(query, normalize_embeddings=True).astype(np.float32)
        terms = self._terms_from_embedding(q_emb)
        self._query_term_cache[query] = terms
        return terms

    def _terms_from_embedding(self, q_emb):
        if self.term_matrix.shape[0] == 0:
            return []

        sims = self.term_matrix @ q_emb                     # (n_terms,) single matmul
        idx = np.where(sims >= self.sim_threshold_query)[0]

        scored = sorted(
            ((self.term_vocab[i], float(sims[i])) for i in idx),
            key=lambda x: x[1],
            reverse=True,
        )

        if self.max_query_terms is not None:
            scored = scored[: self.max_query_terms]

        return [t for t, _ in scored]

    def precompute_query_terms(self, queries, batch_size=256):
        """Batch-encodes all not-yet-cached query texts in one go instead of
        one SentenceTransformer forward pass per query. Call this once up
        front with every query text a caller is about to look up (e.g. all
        pp/qe variants in a grid run) so extract_query_terms() below just
        hits the cache."""
        uncached = sorted({q for q in queries if q not in self._query_term_cache})
        if not uncached:
            return

        embs = self.model.encode(
            uncached,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        ).astype(np.float32)

        for query, q_emb in zip(uncached, embs):
            self._query_term_cache[query] = self._terms_from_embedding(q_emb)

    # -------------------------------------------------
    # 4. JOB SCORING (GRAPH + EXACT + SEMANTIC) via set operations
    # -------------------------------------------------
    def score_job(self, query_terms, job_id):
        job_terms = self.job_terms[job_id]
        if not query_terms or not job_terms:
            return 0.0

        query_set = set(query_terms)

        # exact matches: O(min(len(a), len(b))) via set intersection
        exact_matches = query_set & job_terms
        score = 2.0 * len(exact_matches)

        # semantic neighbor matches: only look up graph neighbors of query
        # terms not already exact-matched, intersect with job_terms
        for qt in query_set - exact_matches:
            neighbors = self.term_graph.get(qt)
            if not neighbors:
                continue
            # iterate the smaller side for speed
            if len(neighbors) < len(job_terms):
                for jt, sim in neighbors.items():
                    if jt in job_terms:
                        score += sim
            else:
                for jt in job_terms:
                    sim = neighbors.get(jt)
                    if sim is not None:
                        score += sim

        return score / (len(query_terms) + len(job_terms) + 1e-9)

    # -------------------------------------------------
    # 5. RETRIEVAL (single query encode + batched dense similarity)
    # -------------------------------------------------
    def retrieve(self, query, top_k=5):
        if not self.job_ids:
            return []

        query_terms = self.extract_query_terms(query)
        if self.query_cache is not None:
            q_emb = self.query_cache.get(query)
        else:
            q_emb = self.model.encode(query, normalize_embeddings=True).astype(np.float32)

        # batched dense similarity for ALL jobs in one matmul instead of a loop
        dense_scores = self.job_matrix @ q_emb               # (n_jobs,)

        kg_scores = np.fromiter(
            (self.score_job(query_terms, jid) for jid in self.job_ids),
            dtype=np.float32,
            count=len(self.job_ids),
        )

        final_scores = 0.6 * kg_scores + 0.4 * dense_scores

        top_idx = np.argpartition(-final_scores, min(top_k, len(final_scores) - 1))[:top_k]
        top_idx = top_idx[np.argsort(-final_scores[top_idx])]

        return [(self.job_ids[i], float(final_scores[i])) for i in top_idx]

    # -------------------------------------------------
    # 6. BATCH RETRIEVAL (e.g. for BEIR-style evaluation over many queries)
    # -------------------------------------------------
    def retrieve_batch(self, queries: dict, top_k=5):
        """
        queries: dict of {query_id: query_text}
        Returns: dict of {query_id: [(job_id, score), ...]}
        Shows a progress bar since BEIR eval runs can involve thousands of queries.
        """
        results = {}
        for qid, qtext in tqdm(queries.items(), desc="Retrieving", unit="query"):
            results[qid] = self.retrieve(qtext, top_k=top_k)
        return results