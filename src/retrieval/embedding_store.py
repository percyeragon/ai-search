import hashlib
import logging
import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


logger = logging.getLogger(__name__)

def doc_content_hash(doc: dict) -> str:
    """Hash über die für das Embedding relevanten Felder."""
    text = (doc.get("title", "") + doc.get("text", "") + doc.get("Aufgabenbereich", ""))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingStore:
    """Persistenter PostgreSQL-Cache: doc_id -> (hash, embedding)."""

    def __init__(self, database_url: str, read_only: bool = False):
        if not isinstance(database_url, str) or not database_url.strip():
            raise TypeError(
                "EmbeddingStore benötigt EMBEDDING_DATABASE_URL als nichtleeren "
                f"PostgreSQL-String, erhalten: {type(database_url).__name__}"
            )
        self.database_url = database_url
        self.read_only = read_only
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
        self.doc_ids: list[str] = []
        self.hashes: dict[str, str] = {}
        self.embeddings: np.ndarray | None = None
        if not self.read_only:
            self._ensure_schema()
        self._load()

    def _connect(self):
        connection = psycopg.connect(self.database_url)
        register_vector(connection)
        return connection

    def _ensure_schema(self):
        # Nicht über _connect(): register_vector setzt die Extension voraus,
        # die hier erst angelegt wird.
        with psycopg.connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    doc_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    embedding vector({self.embedding_dimension}) NOT NULL
                )
                """
            )

    def _load(self):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT doc_id, content_hash, embedding "
                "FROM embeddings ORDER BY doc_id"
            ).fetchall()

        if not rows:
            return

        self.doc_ids = [row[0] for row in rows]
        self.hashes = {row[0]: row[1] for row in rows}
        self.embeddings = np.stack(
            [row[2].to_numpy() for row in rows]
        )

    def diff(self, corpus: dict) -> tuple[list[str], list[str], list[str]]:
        """Gibt (neue, geänderte, gelöschte) doc_ids zurück."""
        current_hashes = {doc_id: doc_content_hash(doc) for doc_id, doc in corpus.items()}
        old_ids = set(self.hashes.keys())
        new_ids = set(current_hashes.keys())

        added = list(new_ids - old_ids)
        removed = list(old_ids - new_ids)
        changed = [
            doc_id for doc_id in (new_ids & old_ids)
            if current_hashes[doc_id] != self.hashes[doc_id]
        ]
        return added, changed, removed

    def update(self, corpus: dict, embed_fn):
        """embed_fn: Callable[[list[str texts]], np.ndarray] - nur für neue/geänderte Docs."""
        added, changed, removed = self.diff(corpus)
        to_embed = added + changed

        if not to_embed and not removed:
            logger.info("Keine Änderungen im Korpus erkannt, Cache bleibt unverändert.")
            return

        logger.info(
            "Korpus-Diff: %d neu, %d geändert, %d entfernt (von %d Docs gesamt)",
            len(added), len(changed), len(removed), len(corpus),
        )

        old_lookup = dict(zip(self.doc_ids, self.embeddings)) if self.embeddings is not None else {}

        # Neue Embeddings nur für tatsächlich geänderte/neue Docs berechnen
        if to_embed:
            texts = [corpus[doc_id].get("text", "") for doc_id in to_embed]
            new_vecs = embed_fn(texts)  # teurer BGE-M3-Call, aber nur für Diff-Menge
            for doc_id, vec in zip(to_embed, new_vecs):
                old_lookup[doc_id] = vec

        # Entfernte Docs aus Lookup entfernen
        for doc_id in removed:
            old_lookup.pop(doc_id, None)

        # Finale Reihenfolge = aktuelle Korpus-Reihenfolge
        final_ids = list(corpus.keys())
        final_embeddings = np.stack([old_lookup[doc_id] for doc_id in final_ids])
        final_hashes = [doc_content_hash(corpus[doc_id]) for doc_id in final_ids]

        self._save(final_ids, final_hashes, final_embeddings)
        self.doc_ids, self.hashes, self.embeddings = (
            final_ids, dict(zip(final_ids, final_hashes)), final_embeddings
        )

    def _save(self, ids, hashes, embeddings):
        if self.read_only:
            raise RuntimeError("EmbeddingStore ist im Read-only-Modus.")
        rows = [
            (doc_id, content_hash, np.asarray(vector, dtype=np.float32).tolist())
            for doc_id, content_hash, vector in zip(ids, hashes, embeddings)
        ]
        with self._connect() as connection:
            connection.execute("DELETE FROM embeddings")
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO embeddings (doc_id, content_hash, embedding) "
                    "VALUES (%s, %s, %s)",
                    rows,
                )
        logger.info("Embedding-Datenbank aktualisiert: %s (%d Vektoren)", self.database_url, len(rows))