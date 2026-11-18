# scripts/train_xgb_model.py
import json
import os
from pathlib import Path

import numpy as np

from src.retrieval.bm25_retrieval import BM25Retriever
from src.retrieval.dense_retrieval import DenseRetriever
from src.rerank.ltr_dataset import LTRDatasetBuilder
from src.rerank.xgb_fusion import XGBFusionModel

DATASET_DIR = Path("data/processed/jobs_interamt_log_queries")

# =====================================================
# CACHE-PFADE (konsistent zu run_pipeline_grid.py / rest_service.py)
# =====================================================
MODELS_DIR = Path("models")
MODEL_OUT = MODELS_DIR / "xgb_fusion.json"
EMBEDDING_DATABASE_URL = os.getenv(
    "EMBEDDING_DATABASE_URL",
    "postgresql://ai_suche:ai_suche@localhost:5432/ai_suche",
)
# LTR-Trainingsdaten (X, y, group): candidate_k=50 über BM25+Dense für
# jede Train-Query neu zu generieren ist der teuerste Teil dieses Skripts
# und bei wiederholten Trainingsläufen (z.B. Hyperparameter-Tuning)
# unnötig, solange sich qrels/corpus/candidate_k nicht ändern.
LTR_DATASET_CACHE_PATH = MODELS_DIR / "ltr_dataset_train.npz"


def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["_id"]] = item
    return data


def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, docid, score = line.strip().split("\t")
            qrels.setdefault(qid, {})[docid] = float(score)
    return qrels


def load_cached_ltr_dataset(cache_path: Path):
    """Lädt (X, y, group) aus dem .npz-Cache, falls vorhanden und lesbar.
    Gibt None zurück, wenn kein Cache existiert oder er beschädigt ist --
    der Aufrufer baut das Dataset dann wie gewohnt neu auf."""
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            return npz["X"], npz["y"], npz["group"]
    except Exception as e:
        print(f"[WARN] LTR-Dataset-Cache unlesbar ({e}), baue neu auf")
        return None


def save_cached_ltr_dataset(cache_path: Path, X, y, group):
    """Speichert (X, y, group) atomar: erst in eine Tempfile im selben
    Verzeichnis schreiben, dann per rename ersetzen -- vermeidet einen
    kaputten/halb geschriebenen Cache bei Abbruch mitten im np.savez()."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez(tmp_path, X=X, y=y, group=group)
    tmp_path.replace(cache_path)

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(DATASET_DIR / "corpus.jsonl")
    queries = load_jsonl(DATASET_DIR / "queries.jsonl")
    qrels = load_qrels(DATASET_DIR / "qrels" / "train.tsv")

    # Defensiv: nur die Queries übergeben, die tatsächlich im Train-Split sind,
    # falls LTRDatasetBuilder.build() nicht selbst nach qrels.keys() filtert
    queries = {qid: q for qid, q in queries.items() if qid in qrels}

    cached_dataset = load_cached_ltr_dataset(LTR_DATASET_CACHE_PATH)
    if cached_dataset is not None:
        X, y, group = cached_dataset
        print(f"LTR-Dataset aus Cache geladen: {LTR_DATASET_CACHE_PATH}")
    else:
        bm25 = BM25Retriever(corpus)
        dense = DenseRetriever(corpus, cache_path=EMBEDDING_DATABASE_URL)

        builder = LTRDatasetBuilder(bm25=bm25, dense=dense, corpus=corpus, qrels=qrels)
        X, y, group = builder.build(queries, candidate_k=50)
        save_cached_ltr_dataset(LTR_DATASET_CACHE_PATH, X, y, group)
        print(f"LTR-Dataset aufgebaut und gecacht unter {LTR_DATASET_CACHE_PATH}")

    model = XGBFusionModel()
    model.train(X, y, group=group)

    model.save(MODEL_OUT)
    print(f"Modell gespeichert unter {MODEL_OUT}")


if __name__ == "__main__":
    main()