"""
Laufzeittest der Suche entlang der Service-Architektur (lokal, ohne Docker/PostgreSQL).

Baut dieselben Komponenten wie ``rest_service.build_pipeline_components()``
auf, ruft für jede Query die echte ``rest_service.search()``-Funktion auf und
misst dabei jede Architektur-Stufe einzeln:

    Preprocessing (Spelling, Abkürzungen, Behördensprache, Varianten, Komposita)
      -> Filter / Geo-Distanzen
      -> Query Expansion (optional)
      -> Retrieval + Fusion (BM25 / Dense / RRF / Alpha / XGB)
           ├─ bm25.scoring
           ├─ dense.query_encode
           └─ dense.faiss_search
      -> Cross-Encoder-Reranking (optional)
      -> Ergebnisaufbau + Sortierung

Abweichungen zum Produktivbetrieb (bewusst, damit es lokal läuft):
  * Korpus kommt aus runtime-data/corpus.jsonl statt von der Interamt-API
    (wird aber mit _normalize_api_record auf das aktuelle Feldschema gebracht).
  * Corpus-Embeddings werden in einer lokalen .npz-Datei statt in PostgreSQL
    gecacht. Zur Laufzeit wird in beiden Fällen nur der FAISS-Index genutzt,
    die Suchlatenz ist daher identisch.
  * Gemessen wird In-Process (ohne HTTP/uvicorn-Overhead).

Aufruf (aus dem Repo-Root):
    .env/Scripts/python -m src.scripts.benchmark_latency
    .env/Scripts/python -m src.scripts.benchmark_latency --n-queries 200 --scenarios bm25,rrf,rrf+ce
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Muss vor dem Import von rest_service gesetzt sein (pgeocode wird dort beim
# Import initialisiert) -- sonst versucht pgeocode, die PLZ-Daten zu laden.
os.environ.setdefault("PGEOCODE_DATA_DIR", str(Path("data/pgeocode").resolve()))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.api import rest_service as svc  # noqa: E402
from src.api.interamt_data import _normalize_api_record  # noqa: E402
from src.retrieval import pipeline as pipeline_mod  # noqa: E402
from src.retrieval.dense_retrieval import DenseRetriever  # noqa: E402


# =====================================================
# SZENARIEN
# =====================================================

NO_PREPROCESSING = {
    "correct_spelling": False,
    "expand_abbreviations": False,
    "normalize_spelling_variants": False,
    "normalize_behoerdensprache": False,
    "split_compounds": False,
}

SCENARIOS: dict[str, dict] = {
    "bm25": {"base": "bm25"},
    "bm25:no_preproc": {"base": "bm25", **NO_PREPROCESSING},
    "dense": {"base": "dense"},
    "rrf": {"base": "rrf"},
    "alpha": {"base": "alpha"},
    "xgb": {"base": "xgb"},
    "bm25+qe": {"base": "bm25", "use_qe": True},
    "dense+ce": {"base": "dense", "use_ce": True},
    "rrf+ce": {"base": "rrf", "use_ce": True},
    "rrf+qe+ce": {"base": "rrf", "use_qe": True, "use_ce": True},
    "rrf:geo_filter": {
        "base": "rrf",
        "filters": {"plz_oder_ort": "10115", "max_distance_km": 50},
    },
    "rrf:cat_filter": {
        "base": "rrf",
        "filters": {"beschaeftigungsart": "Vollzeit"},
    },
    "rrf:geo_filter+sort_distance": {
        "base": "rrf",
        "filters": {"plz_oder_ort": "10115", "max_distance_km": 50},
        "sort_by": "distance",
    },
}

# Reihenfolge der Top-Level-Stufen in der Ausgabe. Sub-Stufen (mit Punkt
# nach Komponenten-Präfix, z.B. "dense.query_encode") sind in ihrer
# Eltern-Stufe enthalten und werden NICHT zur Summe addiert.
TOP_LEVEL_STAGES = [
    "pre.spelling",
    "pre.abbreviations",
    "pre.behoerdensprache",
    "pre.variants",
    "pre.compounds",
    "filter",
    "geo_distances",
    "query_expansion",
    "retrieval_fusion",
    "cross_encoder",
    "result_building",
    "sort",
]
SUB_STAGES = ["bm25.scoring", "dense.query_encode", "dense.faiss_search"]


# =====================================================
# INSTRUMENTIERUNG
# =====================================================

class StageTimer:
    """Sammelt pro Request die Zeit je Stufe (ms, aufsummiert bei Mehrfachaufruf)."""

    def __init__(self):
        self.current: dict[str, float] = defaultdict(float)
        self.cuda = torch.cuda.is_available()

    def reset(self):
        self.current = defaultdict(float)

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def wrap(self, fn, stage):
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            self._sync()
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self._sync()
                self.current[stage] += (time.perf_counter() - t0) * 1000.0

        return timed

    def patch_attr(self, obj, attr, stage):
        setattr(obj, attr, self.wrap(getattr(obj, attr), stage))


class _TimedIndexProxy:
    """FAISS-Indizes sind SWIG-Objekte -> Methode nicht direkt patchbar."""

    def __init__(self, index, timer: StageTimer):
        self._index = index
        self.search = timer.wrap(index.search, "dense.faiss_search")

    def __getattr__(self, name):
        return getattr(self._index, name)


def instrument(pc, timer: StageTimer):
    timer.patch_attr(pc.spell_corrector, "correct", "pre.spelling")
    timer.patch_attr(pc.abbreviation_expander, "expand", "pre.abbreviations")
    timer.patch_attr(pc.behoerden_normalizer, "expand_query", "pre.behoerdensprache")
    timer.patch_attr(pc.variant_normalizer, "expand_query", "pre.variants")
    timer.patch_attr(pc.compound_splitter, "expand_query", "pre.compounds")
    timer.patch_attr(pc.query_expander, "expand", "query_expansion")
    if pc.ce_reranker is not None:
        timer.patch_attr(pc.ce_reranker, "rerank", "cross_encoder")

    timer.patch_attr(pc.bm25, "get_scores", "bm25.scoring")
    timer.patch_attr(pc.dense, "_encode_queries", "dense.query_encode")
    timer.patch_attr(pc.dense.query_cache, "get", "dense.query_encode")
    pc.dense.index = _TimedIndexProxy(pc.dense.index, timer)

    # Base-Scorer werden pro Request in make_scorer() neu erzeugt -> Klassen patchen.
    for cls in (
        pipeline_mod.BM25Scorer,
        pipeline_mod.DenseScorer,
        pipeline_mod.RRFScorer,
        pipeline_mod.AlphaFusionScorer,
        pipeline_mod.XGBScorer,
    ):
        timer.patch_attr(cls, "get_candidate_scores", "retrieval_fusion")

    # Modul-Funktionen, die search() über Modul-Globals auflöst.
    timer.patch_attr(svc, "get_allowed_doc_ids", "filter")
    timer.patch_attr(svc, "_compute_min_distances_km", "geo_distances")
    timer.patch_attr(svc, "build_result_item", "result_building")
    timer.patch_attr(svc, "_sort_results", "sort")


# =====================================================
# KOMPONENTEN-AUFBAU (wie build_pipeline_components, aber lokal)
# =====================================================

class FileCachedDenseRetriever(DenseRetriever):
    """Wie DenseRetriever, aber Embedding-Cache in .npz statt PostgreSQL."""

    def _load_cache(self, paths):
        path = Path(paths["db"])
        if not path.exists():
            return None, None, None
        data = np.load(path, allow_pickle=False)
        ids = [str(x) for x in data["ids"]]
        hashes = dict(zip(ids, (str(h) for h in data["hashes"])))
        return ids, hashes, data["embeddings"].astype(np.float32)

    def _save_cache(self, paths, ids, hashes, embeddings):
        path = Path(paths["db"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            ids=np.array(ids),
            hashes=np.array([hashes[i] for i in ids]),
            embeddings=embeddings.astype(np.float32),
        )
        print(f"Embedding-Cache gespeichert: {path}")


def _timed_build(startup: dict, name: str, fn):
    print(f"[startup] {name} ...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    startup[name] = round(time.perf_counter() - t0, 3)
    print(f"[startup] {name}: {startup[name]:.2f}s", flush=True)
    return result


def build_components(args, startup: dict):
    pc = svc.PipelineComponents()

    def load_corpus():
        corpus = svc.load_jsonl(Path(args.corpus))
        return {doc_id: _normalize_api_record(doc) for doc_id, doc in corpus.items()}

    pc.corpus = _timed_build(startup, "corpus_load", load_corpus)
    pc.spell_corrector = _timed_build(startup, "spell_corrector", lambda: svc.SpellCorrector(pc.corpus))
    pc.abbreviation_expander = _timed_build(startup, "abbreviation_expander", svc.AbbreviationExpander)
    pc.behoerden_normalizer = _timed_build(startup, "behoerden_normalizer", svc.BehoerdenNormalizer)
    pc.variant_normalizer = _timed_build(startup, "variant_normalizer", svc.SpellingVariantNormalizer)
    pc.compound_splitter = _timed_build(startup, "compound_splitter", lambda: svc.CompoundSplitter(pc.corpus))
    pc.bm25 = _timed_build(startup, "bm25_index", lambda: svc.BM25Retriever(pc.corpus))
    pc.dense = _timed_build(
        startup,
        "dense_model+index",
        lambda: FileCachedDenseRetriever(pc.corpus, cache_path=str(args.embedding_cache), device=args.device),
    )
    pc.dense.query_cache = svc.QueryEmbeddingCache(pc.dense.model)
    pc.query_expander = _timed_build(
        startup,
        "query_expander",
        lambda: svc.QueryExpander(bm25=pc.bm25, corpus=pc.corpus, doc_ids=pc.bm25.doc_ids),
    )
    pc.ce_reranker = None
    if not args.skip_ce:
        pc.ce_reranker = _timed_build(startup, "cross_encoder_model", svc.CrossEncoderReranker)
    pc.alpha_retriever = svc.HybridFusionRetriever(bm25_retriever=pc.bm25, dense_retriever=pc.dense, alpha=0.2)

    pc.xgb_retriever = None
    if svc.XGB_MODEL_PATH.exists():
        def load_xgb():
            model = svc.XGBFusionModel()
            model.load(svc.XGB_MODEL_PATH)
            return svc.XGBFusionRetriever(bm25=pc.bm25, dense=pc.dense, model=model, corpus=pc.corpus)

        pc.xgb_retriever = _timed_build(startup, "xgb_model", load_xgb)

    lats, lons, ids = _timed_build(startup, "geo_cache", lambda: svc.build_doc_geo_cache(pc.corpus))
    pc.doc_lats, pc.doc_lons, pc.doc_ids_with_geo = lats, lons, ids
    return pc


# =====================================================
# QUERIES
# =====================================================

def load_queries(path: Path, n: int, warmup: int, seed: int) -> tuple[list[str], list[str]]:
    texts = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = (json.loads(line).get("text") or "").strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                texts.append(text)
    rng = random.Random(seed)
    rng.shuffle(texts)
    if len(texts) < n + warmup:
        raise SystemExit(f"Nur {len(texts)} eindeutige Queries vorhanden, benötigt {n + warmup}.")
    return texts[:warmup], texts[warmup:warmup + n]


# =====================================================
# MESSUNG
# =====================================================

def clear_query_caches(pc):
    pc.dense.query_cache.invalidate()
    if pc.ce_reranker is not None:
        pc.ce_reranker.clear_cache()


def run_scenario(name, params, pc, timer, queries, warmup_queries, args):
    base_req = {"top_k": args.top_k, "candidate_k": args.candidate_k, **params}

    clear_query_caches(pc)
    for q in warmup_queries:
        svc.search(svc.SearchRequest(query=q, **base_req))
    if not args.keep_caches:
        clear_query_caches(pc)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rows = []
    for q in queries:
        req = svc.SearchRequest(query=q, **base_req)
        timer.reset()
        timer._sync()
        t0 = time.perf_counter()
        resp = svc.search(req)
        timer._sync()
        total = (time.perf_counter() - t0) * 1000.0

        stages = dict(timer.current)
        accounted = sum(stages.get(s, 0.0) for s in TOP_LEVEL_STAGES)
        stages["other"] = max(total - accounted, 0.0)
        stages["total"] = total
        rows.append(
            {
                "scenario": name,
                "query": q,
                "num_results": len(resp.results),
                "num_candidates_after_filter": resp.num_candidates_after_filter,
                "preprocessing_steps": len(resp.preprocessing_applied),
                **stages,
            }
        )

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    return rows, peak_mb


def pct(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def summarize(rows, scenario, peak_mb):
    stages = TOP_LEVEL_STAGES + SUB_STAGES + ["other", "total"]
    summary = {"scenario": scenario, "n": len(rows), "gpu_peak_mb": peak_mb, "stages": {}}
    for stage in stages:
        values = [r.get(stage, 0.0) for r in rows]
        if not any(values):
            continue
        summary["stages"][stage] = {
            "mean": statistics.fmean(values),
            "p50": pct(values, 50),
            "p95": pct(values, 95),
            "p99": pct(values, 99),
            "max": max(values),
        }
    total_mean = summary["stages"]["total"]["mean"]
    summary["throughput_qps_sequential"] = 1000.0 / total_mean if total_mean else None
    summary["mean_results"] = statistics.fmean(r["num_results"] for r in rows)
    return summary


def print_summary(summary):
    st = summary["stages"]
    total_mean = st["total"]["mean"]
    print(
        f"\n=== {summary['scenario']}  (n={summary['n']}, "
        f"{summary['throughput_qps_sequential']:.1f} q/s sequentiell, "
        f"Ø {summary['mean_results']:.1f} Treffer"
        + (f", GPU-Peak {summary['gpu_peak_mb']:.0f} MB" if summary["gpu_peak_mb"] else "")
        + ") ==="
    )
    print(f"{'Stufe':<24}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}{'Anteil':>8}  [ms]")
    for stage in TOP_LEVEL_STAGES + SUB_STAGES + ["other", "total"]:
        if stage not in st:
            continue
        s = st[stage]
        label = ("  └ " + stage) if stage in SUB_STAGES else stage
        share = "" if stage in ("total",) else f"{100 * s['mean'] / total_mean:6.1f}%"
        print(
            f"{label:<24}{s['mean']:9.2f}{s['p50']:9.2f}{s['p95']:9.2f}"
            f"{s['p99']:9.2f}{s['max']:9.2f}{share:>8}"
        )


def write_outputs(out_dir: Path, all_rows, summaries, startup, meta):
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scenario", "query", "num_results", "num_candidates_after_filter", "preprocessing_steps"]
    fieldnames += TOP_LEVEL_STAGES + SUB_STAGES + ["other", "total"]
    with open(out_dir / "timings_raw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "stage", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"])
        for s in summaries:
            for stage, v in s["stages"].items():
                w.writerow([s["scenario"], stage] + [round(v[k], 3) for k in ("mean", "p50", "p95", "p99", "max")])

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "startup_seconds": startup, "scenarios": summaries}, f, indent=2, ensure_ascii=False)

    print(f"\nErgebnisse geschrieben nach: {out_dir}")


# =====================================================
# MAIN
# =====================================================

def parse_args():
    p = argparse.ArgumentParser(description="Laufzeittest der Suchpipeline (lokal).")
    p.add_argument("--corpus", default=str(svc.CORPUS_PATH))
    p.add_argument("--queries", default=str(svc.QUERIES_PATH))
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help=f"Kommagetrennt, verfügbar: {', '.join(SCENARIOS)}",
    )
    p.add_argument("--device", default=None, help="cuda / cpu (Default: automatisch)")
    p.add_argument("--embedding-cache", default=".cache/benchmark/dense_embeddings_bge-m3.npz")
    p.add_argument("--skip-ce", action="store_true", help="Cross-Encoder nicht laden, CE-Szenarien überspringen")
    p.add_argument(
        "--keep-caches",
        action="store_true",
        help="Query-Embedding-/CE-Caches nach dem Warm-up NICHT leeren (misst Cache-Hit-Verhalten)",
    )
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    requested = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unbekannte Szenarien: {unknown}")

    startup: dict[str, float] = {}
    t_start = time.perf_counter()
    pc = build_components(args, startup)
    startup["total"] = round(time.perf_counter() - t_start, 3)
    svc.PC = pc

    timer = StageTimer()
    instrument(pc, timer)

    warmup_q, queries = load_queries(Path(args.queries), args.n_queries, args.warmup, args.seed)

    all_rows, summaries = [], []
    for name in requested:
        params = SCENARIOS[name]
        if params.get("base") == "xgb" and pc.xgb_retriever is None:
            print(f"\n[skip] {name}: kein XGB-Modell unter {svc.XGB_MODEL_PATH} (make train-xgb)")
            continue
        if params.get("use_ce") and pc.ce_reranker is None:
            print(f"\n[skip] {name}: Cross-Encoder deaktiviert (--skip-ce)")
            continue
        print(f"\n[run] {name} ...", flush=True)
        rows, peak_mb = run_scenario(name, params, pc, timer, queries, warmup_q, args)
        summary = summarize(rows, name, peak_mb)
        print_summary(summary)
        all_rows.extend(rows)
        summaries.append(summary)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "device": pc.dense.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "corpus_docs": len(pc.corpus),
        "n_queries": len(queries),
        "warmup": args.warmup,
        "seed": args.seed,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "keep_caches": args.keep_caches,
        "mode": "in-process (rest_service.search), ohne HTTP",
    }
    out_dir = Path(args.output_dir or f"results/benchmark/{datetime.now():%Y%m%d_%H%M%S}")
    write_outputs(out_dir, all_rows, summaries, startup, meta)

    print("\nStartup-Zeiten [s]:")
    for k, v in startup.items():
        print(f"  {k:<24}{v:8.2f}")


if __name__ == "__main__":
    main()
