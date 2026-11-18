"""
REST Service für die Interamt-Retrieval-Pipeline.

Startet einen FastAPI-Server, der die bestehende ComponentPipeline
(BM25 / Dense / RRF / Alpha / XGB + optional QE / CE) über einen
/search Endpoint verfügbar macht.

Zusätzlich unterstützt der Endpoint ein Filter-Objekt, das VOR dem
eigentlichen Retrieval angewendet wird (Pre-Filter): die Kandidatenmenge
wird anhand von PLZ/Ort + Radius, Dienstverhältnis, Bereich,
Beschäftigungsdauer, Dienstort und Teilzeit/Vollzeit eingeschränkt,
bevor BM25/Dense/etc. überhaupt scoren.

NEU: /admin/reload Endpoint, um den Korpus (und alle davon abgeleiteten
Komponenten) zur Laufzeit zu aktualisieren, ohne den Prozess neu zu
starten. Die neue Pipeline wird komplett in einem separaten Objekt
aufgebaut ("Double Buffering") und erst danach atomar eingehängt, damit
laufende /search-Requests währenddessen nie einen halb-aktualisierten
Zustand sehen. DenseRetriever nutzt dabei intern einen inkrementellen
Embedding-Cache (nur neue/geänderte Dokumente werden neu encodiert).

Start:
    uvicorn rest_service:app --reload --port 8000

Beispiel-Request:
    POST /search
    {
        "query": "Sachbearbeiter IT-Sicherheit",
        "top_k": 10,
        "base": "bm25",
        "use_qe": true,
        "use_ce": true,
        "filters": {
            "plz_oder_ort": "18055",
            "max_distance_km": 50,
            "beschaeftigungsart": "Vollzeit",
            "dienstverhaeltnis": ["Beamter", "Angestellter"]
        }
    }

Beispiel-Request (Reload nach Korpus-Update):
    POST /admin/reload
    -> liest corpus.jsonl neu ein, encodiert nur geänderte/neue Docs,
    baut BM25/FAISS/CE/Alpha/XGB neu auf und tauscht den globalen
       State atomar aus.

NEU: /evaluate und /evaluate/grid Endpoints. Beide nutzen die bereits
beim Startup/Reload initialisierten Komponenten (BM25/Dense/CE/QE)
wieder -- es wird NICHTS davon neu aufgebaut, nur evaluate_model() auf
Basis von queries.jsonl + qrels/test.tsv (aus demselben DATASET_DIR)
ausgeführt. Das entspricht fachlich run_pipeline_grid.py, nur eben ohne
den teuren Re-Init aller Retriever bei jedem Aufruf.

Beispiel-Request (einzelne Kombination):
    POST /evaluate
    {
        "base": "rrf",
        "use_qe": true,
        "use_ce": true,
        "k": 10,
        "candidate_k": 50
    }

Beispiel-Request (mehrere Kombinationen, wie im Grid-Skript):
    POST /evaluate/grid
    {
        "bases": ["bm25", "rrf", "alpha"],
        "k": 10
    }
"""

import itertools
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pgeocode
import psycopg
from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from datetime import datetime

from src.retrieval.bm25_retrieval import BM25Retriever
from src.retrieval.dense_retrieval import DenseRetriever
from src.retrieval.hybrid_fusion import HybridFusionRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.query_embedding_cache import QueryEmbeddingCache
from src.retrieval.xgb_fusion_retriever import XGBFusionRetriever
from src.retrieval.spell_correction import SpellCorrector
from src.retrieval.abbreviation_expansion import AbbreviationExpander
from src.retrieval.spelling_variants import SpellingVariantNormalizer
from src.retrieval.behoerden_normalizer import BehoerdenNormalizer
from src.retrieval.compound_splitter import CompoundSplitter

from src.retrieval.pipeline import (
    BM25Scorer,
    DenseScorer,
    RRFScorer,
    AlphaFusionScorer,
    XGBScorer,
    ComponentPipeline,
)
from src.rerank.xgb_fusion import XGBFusionModel
from src.api.interamt_data import fetch_and_store_corpus

logger = logging.getLogger("rest_service")
logging.basicConfig(level=logging.INFO)


# =====================================================
# KONFIGURATION
# =====================================================

DATASET_DIR = Path("data/processed/jobs_interamt_log_queries")
CORPUS_PATH = Path(os.getenv("INTERAMT_CORPUS_PATH", "runtime-data/corpus.jsonl"))
QUERIES_PATH = DATASET_DIR / "queries.jsonl"
QRELS_PATH = DATASET_DIR / "qrels" / "test.tsv"
XGB_MODEL_PATH = Path("models/xgb_fusion.json")
EMBEDDING_DATABASE_URL = os.getenv(
    "EMBEDDING_DATABASE_URL",
    "postgresql://ai_suche:ai_suche@localhost:5432/ai_suche",
)
AI_SUCH_ROLE = os.getenv("AI_SUCH_ROLE", "master").strip().lower()
if AI_SUCH_ROLE not in {"master", "worker"}:
    raise RuntimeError("AI_SUCH_ROLE muss 'master' oder 'worker' sein.")
READ_ONLY_WORKER = AI_SUCH_ROLE == "worker"
ABBREVIATION_CONFIG_PATH = Path("config/abbreviations.json")
VARIANT_CONFIG_PATH = Path("config/spelling_variants.json")
BEHOERDEN_CONFIG_PATH = Path("config/behoerdensprache.json")
COMPOUND_DICT_PATH = Path("config/compound_dictionary.json")

# Seit der Erweiterung von transform_dtag_to_beir.py schreibt corpus.jsonl
# diese Felder als echte JSON-Keys (zusätzlich zum unveränderten "text").
# Für ältere/nicht neu generierte corpus.jsonl-Dateien (nur "_id"/"title"/
# "text") greift automatisch der Fallback über parse_preamble() weiter unten.
FIELD_DIENSTVERHAELTNIS = "dienstverhaeltnis"  # Liste von Labels, z.B. ["Beamter"], ["Angestellter"]
FIELD_BEREICH = "bereich"                      # Liste, aus aufgabenbereich[]
FIELD_BESCHAEFTIGUNGSDAUER = "beschaeftigungsdauer"
FIELD_BESCHAEFTIGUNGSART = "beschaeftigungsart"  # "Vollzeit" / "Teilzeit" / "Vollzeit oder Teilzeit"
FIELD_DIENSTORT = "dienstort"    # primärer Ortsname (erster Einsatzort)
FIELD_BUNDESLAND = "bundesland"
FIELD_EINSATZORTE = "einsatzorte"  # Liste von {ort, bundesland, plz, latitude, longitude}
FIELD_DIENSTSTELLE = "dienststelle"
FIELD_URL = "url"
FIELD_KENNUNG = "kennung"
FIELD_VEROEFFENTLICHT_AM = "veroeffentlicht_am"
FIELD_BEWERBUNGSFRIST = "bewerbungsfrist"
FIELD_BEWERBUNGSPROZESS = "bewerbungsprozess"
FIELD_ANZAHL_POSTEN = "anzahl_posten"
FIELD_TEXT = "text"

SNIPPET_MAX_CHARS = 280

# Labels der strukturierten Präambel am Anfang jedes "text"-Feldes -- nur
# noch als Fallback relevant, falls corpus.jsonl (noch) keine eigenen
# JSON-Felder für obige FIELD_*-Konstanten hat.
PREAMBLE_LABELS = ["Stelle", "Behörde", "Fachrichtung", "Aufgabenbereich", "Einsatzort", "Beschäftigung"]
_PREAMBLE_LABEL_PATTERN = re.compile(r"\b(" + "|".join(PREAMBLE_LABELS) + r"):\s*([^.]*)\.")
# Nach dieser Zeichen-Position im Text brechen wir das Präambel-Parsing ab,
# um kein "Behörde:"/"Einsatzort:" aus dem freien Fließtext aufzuschnappen.
_PREAMBLE_SEARCH_LIMIT_CHARS = 500




# =====================================================
# DATEN-MODELLE (Request / Response)
# =====================================================

class SearchFilters(BaseModel):
    plz_oder_ort: Optional[str] = Field(
        default=None, description="Referenzpunkt für die Entfernungsberechnung, z.B. '18055' oder 'Rostock'"
    )
    max_distance_km: Optional[float] = Field(
        default=None, description="Maximale Luftlinien-Entfernung in km (nur wirksam zusammen mit plz_oder_ort)"
    )
    dienstverhaeltnis: Optional[List[str]] = None
    bereich: Optional[List[str]] = None
    beschaeftigungsdauer: Optional[List[str]] = None
    dienstort: Optional[List[str]] = None
    beschaeftigungsart: Optional[Literal["Teilzeit", "Vollzeit"]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    candidate_k: int = 50
    filters: Optional[SearchFilters] = None
    base: Literal["bm25", "dense", "rrf", "alpha", "xgb"] = Field(
        default="bm25",
        description=(
            "Retrieval-Basis: 'bm25' = Stichwortsuche mit BM25; "
            "'dense' = semantische Vektorsuche; "
            "'rrf' = Kombination aus BM25 und Dense mit Reciprocal Rank Fusion; "
            "'alpha' = gewichtete Kombination aus BM25 und Dense; "
            "'xgb' = XGBoost-Learning-to-Rank-Fusion."
        ),
    )
    use_qe: bool = False
    use_kg: bool = Field(
        default=False,
        description="Veraltet: Knowledge-Graph-Reranking ist in der Produktivkonfiguration deaktiviert.",
    )
    use_ce: bool = False
    normalize_scores: bool = False
    alpha: float = 0.2
    sort_by: Literal["relevance", "recency", "distance"] = Field(
        default="relevance",
        description=(
            "Sortierung der Ergebnisse: 'relevance' (Pipeline-Score, Default), "
            "'recency' (neueste Veröffentlichung zuerst) oder 'distance' "
            "(nächstgelegener Einsatzort zuerst — erfordert filters.plz_oder_ort). "
            "Dokumente ohne Sortierwert (kein Datum / keine auflösbare Distanz) "
            "landen am Ende statt den Request scheitern zu lassen."
        ),
    )
    correct_spelling: bool = Field(
        default=True, description="Rechtschreibfehler in der Query vor dem Retrieval automatisch korrigieren"
    )
    expand_abbreviations: bool = Field(
        default=True,
        description="Bekannte Abkürzungen (TVöD, m/w/d, SB, IT, ...) erkennen und ausgeschriebene Form ergänzen",
    )
    normalize_spelling_variants: bool = Field(
        default=True,
        description="Schreibvarianten (Gender-Formen, ß/ss, Homeoffice/Home-Office, ...) normalisieren",
    )
    normalize_behoerdensprache: bool = Field(
        default=True,
        description="Amtsdeutsch-Begriffe (Tätigkeitsmerkmale, Sachbearbeitung, ...) um Klartext-Synonyme ergänzen und umgekehrt",
    )
    split_compounds: bool = True
    include_document_details: bool = Field(
        default=True,
        description="Zusätzliche Dokument-Metadaten (Dienststelle, Bereich, Dienstort, Bewerbungsfrist, Snippet, ...) in den Ergebnissen mitliefern",
    )


class DocumentDetails(BaseModel):
    """
    Zusätzliche Metadaten zu einem Treffer. Kommen primär aus den echten
    JSON-Feldern in corpus.jsonl (seit der Erweiterung von
    transform_dtag_to_beir.py); für ältere corpus.jsonl-Dateien ohne diese
    Felder greift ein Fallback über die Text-Präambel (siehe parse_preamble).
    """
    dienstverhaeltnis: Optional[List[str]] = None
    bereich: Optional[List[str]] = None
    fachrichtung: Optional[str] = None
    beschaeftigungsart: Optional[str] = None
    beschaeftigungsdauer: Optional[Any] = None
    dienstort: Optional[str] = None
    bundesland: Optional[str] = None
    einsatzorte: Optional[List[str]] = None
    dienststelle: Optional[str] = None
    url: Optional[str] = None
    kennung: Optional[str] = None
    veroeffentlicht_am: Optional[str] = None
    bewerbungsfrist: Optional[str] = None
    bewerbungsprozess: Optional[str] = None
    anzahl_posten: Optional[int] = None
    snippet: Optional[str] = None


class SearchResultItem(BaseModel):
    doc_id: str
    title: Optional[str] = None
    score: float
    distance_km: Optional[float] = None
    details: Optional[DocumentDetails] = None


class QueryPreprocessingStep(BaseModel):
    step: str
    result: str


class SearchResponse(BaseModel):
    query: str
    base: str
    num_candidates_after_filter: Optional[int] = None
    results: List[SearchResultItem]
    final_query: str
    preprocessing_applied: List[QueryPreprocessingStep] = Field(default_factory=list)


class ReloadResponse(BaseModel):
    status: str
    num_documents: int
    duration_seconds: float


class ServiceStatusResponse(BaseModel):
    status: str
    pipeline_ready: bool
    database: Dict[str, Any]
    num_documents: int
    components: Dict[str, bool]


class EvaluateRequest(BaseModel):
    base: Literal["bm25", "dense", "rrf", "alpha", "xgb"] = Field(
        default="bm25",
        description=(
            "Zu evaluierende Retrieval-Basis: 'bm25' (Stichwortsuche), "
            "'dense' (semantische Vektorsuche), 'rrf' (BM25 + Dense via "
            "Reciprocal Rank Fusion), 'alpha' (gewichtete BM25/Dense-Fusion) "
            "oder 'xgb' (XGBoost-Learning-to-Rank-Fusion)."
        ),
    )
    use_qe: bool = False
    use_kg: bool = Field(
        default=False,
        description="Veraltet: Knowledge-Graph-Reranking ist in der Produktivkonfiguration deaktiviert.",
    )
    use_ce: bool = False
    k: int = 10
    candidate_k: int = 50
    normalize_scores: bool = False


class EvaluateResponse(BaseModel):
    model: str
    num_queries: int
    metrics: Dict[str, float]


class EvaluateGridRequest(BaseModel):
    bases: List[Literal["bm25", "dense", "rrf", "alpha", "xgb"]] = Field(
        default_factory=lambda: ["bm25", "dense", "rrf", "alpha"],
        description=(
            "Retrieval-Basen, die im Grid verglichen werden: 'bm25' (Stichwortsuche), "
            "'dense' (semantische Vektorsuche), 'rrf' (BM25 + Dense via Reciprocal "
            "Rank Fusion), 'alpha' (gewichtete BM25/Dense-Fusion) und 'xgb' "
            "(XGBoost-Learning-to-Rank-Fusion)."
        ),
    )
    skip_qe: bool = False
    skip_kg: bool = False
    skip_ce: bool = False
    k: int = 10
    candidate_k: int = 50
    normalize_scores: bool = False


class EvaluateGridResultItem(BaseModel):
    model: str
    metrics: Dict[str, float]


class EvaluateGridResponse(BaseModel):
    num_queries: int
    num_combinations: int
    results: List[EvaluateGridResultItem]


# =====================================================
# GEO-HILFSFUNKTIONEN (PLZ/Ort -> Koordinaten, Distanz)
# =====================================================

_nomi = pgeocode.Nominatim("de")


def resolve_to_latlon(value: str) -> Optional[tuple]:
    """Löst eine 5-stellige PLZ oder einen Ortsnamen zu (lat, lon) auf."""
    value = value.strip()
    if not value:
        return None

    if value.isdigit() and len(value) == 5:
        rec = _nomi.query_postal_code(value)
        if rec is None or np.isnan(rec.latitude):
            return None
        return float(rec.latitude), float(rec.longitude)

    df = _nomi._data
    match = df[df["place_name"].str.lower() == value.lower()]
    if match.empty:
        return None
    rec = match.iloc[0]
    if np.isnan(rec.latitude):
        return None
    return float(rec.latitude), float(rec.longitude)


def haversine_km(lat1: float, lon1: float, lat2_arr: np.ndarray, lon2_arr: np.ndarray) -> np.ndarray:
    """Vektorisierte Haversine-Distanz in km."""
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2_arr), np.radians(lon2_arr)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

_VEROEFFENTLICHT_AM_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S")


def _parse_veroeffentlicht_am(raw: Optional[str]) -> Optional[datetime]:
    """Robustes Parsing für veroeffentlicht_am -- Format aus der Interamt-API ist nicht 100% garantiert."""
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    for fmt in _VEROEFFENTLICHT_AM_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _compute_min_distances_km(
    ref_lat: float,
    ref_lon: float,
    doc_lats: np.ndarray,
    doc_lons: np.ndarray,
    doc_ids_with_geo: List[str],
) -> Dict[str, float]:
    """
    Minimale Distanz je doc_id zum Referenzpunkt. Ein Job kann mehrere
    Einsatzorte (= mehrere Einträge im Geo-Cache) haben -- hier zählt
    jeweils der nächstgelegene.
    """
    if len(doc_ids_with_geo) == 0:
        return {}
    dists = haversine_km(ref_lat, ref_lon, doc_lats, doc_lons)
    min_dist: Dict[str, float] = {}
    for doc_id, d in zip(doc_ids_with_geo, dists):
        d = float(d)
        if doc_id not in min_dist or d < min_dist[doc_id]:
            min_dist[doc_id] = d
    return min_dist


def _sort_results(results: List[SearchResultItem], sort_by: str, corpus: dict) -> List[SearchResultItem]:
    """
    Sortiert die (bereits nach Pipeline-Score geordneten) Ergebnisse ggf. um.
    'relevance' lässt die Reihenfolge unverändert -- Score-Ranking bleibt
    Aufgabe der Pipeline. Dokumente ohne den jeweiligen Sortierwert
    rutschen ans Ende, statt den Request scheitern zu lassen.
    """
    if sort_by == "relevance":
        return results

    if sort_by == "recency":
        def _key(item: SearchResultItem):
            dt = _parse_veroeffentlicht_am(corpus.get(item.doc_id, {}).get(FIELD_VEROEFFENTLICHT_AM))
            # (kein Datum -> ans Ende), sonst neuestes zuerst
            return (dt is None, -dt.timestamp() if dt else 0.0)
        return sorted(results, key=_key)

    if sort_by == "distance":
        def _key(item: SearchResultItem):
            # (keine Distanz -> ans Ende), sonst nächstgelegen zuerst
            return (item.distance_km is None, item.distance_km if item.distance_km is not None else float("inf"))
        return sorted(results, key=_key)

    raise HTTPException(status_code=400, detail=f"Unbekannter sort_by-Wert: {sort_by}")


# =====================================================
# FILTER-LOGIK
# =====================================================

def _any_in(doc_value: Any, allowed_values: List[str]) -> bool:
    """Prüft ob doc_value (String oder Liste) case-insensitiv mit einem der erlaubten Werte übereinstimmt."""
    if doc_value is None:
        return False
    allowed_lower = {v.lower() for v in allowed_values}
    if isinstance(doc_value, list):
        return any(str(v).lower() in allowed_lower for v in doc_value)
    return str(doc_value).lower() in allowed_lower


def _arbeitszeit_matches(doc_value: Any, wanted: str) -> bool:
    """
    "Beschäftigung" kann im Korpus "Vollzeit oder Teilzeit" lauten (beide
    möglich). Ein Filter auf "Vollzeit" soll solche Jobs trotzdem finden,
    daher hier Teilstring- statt Exact-Match (case-insensitiv).
    """
    if not doc_value:
        return False
    return wanted.lower() in str(doc_value).lower()


def get_allowed_doc_ids(
    corpus: dict,
    filters: Optional[SearchFilters],
    doc_lats: np.ndarray,
    doc_lons: np.ndarray,
    doc_ids_with_geo: List[str],
) -> Optional[set]:
    """
    Berechnet die Menge erlaubter doc_ids basierend auf den Filterkriterien.
    Rückgabe None bedeutet: kein Filter aktiv, kompletter Korpus erlaubt.
    """
    if filters is None:
        return None

    has_categorical = any(
        [
            filters.dienstverhaeltnis,
            filters.bereich,
            filters.beschaeftigungsdauer,
            filters.dienstort,
            filters.beschaeftigungsart,
        ]
    )
    has_geo = filters.plz_oder_ort and filters.max_distance_km is not None

    if not has_categorical and not has_geo:
        return None

    allowed = set(corpus.keys())

    if has_categorical:
        for doc_id, doc in corpus.items():
            if doc_id not in allowed:
                continue
            if filters.dienstverhaeltnis and not _any_in(doc.get(FIELD_DIENSTVERHAELTNIS), filters.dienstverhaeltnis):
                allowed.discard(doc_id)
                continue
            if filters.bereich and not _any_in(doc.get(FIELD_BEREICH), filters.bereich):
                allowed.discard(doc_id)
                continue
            if filters.beschaeftigungsdauer and not _any_in(
                doc.get(FIELD_BESCHAEFTIGUNGSDAUER), filters.beschaeftigungsdauer
            ):
                allowed.discard(doc_id)
                continue
            if filters.dienstort and not _any_in(doc.get(FIELD_DIENSTORT), filters.dienstort):
                allowed.discard(doc_id)
                continue
            if filters.beschaeftigungsart and not _arbeitszeit_matches(
                doc.get(FIELD_BESCHAEFTIGUNGSART), filters.beschaeftigungsart
            ):
                allowed.discard(doc_id)
                continue

    if has_geo:
        ref = resolve_to_latlon(filters.plz_oder_ort)
        if ref is None:
            logger.warning("PLZ/Ort '%s' konnte nicht aufgelöst werden.", filters.plz_oder_ort)
            return set()  # konservativ: kein Treffer statt stillem Ignorieren

        ref_lat, ref_lon = ref
        dists = haversine_km(ref_lat, ref_lon, doc_lats, doc_lons)
        within_radius = {
            doc_ids_with_geo[i] for i, d in enumerate(dists) if d <= filters.max_distance_km
        }
        allowed &= within_radius

    return allowed


# =====================================================
# GLOBALER STATE (einmal beim Start geladen, teure Objekte)
# =====================================================

class PipelineComponents:
    corpus: dict = {}
    queries: dict = {}
    qrels: dict = {}
    bm25: Optional[BM25Retriever] = None
    dense: Optional[DenseRetriever] = None
    alpha_retriever: Optional[HybridFusionRetriever] = None
    xgb_retriever: Optional[XGBFusionRetriever] = None
    query_expander: Optional[QueryExpander] = None
    ce_reranker: Optional[CrossEncoderReranker] = None
    spell_corrector: Optional[SpellCorrector] = None
    abbreviation_expander: Optional[AbbreviationExpander] = None
    variant_normalizer: Optional[SpellingVariantNormalizer] = None
    behoerden_normalizer: Optional[BehoerdenNormalizer] = None
    compound_splitter: Optional[CompoundSplitter] = None
    doc_lats: np.ndarray = np.array([])
    doc_lons: np.ndarray = np.array([])
    doc_ids_with_geo: List[str] = []


PC = PipelineComponents()

# Schützt build_pipeline_components() davor, dass zwei Reloads
# gleichzeitig laufen (z.B. wenn versehentlich zweimal getriggert wird).
_reload_lock = threading.Lock()


def load_jsonl(path: Path) -> dict:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["_id"]] = item
    return data


def load_qrels(path: Path) -> dict:
    """Erwartet TSV mit Header-Zeile: query-id\tcorpus-id\tscore (BEIR-Format)."""
    qrels: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # Header überspringen
        for line in f:
            qid, docid, score = line.strip().split("\t")
            qrels.setdefault(qid, {})[docid] = float(score)
    return qrels


def build_doc_geo_cache(corpus: dict):
    """
    Löst pro Dokument den/die Einsatzort(e) zu Koordinaten auf (einmalig,
    gecacht). Bevorzugt die bereits in "einsatzorte" mitgelieferten echten
    Koordinaten (aus der Interamt-API, präzise); nur falls die fehlen, wird
    per Ortsname (pgeocode) nachgeocoded -- als Fallback für ältere
    corpus.jsonl-Dateien ohne "einsatzorte" bzw. für Orte ohne Koordinaten.
    Ein Job kann so mehrere Einträge (mehrere Einsatzorte) im Cache haben.
    """
    place_cache: dict = {}
    lats, lons, ids = [], [], []

    for doc_id, doc in corpus.items():
        einsatzorte = doc.get(FIELD_EINSATZORTE)
        used_precomputed = False
        if einsatzorte:
            for entry in einsatzorte:
                if not isinstance(entry, dict):
                    continue
                lat, lon = entry.get("latitude"), entry.get("longitude")
                if lat is None or lon is None:
                    continue
                try:
                    lats.append(float(lat))
                    lons.append(float(lon))
                    ids.append(doc_id)
                    used_precomputed = True
                except (TypeError, ValueError):
                    continue
        if used_precomputed:
            continue

        # Fallback: Ortsname auflösen (altes Schema ohne "einsatzorte" bzw.
        # ohne Koordinaten darin).
        ort = doc.get(FIELD_DIENSTORT)
        if isinstance(ort, list):
            ort = ort[0] if ort else None
        if not ort:
            ort = parse_preamble(doc.get(FIELD_TEXT)).get("Einsatzort")
        if not ort:
            continue

        if ort not in place_cache:
            place_cache[ort] = resolve_to_latlon(ort)

        latlon = place_cache[ort]
        if latlon is not None:
            lats.append(latlon[0])
            lons.append(latlon[1])
            ids.append(doc_id)

    return np.array(lats), np.array(lons), ids


def build_pipeline_components() -> PipelineComponents:
    """
    Baut eine komplette, frische PipelineComponents-Instanz auf Basis
    der aktuellen corpus.jsonl auf. Wird sowohl beim Startup als auch
    beim /admin/reload verwendet.

    Läuft bewusst auf einem NEUEN Objekt (nicht auf dem globalen PC),
    damit der bisherige globale State während des kompletten Builds
    weiter Requests bedienen kann (Double-Buffering). Der Aufrufer
    tauscht das Ergebnis erst danach atomar in die globale Variable PC.
    """
    # Vor dem teuren Korpus-Download und Encoding abbrechen, falls die
    # Embedding-Datenbank nicht erreichbar ist.
    database = _database_status()
    if database["status"] != "ok":
        raise RuntimeError(f"Embedding-Datenbank nicht erreichbar: {database['error']}")

    new_pc = PipelineComponents()

    if READ_ONLY_WORKER:
        if not CORPUS_PATH.exists():
            raise RuntimeError(
                f"Worker-Korpus nicht gefunden: {CORPUS_PATH}. "
                "Der Master muss den Korpus zuerst bereitstellen."
            )
        logger.info("Worker-Modus: verwende vorhandenen Korpus %s.", CORPUS_PATH)
    else:
        logger.info("Master-Modus: lade aktuellen Korpus von der Interamt-API ...")
        fetch_and_store_corpus(CORPUS_PATH)
    new_pc.corpus = load_jsonl(CORPUS_PATH)

    logger.info("Initialisiere Rechtschreibkorrektur ...")
    new_pc.spell_corrector = SpellCorrector(new_pc.corpus)

    logger.info("Initialisiere Abkürzungserkennung ...")
    new_pc.abbreviation_expander = AbbreviationExpander(
    custom_path=ABBREVIATION_CONFIG_PATH if ABBREVIATION_CONFIG_PATH.exists() else None
    )

    logger.info("Initialisiere Behördensprache-Normalisierung ...")
    new_pc.behoerden_normalizer = BehoerdenNormalizer(
        custom_path=BEHOERDEN_CONFIG_PATH if BEHOERDEN_CONFIG_PATH.exists() else None
    )

    logger.info("Initialisiere Schreibweisen-Normalisierung ...")
    new_pc.variant_normalizer = SpellingVariantNormalizer(
        custom_path=VARIANT_CONFIG_PATH if VARIANT_CONFIG_PATH.exists() else None
    )

    logger.info("Initialisiere Komposita-Splitter ...")
    new_pc.compound_splitter = CompoundSplitter(
        new_pc.corpus,
        custom_path=COMPOUND_DICT_PATH if COMPOUND_DICT_PATH.exists() else None,
    )

    # Queries/Qrels sind nur für die /evaluate-Endpoints relevant und
    # rein optional -- falls die Dateien (noch) nicht existieren, bleibt
    # der Live-Service trotzdem voll funktionsfähig, nur /evaluate liefert
    # dann einen 503.
    if QUERIES_PATH.exists() and QRELS_PATH.exists():
        logger.info("Lade Evaluations-Queries/Qrels aus %s ...", DATASET_DIR)
        new_pc.queries = load_jsonl(QUERIES_PATH)
        new_pc.qrels = load_qrels(QRELS_PATH)
    else:
        logger.warning(
            "queries.jsonl oder qrels/test.tsv nicht gefunden unter %s — "
            "/evaluate-Endpoints bleiben deaktiviert.",
            DATASET_DIR,
        )
        new_pc.queries = {}
        new_pc.qrels = {}

    logger.info("Initialisiere BM25 ...")
    new_pc.bm25 = BM25Retriever(new_pc.corpus)

    logger.info("Initialisiere Dense Retriever (inkrementeller Embedding-Cache) ...")
    # DenseRetriever muss intern per Content-Hash erkennen, welche Docs
    # neu/geändert/entfernt sind, und nur diese neu encodieren.
    new_pc.dense = DenseRetriever(
        new_pc.corpus,
        cache_path=EMBEDDING_DATABASE_URL,
        read_only=READ_ONLY_WORKER,
    )

    # Dedupliziert Query-Embeddings für den Dense-Pfad innerhalb einer
    # PipelineComponents-Instanz.
    query_cache = QueryEmbeddingCache(new_pc.dense.model)
    new_pc.dense.query_cache = query_cache

    logger.info("Initialisiere Query Expansion ...")
    new_pc.query_expander = QueryExpander(
        bm25=new_pc.bm25, corpus=new_pc.corpus, doc_ids=new_pc.bm25.doc_ids
    )

    logger.info("Initialisiere Cross-Encoder Reranker ...")
    new_pc.ce_reranker = CrossEncoderReranker()

    logger.info("Initialisiere Alpha-Fusion Retriever ...")
    new_pc.alpha_retriever = HybridFusionRetriever(
        bm25_retriever=new_pc.bm25, dense_retriever=new_pc.dense, alpha=0.2
    )

    # XGB-Retriever: nur aktivieren, wenn ein trainiertes Modell vorliegt
    # (Training passiert offline, z.B. via scripts/train_xgb_model.py).
    # Ein Korpus-Reload lädt bewusst NICHT automatisch ein neues Modell
    # neu, sondern nur die aktuell auf Disk liegende Datei -- falls das
    # XGB-Modell separat neu trainiert wurde, liegt es bereits unter
    # XGB_MODEL_PATH und wird hier mit aufgenommen.
    if XGB_MODEL_PATH.exists():
        logger.info("Lade trainiertes XGB-Modell aus %s ...", XGB_MODEL_PATH)
        trained_model = XGBFusionModel()
        trained_model.load(XGB_MODEL_PATH)
        new_pc.xgb_retriever = XGBFusionRetriever(
            bm25=new_pc.bm25, dense=new_pc.dense, model=trained_model, corpus=new_pc.corpus
        )
    else:
        logger.warning(
            "Kein trainiertes XGB-Modell unter %s gefunden — base='xgb' bleibt deaktiviert.",
            XGB_MODEL_PATH,
        )
        new_pc.xgb_retriever = None

    logger.info("Baue Geo-Cache für Distanzfilter ...")
    new_pc.doc_lats, new_pc.doc_lons, new_pc.doc_ids_with_geo = build_doc_geo_cache(new_pc.corpus)

    logger.info("Pipeline-Build abgeschlossen. %d Dokumente geladen.", len(new_pc.corpus))
    return new_pc


# =====================================================
# HILFSFUNKTIONEN FÜR SUCHERGEBNISSE
# =====================================================

def _make_snippet(text: Optional[str], max_chars: int = SNIPPET_MAX_CHARS) -> Optional[str]:
    """Kürzt einen Text auf eine lesbare Länge (Wortgrenze, kein harter Cut)."""
    if not text or not isinstance(text, str):
        return None
    text = " ".join(text.split())  # Whitespace/Zeilenumbrüche normalisieren
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + "…"


def parse_preamble(text: Optional[str]) -> Dict[str, Any]:
    """
    Die "text"-Felder im Korpus beginnen mit einer strukturierten Präambel
    aus Label:Wert-Sätzen, z.B.
        "Stelle: ... . Behörde: Stadt Rheine. Aufgabenbereich: allgemeine
         verwaltung, it und telekommunikation. Einsatzort: Rheine.
         Beschäftigung: Vollzeit. <freier Fließtext ...>"
    corpus.jsonl hat dafür KEINE eigenen JSON-Felder -- wir parsen die
    Präambel deshalb hier heraus. Gibt ein Dict mit den gefundenen Labels
    (als Rohstrings) plus "_preamble_end" (Zeichenindex, ab dem der freie
    Fließtext beginnt) zurück.
    """
    result: Dict[str, Any] = {"_preamble_end": 0}
    if not text:
        return result

    for match in _PREAMBLE_LABEL_PATTERN.finditer(text):
        if match.start() > _PREAMBLE_SEARCH_LIMIT_CHARS:
            break
        label, value = match.group(1), match.group(2).strip()
        if label not in result:  # erstes Vorkommen zählt -- das ist die Präambel
            result[label] = value
        result["_preamble_end"] = max(result["_preamble_end"], match.end())

    return result


def _format_einsatzorte(einsatzorte: Optional[List[dict]]) -> Optional[List[str]]:
    """Wandelt die strukturierte einsatzorte-Liste in lesbare 'Ort (Bundesland)'-Strings für die Anzeige um."""
    if not einsatzorte:
        return None
    formatted = []
    for entry in einsatzorte:
        if not isinstance(entry, dict):
            continue
        ort = entry.get("ort")
        bundesland = entry.get("bundesland")
        if ort and bundesland:
            formatted.append(f"{ort} ({bundesland})")
        elif ort:
            formatted.append(ort)
    return formatted or None


def build_document_details(doc: dict) -> Optional[DocumentDetails]:
    """
    Baut das DocumentDetails-Objekt für ein Korpusdokument. Echte
    JSON-Felder (seit der Erweiterung von transform_dtag_to_beir.py) haben
    Vorrang; ansonsten wird aus der Text-Präambel geparst (siehe
    parse_preamble, für ältere corpus.jsonl-Dateien). Liefert None, wenn
    gar nichts gefunden wurde.
    """
    text = doc.get(FIELD_TEXT)
    preamble = parse_preamble(text)
    preamble_end = preamble.get("_preamble_end", 0)

    bereich = doc.get(FIELD_BEREICH)
    if bereich is None:
        aufgabenbereich = preamble.get("Aufgabenbereich")
        bereich = [b.strip() for b in aufgabenbereich.split(",") if b.strip()] if aufgabenbereich else None

    body_text = text[preamble_end:] if text else None

    details = DocumentDetails(
        dienstverhaeltnis=doc.get(FIELD_DIENSTVERHAELTNIS),
        bereich=bereich,
        fachrichtung=doc.get("fachrichtung") or preamble.get("Fachrichtung"),
        beschaeftigungsart=doc.get(FIELD_BESCHAEFTIGUNGSART) or preamble.get("Beschäftigung"),
        beschaeftigungsdauer=doc.get(FIELD_BESCHAEFTIGUNGSDAUER),
        dienstort=doc.get(FIELD_DIENSTORT) or preamble.get("Einsatzort"),
        bundesland=doc.get(FIELD_BUNDESLAND),
        einsatzorte=_format_einsatzorte(doc.get(FIELD_EINSATZORTE)),
        dienststelle=doc.get(FIELD_DIENSTSTELLE) or preamble.get("Behörde"),
        url=doc.get(FIELD_URL),
        kennung=doc.get(FIELD_KENNUNG),
        veroeffentlicht_am=doc.get(FIELD_VEROEFFENTLICHT_AM),
        bewerbungsfrist=doc.get(FIELD_BEWERBUNGSFRIST),
        bewerbungsprozess=doc.get(FIELD_BEWERBUNGSPROZESS),
        anzahl_posten=doc.get(FIELD_ANZAHL_POSTEN),
        snippet=_make_snippet(body_text),
    )
    if not any(v is not None for v in details.model_dump().values()):
        return None
    return details


def build_result_item(
    doc_id: str,
    score: float,
    corpus: dict,
    include_details: bool,
    distance_km: Optional[float] = None,
) -> SearchResultItem:
    doc = corpus.get(doc_id, {})
    return SearchResultItem(
        doc_id=doc_id,
        title=doc.get("title"),
        score=float(score),
        distance_km=round(distance_km, 1) if distance_km is not None else None,
        details=build_document_details(doc) if include_details else None,
    )

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Interamt Retrieval REST Service",
    description=(
        "REST-API für die Suche in Interamt-Stellenausschreibungen. "
        "Die Beispiele im POST /search-Request können direkt über "
        "Swagger UI ausprobiert werden."
    ),
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "REST %s %s fehlgeschlagen nach %.3fs",
            request.method,
            request.url.path,
            time.monotonic() - start,
        )
        raise

    logger.info(
        "REST %s %s -> %d in %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        time.monotonic() - start,
    )
    return response


@app.on_event("startup")
def startup_event():
    global PC
    PC = build_pipeline_components()


def make_scorer(base: str):
    if base == "bm25":
        return BM25Scorer(PC.bm25)
    if base == "dense":
        return DenseScorer(PC.dense)
    if base == "rrf":
        return RRFScorer(PC.bm25, PC.dense)
    if base == "alpha":
        return AlphaFusionScorer(PC.alpha_retriever)
    if base == "xgb":
        if PC.xgb_retriever is None:
            raise HTTPException(
                status_code=400,
                detail="XGB-Retriever ist nicht geladen. Bitte zunächst ein trainiertes Modell bereitstellen.",
            )
        return XGBScorer(PC.xgb_retriever)
    raise HTTPException(status_code=400, detail=f"Unbekannte base: {base}")


def _database_status() -> Dict[str, Any]:
    try:
        with psycopg.connect(EMBEDDING_DATABASE_URL, connect_timeout=2) as connection:
            connection.execute("SELECT 1")
        return {"status": "ok"}
    except Exception as error:
        logger.warning("Datenbank-Healthcheck fehlgeschlagen: %s", error)
        return {"status": "error", "error": str(error)}


def _service_status() -> ServiceStatusResponse:
    components = {
        "bm25": PC.bm25 is not None,
        "dense": PC.dense is not None,
        "alpha": PC.alpha_retriever is not None,
        "xgb": PC.xgb_retriever is not None,
    }
    pipeline_ready = bool(PC.corpus) and all(
        components[name] for name in ("bm25", "dense", "alpha")
    )
    database = _database_status()
    ready = pipeline_ready and database["status"] == "ok"
    return ServiceStatusResponse(
        status="ok" if ready else "degraded",
        pipeline_ready=pipeline_ready,
        database=database,
        num_documents=len(PC.corpus),
        components=components,
    )


@app.get("/health", response_model=ServiceStatusResponse)
def health():
    status = _service_status()
    if status.status != "ok":
        raise HTTPException(status_code=503, detail=status.model_dump())
    return status


@app.get("/status", response_model=ServiceStatusResponse)
def service_status():
    return _service_status()


@app.post("/admin/reload", response_model=ReloadResponse)
def reload_pipeline():
    """
    Liest corpus.jsonl neu ein und baut die komplette Pipeline neu auf.
    DenseRetriever nutzt dabei intern seinen inkrementellen Cache, sodass
    nur neue/geänderte Dokumente tatsächlich neu encodiert werden.

    Läuft NICHT als Blocking-Operation innerhalb des globalen State: Die
    neue Pipeline wird komplett separat aufgebaut. Erst nach erfolgreichem
    Aufbau wird die globale Referenz PC atomar ausgetauscht, sodass
    laufende /search-Requests währenddessen unbeeinträchtigt weiterlaufen.

    Falls der Build fehlschlägt (z.B. defekte corpus.jsonl), bleibt der
    bisherige PC unverändert erhalten und ein 500er wird zurückgegeben.
    """
    if READ_ONLY_WORKER:
        raise HTTPException(
            status_code=403,
            detail="Pipeline-Reload ist nur im Master-Modus erlaubt.",
        )

    if not _reload_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Ein Reload läuft bereits. Bitte warten, bis dieser abgeschlossen ist.",
        )

    start = time.monotonic()
    try:
        try:
            new_pc = build_pipeline_components()
        except Exception as e:
            logger.exception("Pipeline-Reload fehlgeschlagen, bisheriger Zustand bleibt aktiv.")
            raise HTTPException(status_code=500, detail=f"Reload fehlgeschlagen: {e}")

        global PC
        PC = new_pc  # atomarer Referenz-Swap (GIL-geschützt)

        duration = time.monotonic() - start
        logger.info("Reload abgeschlossen in %.1fs (%d Dokumente).", duration, len(PC.corpus))

        return ReloadResponse(
            status="reloaded",
            num_documents=len(PC.corpus),
            duration_seconds=round(duration, 2),
        )
    finally:
        _reload_lock.release()


def _require_eval_data():
    if not PC.queries or not PC.qrels:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Keine Evaluationsdaten geladen (erwartet unter "
                f"{QUERIES_PATH} und {QRELS_PATH}). /evaluate ist erst "
                f"nutzbar, wenn beide Dateien vorhanden sind und die "
                f"Pipeline (neu) geladen wurde."
            ),
        )

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Stellenausschreibungen suchen",
    description=(
        "Führt eine Suche mit BM25, Dense Retrieval oder einer Fusion aus. "
        "Im Request-Body stehen mehrere direkt ausführbare Beispiele zur Verfügung."
    ),
)
def search(
    req: SearchRequest = Body(
        ...,
        openapi_examples={
            "einfache Suche": {
                "summary": "Einfache BM25-Suche",
                "description": "Sucht nach IT-Sicherheitsstellen und liefert die zehn relevantesten Treffer.",
                "value": {
                    "query": "Sachbearbeiter IT-Sicherheit",
                    "top_k": 10,
                    "base": "bm25",
                    "use_qe": False,
                    "use_kg": False,
                    "use_ce": False,
                },
            },
            "Suche mit Filtern": {
                "summary": "Suche mit Ort und Beschäftigungsart",
                "description": "Beschränkt die Suche auf Vollzeitstellen im Umkreis von 50 km um Rostock.",
                "value": {
                    "query": "Verwaltung Sachbearbeitung",
                    "top_k": 10,
                    "candidate_k": 50,
                    "base": "rrf",
                    "use_qe": True,
                    "use_ce": True,
                    "filters": {
                        "plz_oder_ort": "18055",
                        "max_distance_km": 50,
                        "beschaeftigungsart": "Vollzeit",
                        "dienstverhaeltnis": ["Beamter", "Angestellter"],
                    },
                    "sort_by": "distance",
                },
            },
            "Dense-Suche": {
                "summary": "Semantische Dense-Suche",
                "description": "Verwendet den Dense-Retriever für eine semantische Suche ohne exakte Stichwortübereinstimmung.",
                "value": {
                    "query": "Beratung von Bürgerinnen und Bürgern zu Sozialleistungen",
                    "top_k": 5,
                    "candidate_k": 50,
                    "base": "dense",
                    "normalize_scores": True,
                },
            },
        },
    )
):
    if not PC.corpus:
        raise HTTPException(status_code=503, detail="Service noch nicht vollständig initialisiert.")
    if req.use_kg:
        raise HTTPException(
            status_code=410,
            detail="Knowledge-Graph-Reranking ist in der Produktivkonfiguration deaktiviert.",
        )

    # --- Referenzpunkt für Distanz (unabhängig von max_distance_km, da auch
    #     sort_by='distance' ohne Radius-Filter funktionieren soll) ---
    ref_latlon = None
    if req.filters and req.filters.plz_oder_ort:
        ref_latlon = resolve_to_latlon(req.filters.plz_oder_ort)
        if ref_latlon is None:
            logger.warning("PLZ/Ort '%s' konnte nicht aufgelöst werden.", req.filters.plz_oder_ort)

    if req.sort_by == "distance" and ref_latlon is None:
        raise HTTPException(
            status_code=400,
            detail="sort_by='distance' erfordert einen auflösbaren filters.plz_oder_ort.",
        )

    distances_km: Dict[str, float] = {}
    if ref_latlon is not None:
        distances_km = _compute_min_distances_km(
            ref_latlon[0], ref_latlon[1], PC.doc_lats, PC.doc_lons, PC.doc_ids_with_geo
        )

    query_for_retrieval = req.query
    preprocessing_applied: List[QueryPreprocessingStep] = []

    # --- Rechtschreibkorrektur (vor Filterung und Retrieval) ---
    if req.correct_spelling and PC.spell_corrector is not None:
        corrected_query, spelling_corrected = PC.spell_corrector.correct(req.query)
        if spelling_corrected:
            logger.info("Rechtschreibkorrektur: '%s' -> '%s'", req.query, corrected_query)
            query_for_retrieval = corrected_query
            preprocessing_applied.append(
                QueryPreprocessingStep(step="spelling_correction", result=corrected_query)
            )

    # --- Abkürzungserkennung (nach Rechtschreibkorrektur, vor Retrieval) ---
    if req.expand_abbreviations and PC.abbreviation_expander is not None:
        expanded, abbreviations_expanded = PC.abbreviation_expander.expand(query_for_retrieval)
        if abbreviations_expanded:
            logger.info("Abkürzungen erkannt: '%s' -> '%s'", query_for_retrieval, expanded)
            query_for_retrieval = expanded
            preprocessing_applied.append(
                QueryPreprocessingStep(step="abbreviation_expansion", result=expanded)
            )

    # --- Behördensprache/Amtsdeutsch-Synonyme (nach Abkürzungen, vor Schreibvarianten) ---
    if req.normalize_behoerdensprache and PC.behoerden_normalizer is not None:
        behoerden_expanded, behoerdensprache_expanded = PC.behoerden_normalizer.expand_query(query_for_retrieval)
        if behoerdensprache_expanded:
            logger.info("Behördensprache erweitert: '%s' -> '%s'", query_for_retrieval, behoerden_expanded)
            query_for_retrieval = behoerden_expanded
            preprocessing_applied.append(
                QueryPreprocessingStep(step="behoerdensprache_expansion", result=behoerden_expanded)
            )

    # --- Schreibvarianten-Normalisierung (nach Abkürzungen, vor Retrieval) ---
    if req.normalize_spelling_variants and PC.variant_normalizer is not None:
        normalized, variants_normalized = PC.variant_normalizer.expand_query(query_for_retrieval)
        if variants_normalized:
            logger.info("Schreibvarianten normalisiert: '%s' -> '%s'", query_for_retrieval, normalized)
            query_for_retrieval = normalized
            preprocessing_applied.append(
                QueryPreprocessingStep(step="spelling_variant_normalization", result=normalized)
            )

    if req.split_compounds and PC.compound_splitter is not None:
        split_result, compounds_split = PC.compound_splitter.expand_query(query_for_retrieval)
        if compounds_split:
            query_for_retrieval = split_result
            preprocessing_applied.append(
                QueryPreprocessingStep(step="compound_splitting", result=split_result)
            )

    # --- Pre-Filter (PLZ/Ort, Dienstverhältnis, Bereich, ...) ---
    allowed_ids = get_allowed_doc_ids(
        PC.corpus, req.filters, PC.doc_lats, PC.doc_lons, PC.doc_ids_with_geo
    )

    if allowed_ids is not None and len(allowed_ids) == 0:
        # Filter hat alles herausgefiltert -> leeres Ergebnis, kein Fehler
        return SearchResponse(
            query=req.query,
            base=req.base,
            num_candidates_after_filter=0,
            results=[],
            final_query=query_for_retrieval,
            preprocessing_applied=preprocessing_applied,
        )

    # --- Pipeline zusammenbauen (BM25 / Dense / RRF / Alpha / XGB + Modifier) ---
    pipeline = ComponentPipeline(
        scorer=make_scorer(req.base),
        corpus=PC.corpus,
        query_expander=PC.query_expander if req.use_qe else None,
        ce_reranker=PC.ce_reranker if req.use_ce else None,
        candidate_k=req.candidate_k,
        normalize_before_kg=req.normalize_scores,
    )

    # Wichtig: hier wird die ggf. korrigierte/erweiterte Query verwendet, nicht req.query.
    raw_results = pipeline.search(
        query_for_retrieval, k=req.top_k, allowed_ids=allowed_ids, return_scores=True
    )

    results = [
        build_result_item(
            doc_id, score, PC.corpus, req.include_document_details, distances_km.get(doc_id)
        )
        for doc_id, score in raw_results
    ]

    # --- Nachträgliche Sortierung nach Aktualität/Entfernung (Default: Relevanz-Ranking der Pipeline bleibt erhalten) ---
    results = _sort_results(results, req.sort_by, PC.corpus)

    return SearchResponse(
        query=req.query,
        base=req.base,
        num_candidates_after_filter=len(allowed_ids) if allowed_ids is not None else len(PC.corpus),
        results=results,
        final_query=query_for_retrieval,
        preprocessing_applied=preprocessing_applied,
    )