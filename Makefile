# =====================================================
# Masterarbeit: Retrieval Experiments Pipeline
# =====================================================
PYTHON := .env/bin/python
DATA_DIR := data/processed

# Transform scripts
TRANSFORM_MIXED := src.transform_data.transform_interamt_jobs_mixed_queries
TRANSFORM_EXPERT := src.transform_data.transform_interamt_jobs_expert_queries
TRANSFORM_LOG_QUERIES := src.transform_data.transform_interamt_jobs_log_queries
SPLIT_QRELS_SCRIPT := src.scripts.split_qrels
EXTRACT_QUERIES_SCRIPT := src.scripts.extract_queries_from_log

# Evaluation scripts
RUN_SCRIPT := src.evaluation.run_experiments
RUN_PIPELINE_SCRIPT := src.evaluation.run_pipeline_experiment
RUN_GRID_SCRIPT := src.evaluation.run_pipeline_grid
PLOT_GRID_SCRIPT := src.evaluation.plot_grid_results
QREL_EVAL_SCRIPT := src.evaluation.qrel_eval
SIGNIFICANCE_SCRIPT := src.evaluation.significance_test

# Default dataset sizes
SIZE ?= full

# Qrels-Split defaults (Train/Test für XGB-Training)
TEST_FRAC ?= 0.3
SEED ?= 17

# Component-test defaults
DATASET ?= natural
BASE ?= bm25

# Plot-only defaults (siehe 'plots'-Target)
RESULTS_DIR ?= results/components

# Log-Query-Extraktion defaults (siehe 'extract-queries-from-log'-Target)
LOG_INPUT ?= data/raw/gesamt.log
QUERIES_OUT ?= data/raw/queries_from_log.jsonl
MIN_COUNT ?= 1

# REST Service
API_MODULE := src.api.rest_service:app
PORT ?= 8000

TRAIN_XGB_SCRIPT := src.scripts.train_xgb_model

QE_FLAG     := $(if $(QE),--qe,)
KG_FLAG     := $(if $(KG),--kg,)
CE_FLAG     := $(if $(CE),--ce,)
NORM_FLAG   := $(if $(NORM),--normalize-scores,)
TAG_FLAG    := $(if $(TAG),--tag $(TAG),)
K_FLAG      := $(if $(K),--k $(K),)
CANDK_FLAG  := $(if $(CANDIDATE_K),--candidate-k $(CANDIDATE_K),)
ALPHA_FLAG  := $(if $(ALPHA),--alpha $(ALPHA),)

# Grid-test (volle Kombinatorik) defaults
SKIPPP_FLAG := $(if $(SKIP_PREPROCESSING),--skip-preprocessing,)
BASES_FLAG  := $(if $(BASES),--bases $(BASES),)
SKIPQE_FLAG := $(if $(SKIP_QE),--skip-qe,)
SKIPKG_FLAG := $(if $(SKIP_KG),--skip-kg,)
SKIPCE_FLAG := $(if $(SKIP_CE),--skip-ce,)

# =====================================================
# HELP
# =====================================================
.PHONY: help setup transform-mixed transform-expert transform-log-queries natural keyword mixed component grid split-qrels serve train-xgb plots significance-test clean extract-queries-from-log

help:
	@echo "Targets:"
	@echo "  make setup"
	@echo "  make transform-mixed SIZE=1000"
	@echo "  make transform-expert SIZE=1000 [TEST_FRAC=0.2] [SEED=42]  (splittet Qrels automatisch in train/test)"
	@echo "  make transform-log-queries SIZE=1000"
	@echo "  make natural SIZE=1000"
	@echo "  make keyword SIZE=1000"
	@echo "  make mixed SIZE=1000"
	@echo "  make component DATASET=natural BASE=bm25 [QE=1] [KG=1] [CE=1] [NORM=1] [TAG=...] [K=10] [CANDIDATE_K=50] [ALPHA=0.2]"
	@echo "  make grid DATASET=natural [BASES=\"bm25 dense\"] [SKIP_PREPROCESSING=1] [SKIP_QE=1] [SKIP_KG=1] [SKIP_CE=1] [NORM=1] [K=10] [CANDIDATE_K=50] [ALPHA=0.2]"
	@echo "  make plots [RESULTS_DIR=results/components]  (Plots aus vorhandenen Grid-Ergebnissen neu erzeugen, ohne Retrieval)"
	@echo "  make split-qrels DATASET=expert [TEST_FRAC=0.2] [SEED=42]  (manuell neu splitten, ohne neu zu transformieren)"
	@echo "  make extract-queries-from-log [LOG_INPUT=data/raw/gesamt.log] [QUERIES_OUT=data/raw/queries_from_log.jsonl] [MIN_COUNT=1]"
	@echo "  make serve [PORT=8080]"
	@echo "  make train-xgb"
	@echo "  make clean"

# =====================================================
# SETUP
# =====================================================
setup:
	python3 -m venv .env
	.env/bin/pip install -r requirements.txt --break-system-packages

# =====================================================
# DATA TRANSFORMATION
# =====================================================
DATA_RAW := data/raw/output_v1.json
QUERIES_RAW := data/raw/queries_from_log.jsonl

transform-mixed:
	$(PYTHON) -m $(TRANSFORM_MIXED) \
		--input $(DATA_RAW) \
		--output $(DATA_DIR)/jobs_interamt_mixed \
		--size $(SIZE)

# Erzeugt corpus/queries/qrels und splittet die Qrels direkt danach in
# train.tsv / test.tsv, damit XGB nie auf denselben Queries trainiert und
# evaluiert wird. Die vollständigen (ungesplitteten) Qrels bleiben zusätzlich
# als qrels/all.tsv erhalten.
transform-expert:
	$(PYTHON) -m $(TRANSFORM_EXPERT) \
		--input $(DATA_RAW) \
		--queries $(QUERIES_RAW) \
		--output $(DATA_DIR)/jobs_interamt_expert \
		--size $(SIZE)
	$(PYTHON) -m $(SPLIT_QRELS_SCRIPT) \
		--dataset $(DATA_DIR)/jobs_interamt_expert \
		--test-frac $(TEST_FRAC) \
		--seed $(SEED)

transform-log-queries:
	$(PYTHON) -m $(TRANSFORM_LOG_QUERIES) \
		--input $(DATA_RAW) \
		--queries $(QUERIES_RAW) \
		--output $(DATA_DIR)/jobs_interamt_log_queries \
		--size $(SIZE)
	$(PYTHON) -m $(SPLIT_QRELS_SCRIPT) \
		--dataset $(DATA_DIR)/jobs_interamt_log_queries \
		--test-frac $(TEST_FRAC) \
		--seed $(SEED)


# =====================================================
# QUERIES AUS PRODUKTIONS-LOG EXTRAHIEREN
# =====================================================
# Baut aus einem Interamt-Stellensuche-Logfile eine queries.jsonl (nur
# Zeilen mit nicht-leerem suchText, dedupliziert nach Parameter-Kombination).
# Kann direkt als --queries für transform-expert genutzt werden (z.B. indem
# man QUERIES_RAW oben auf QUERIES_OUT umbiegt oder beim Aufruf überschreibt:
# make transform-expert QUERIES_RAW=data/raw/queries_from_log.jsonl).
#
# Beispiele:
#   make extract-queries-from-log
#   make extract-queries-from-log LOG_INPUT=data/raw/gesamt.log MIN_COUNT=2
extract-queries-from-log:
	$(PYTHON) -m $(EXTRACT_QUERIES_SCRIPT) \
		--input $(LOG_INPUT) \
		--output $(QUERIES_OUT) \
		--min-count $(MIN_COUNT)

# =====================================================
# FULL EXPERIMENTS (alle Modelle auf einmal, run_experiments.py)
# =====================================================
mixed:
ifndef SIZE
	$(error SIZE nicht gesetzt (z.B. SIZE=1000))
endif
	$(PYTHON) -m $(RUN_SCRIPT) \
		--dataset $(DATA_DIR)/jobs_interamt_mixed \
		--tag mixed_$(SIZE)

# =====================================================
# COMPONENT TEST (eine einzelne Kombination, run_pipeline_experiment.py)
# =====================================================
# Beispiele:
#   make component BASE=bm25 KG=1
#   make component BASE=bm25 KG=1 NORM=1
#   make component DATASET=keyword BASE=dense CE=1
#   make component BASE=rrf QE=1 KG=1 CE=1 TAG=rrf_full_stack
component:
	$(PYTHON) -m $(RUN_PIPELINE_SCRIPT) \
		--dataset $(DATA_DIR)/jobs_interamt_$(DATASET) \
		--base $(BASE) \
		$(QE_FLAG) $(KG_FLAG) $(CE_FLAG) $(NORM_FLAG) $(TAG_FLAG) $(K_FLAG) $(CANDK_FLAG) $(ALPHA_FLAG)

# =====================================================
# GRID TEST (volle Kombinatorik, run_pipeline_grid.py)
# =====================================================
# Beispiele:
#   make grid DATASET=natural                     -> alle 5 Bases x QE x KG x CE (40 Läufe)
#   make grid DATASET=expert SKIP_PREPROCESSING=1  -> keine Vorverarbeitung, sonst volle Kombinatorik
#   make grid DATASET=natural BASES="bm25 dense"   -> nur diese Bases, volle QE/KG/CE-Kombinatorik
#   make grid DATASET=natural SKIP_CE=1            -> Cross-Encoder-Varianten überspringen (schneller)
#   make grid DATASET=natural NORM=1               -> Scores vor KG-Rerank normalisieren
grid:
	$(PYTHON) -m $(RUN_GRID_SCRIPT) \
		--dataset $(DATA_DIR)/jobs_interamt_$(DATASET) \
		$(BASES_FLAG) $(SKIPPP_FLAG) $(SKIPQE_FLAG) $(SKIPKG_FLAG) $(SKIPCE_FLAG) $(NORM_FLAG) $(K_FLAG) $(CANDK_FLAG) $(ALPHA_FLAG) --use-cached-xgb

# =====================================================
# PLOTS ONLY (aus vorhandenen Grid-Ergebnissen, ohne Retrieval)
# =====================================================
# Liest grid_summary.csv + die per-Query-JSONs aus RESULTS_DIR (Standard:
# results/components, wie von 'make grid' erzeugt) und erzeugt alle Plots
# neu. Läuft in Sekunden statt Minuten/Stunden, da keine Retriever/Reranker
# instanziiert werden -- praktisch beim Iterieren an src/evaluation/plots.py.
#
# Beispiele:
#   make plots
#   make plots RESULTS_DIR=results/components_expert
plots:
	$(PYTHON) -m $(PLOT_GRID_SCRIPT) \
		--results-dir $(RESULTS_DIR)

# =====================================================
# BOOTSTRAP-SIGNIFIKANZTESTS + WIN/LOSS PRO QUERY (ggü. BM25-Baseline)
# =====================================================
# Liest grid_summary.csv + die per-Query-JSONs aus RESULTS_DIR und testet
# jede Konfiguration per paired Bootstrap gegen BASE (Standard: bm25).
#
# Beispiele:
#   make significance-test
#   make significance-test RESULTS_DIR=results/components_expert BASE=bm25
significance-test:
	$(PYTHON) -m $(SIGNIFICANCE_SCRIPT) \
		--results-dir $(RESULTS_DIR) \
		--baseline $(BASE)


qrel-eval:
	$(PYTHON) -m $(QREL_EVAL_SCRIPT) \
		--corpus $(if $(CORPUS),$(CORPUS),$(DATA_DIR)/jobs_interamt_$(DATASET)/corpus.jsonl) \
		--queries $(if $(QUERIES),$(QUERIES),$(DATA_DIR)/jobs_interamt_$(DATASET)/queries.jsonl) \
		--qrels $(if $(QRELS),$(QRELS),$(DATA_DIR)/jobs_interamt_$(DATASET)/qrels/test.tsv) \
		$(if $(N_SAMPLES),--n-samples $(N_SAMPLES))

# =====================================================
# QRELS SPLIT (manuell, z.B. um mit anderem TEST_FRAC/SEED neu zu splitten,
# ohne den kompletten Transform-Schritt erneut laufen zu lassen)
# =====================================================
# Beispiele:
#   make split-qrels DATASET=expert
#   make split-qrels DATASET=expert TEST_FRAC=0.3 SEED=7
split-qrels:
	$(PYTHON) -m $(SPLIT_QRELS_SCRIPT) \
		--dataset $(DATA_DIR)/jobs_interamt_$(DATASET) \
		--test-frac $(TEST_FRAC) \
		--seed $(SEED)

# =====================================================
# REST SERVICE
# =====================================================
# Beispiele:
#   make serve
#   make serve PORT=8080
serve:
	$(PYTHON) -m uvicorn $(API_MODULE) --reload --port $(PORT)

# =====================================================
# XGB MODEL TRAINING (einmalig, offline)
# =====================================================
# Trainiert das XGB-Fusion-Modell und speichert es unter models/xgb_fusion.json.
# Muss einmal ausgeführt werden, bevor base=xgb im REST-Service funktioniert.
train-xgb:
	$(PYTHON) -m $(TRAIN_XGB_SCRIPT)

# =====================================================
# LAUFZEITTEST (lokal, in-process, ohne Docker/PostgreSQL)
# =====================================================
# Misst Latenz pro Architektur-Stufe (Preprocessing, Filter, Retrieval/Fusion,
# Cross-Encoder, Ergebnisaufbau) über echte Log-Queries.
# Ergebnisse: results/benchmark/<timestamp>/
#
# Beispiele:
#   make benchmark
#   make benchmark N_QUERIES=200 SCENARIOS=bm25,rrf,rrf+ce
N_QUERIES ?= 100
benchmark:
	$(PYTHON) -m src.scripts.benchmark_latency \
		--n-queries $(N_QUERIES) \
		$(if $(SCENARIOS),--scenarios $(SCENARIOS),)

# PDF-Bericht aus einem Lauf: make benchmark-report RUN=results/benchmark/20261118_164249
benchmark-report:
	$(PYTHON) -m src.scripts.benchmark_report $(RUN)

# =====================================================
# CLEAN
# =====================================================
clean:
	rm -rf src/evaluation/results
	rm -rf results