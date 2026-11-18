# ai-suche

REST-API für die semantische Vorauswahl/Suche über Interamt-Stellenausschreibungen. Kombiniert BM25, Dense Retrieval (FAISS + BAAI/bge-m3) und Hybrid-Fusion (RRF / Alpha / XGBoost-LTR) mit optionaler Query-Preprocessing-Pipeline und Reranking.s

## Architektur

```
Anfrage → Query Preprocessing → Retrieval (BM25 / Dense) → Fusion (RRF / Alpha / XGB) → Reranking (optional) → Ergebnisse
```

- **Query Preprocessing** (`src/retrieval/`): Rechtschreibkorrektur (`spell_correction.py`, symspellpy), Abkürzungsexpansion (`abbreviation_expansion.py`), Schreibvarianten/ß-ss/Gender-Suffixe (`spelling_variants.py`), Komposita-Zerlegung (`compound_splitter.py`), Behördensprache-Normalisierung (`behoerden_normalizer.py`)
- **Retrieval**: BM25 (`bm25_retrieval.py`), Dense Retrieval mit FAISS + bge-m3 (`dense_retrieval.py`, `embedding_store.py`, `query_embedding_cache.py`)
- **Fusion**: Reciprocal Rank Fusion (`rrf_fusion.py`), gewichtete Alpha-Fusion, XGBoost-basierte Learning-to-Rank-Fusion (`xgb_fusion_retriever.py`, `src/rerank/xgb_fusion.py`)
- **Reranking** (optional): Cross-Encoder (`cross_encoder_reranker.py`), Knowledge-Graph-basiert (`kg_reranker.py`, `semantic_skill_kg.py`)
- **API**: FastAPI-Service (`src/api/rest_service.py`)

## Voraussetzungen

- **Python 3.11 oder 3.12** (nicht 3.14 verwenden — `symspellpy` hat C++-Abhängigkeiten, die unter 3.14 nicht kompatibel sind)
- Zugriff auf die Modell-Artefakte:
  - Dense-Embedding-Index (FAISS, `BAAI/bge-m3`)
  - Trainiertes XGB-Fusion-Modell unter `models/xgb_fusion.json` (wird beim Start geladen; falls nicht vorhanden, vorher mit `scripts/train_xgb_model.py` erzeugen)
- Eine lokale `.env.local` mit `INTERAMT_API_KEY` für den täglichen Korpus-Abruf

## Setup

```bash
# Virtuelle Umgebung anlegen
python3.11 -m venv .venv
source .venv/bin/activate   # unter WSL/Linux

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Daten & Modelle vorbereiten

1. `INTERAMT_API_KEY` in der Umgebung oder in `.env.local` setzen. Beim Start und bei `POST /admin/reload` wird `https://mps.interamt.de/dtag/api/query` abgerufen. Das Feld `tbehoerde` wird rekursiv vor dem Einlesen entfernt.
2. Dense-Embedding-Index initial aufbauen (wird beim ersten Start automatisch erzeugt). Die Corpus-Embeddings werden dabei in PostgreSQL gespeichert; der FAISS-Index wird weiterhin zur Laufzeit aus der Datenbank aufgebaut.
3. XGB-Fusion-Modell trainieren, falls `models/xgb_fusion.json` noch nicht existiert:
   ```bash
   make train-xgb
   ```

## API starten

```bash
make serve
```

Der Prozess startet standardmäßig als `master`: Er ruft den Korpus ab und darf
den Embedding-Cache in PostgreSQL aktualisieren. Weitere API-Instanzen können
als lesende `worker` gestartet werden, indem `AI_SUCH_ROLE=worker` gesetzt wird.
Der Master muss den Korpus und den Embedding-Cache zuerst aufgebaut haben.

Mit Docker kann beispielsweise ein Master und danach ein Worker gestartet
werden:

```bash
docker compose up -d --build
docker compose run -d --name ai-suche-worker -p 8001:8000 \
  -e AI_SUCH_ROLE=worker ai-suche-api
```

Worker laden nur den vorhandenen Korpus und Embedding-Cache. Korpus-Download,
Schema-Erzeugung, Cache-Schreibzugriffe und `POST /admin/reload` sind dort
gesperrt.

Die interaktive Swagger-Dokumentation ist anschließend unter
`http://localhost:8000/docs` erreichbar. Dort stehen beim Endpoint `POST /search`
mehrere Beispielanfragen zur Auswahl und können direkt mit „Try it out“ ausgeführt
werden. Die OpenAPI-Spezifikation ist unter `http://localhost:8000/openapi.json`
und ReDoc unter `http://localhost:8000/redoc` verfügbar.

Für den produktiven Betrieb empfiehlt sich ein Prozessmanager/Reverse Proxy davor (z. B. `systemd`-Service + nginx, oder ein Container-Setup), sowie mehrere Worker über `--workers`, sofern der Service zustandslos genug ist (Vorsicht: der Dense-Index und Caches werden pro Prozess geladen — RAM-Bedarf mit der Anzahl Worker multiplizieren).

## Endpunkte (Stand bekannter Architektur)

| Methode | Pfad | Beschreibung |
|---|---|---|
| `POST` | `/search` | Hauptsuche: Query-Preprocessing, Retrieval, Fusion, optionale Filter (z. B. `bereich`, `dienstort`, `beschaeftigungsart`, Geo-Radius) und Reranking. Antwort enthält u. a. `preprocessing_applied`, `final_query` sowie pro Treffer ein `details`-Objekt. |
| `POST` | `/admin/reload` | Lädt Corpus/Modelle neu (Double-Buffering, ohne Downtime). |
| `GET` | `/health` | Readiness-Check für Pipeline und PostgreSQL; liefert bei fehlender Bereitschaft HTTP `503`. |
| `GET` | `/status` | Detaillierter Service- und Datenbankstatus inklusive Korpusgröße und aktiver Komponenten. |

### Beispielanfragen

Die Beispiele sind direkt in der Swagger-Oberfläche hinterlegt. Eine einfache Suche
kann beispielsweise auch per `curl` ausgeführt werden:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Sachbearbeiter IT-Sicherheit","top_k":10,"base":"bm25"}'
```


## Konfiguration

- Pfade zu Corpus, FAISS-Index, XGB-Modell
- PostgreSQL-Verbindung für die Embedding-Datenbank über `EMBEDDING_DATABASE_URL`
- Prozessrolle über `AI_SUCH_ROLE` (`master`, Standard, oder `worker`)
- Modellname für Dense Retrieval (`BAAI/bge-m3`)
- GPU/CPU-Auswahl für Cross-Encoder-Reranking
- Cache-Verzeichnisse

## Bekannte Einschränkungen

- Kategoriale Vor-Filter (`bereich`/`dienstort`/`beschaeftigungsart`/`dienstverhaeltnis`) sind nur zuverlässig, wenn der Corpus mit strukturierten Feldern (nicht nur Text-Präambel) erzeugt wurde.
- Der Dense-Index sowie diverse Caches (Query-Embeddings, Cross-Encoder-Scores, Semantic-Skill-KG) sind prozesslokal — bei mehreren Workern/Instanzen entsprechend Warm-up-Zeit und RAM einplanen.

## Troubleshooting

- **`FileNotFoundError` beim Speichern von Modellen/Caches**: Zielverzeichnis (`models/`) existiert nicht — wird durch `os.makedirs` beim Start erzeugt; falls der Fehler dennoch auftritt, Schreibrechte im Deployment-Verzeichnis prüfen.
- **Leere/keine Ergebnisse bei gesetzten kategorialen Filtern**: Corpus-Schema prüfen (siehe "Bekannte Einschränkungen").
- **PostgreSQL**: Mit `docker compose up --build` wird PostgreSQL mit `pgvector` gestartet. Die Embeddings liegen in der Tabelle `embeddings`; der Standardzugriff erfolgt über `EMBEDDING_DATABASE_URL`.
