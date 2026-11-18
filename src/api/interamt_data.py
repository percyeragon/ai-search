"""Abruf und Normalisierung des aktuellen Interamt-Korpus."""

import json
import logging
import os
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(".env.local")
else:
    env_file = Path(".env.local")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


logger = logging.getLogger("rest_service")

INTERAMT_API_URL = os.getenv(
    "INTERAMT_API_URL", "https://mps.interamt.de/dtag/api/query"
)
INTERAMT_API_KEY_ENV = "INTERAMT_API_KEY"
INTERAMT_FETCH_TIMEOUT = float(os.getenv("INTERAMT_FETCH_TIMEOUT", "120"))


class _HtmlTextExtractor(HTMLParser):
    _BLOCK_TAGS = {"br", "div", "li", "p", "td", "th", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif tag in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self._ignored_depth:
            self.parts.append(data)


def _html_to_text(value):
    if not isinstance(value, str):
        return ""
    parser = _HtmlTextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


def _remove_tbehoerde(value):
    if isinstance(value, dict):
        return {
            key: _remove_tbehoerde(item)
            for key, item in value.items()
            if key != "tbehoerde"
        }
    if isinstance(value, list):
        return [_remove_tbehoerde(item) for item in value]
    return value


def _extract_documents(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results", "jobs", "stellen", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError("Interamt-API hat kein JSON-Objekt oder JSON-Array geliefert.")


_DIENSTVERHAELTNIS_LABELS = {
    "ANGESTELLTER": "Angestellter",
    "BEAMTER": "Beamter",
    "BEAMTENANWAERTER": "Beamtenanwärter",
    "AUSZUBILDENDER": "Auszubildender",
    "DUALES_STUDIUM": "Duales Studium",
    "PRAKTIKUM": "Praktikum",
}


def _normalize_api_record(record):
    """Adds the canonical corpus fields used by retrieval and response code."""
    dienstverhaeltnise = record.get("dienstverhaeltnise")
    if "dienstverhaeltnis" not in record and isinstance(dienstverhaeltnise, list):
        record["dienstverhaeltnis"] = [
            _DIENSTVERHAELTNIS_LABELS.get(
                item.get("type"), item.get("type")
            )
            for item in dienstverhaeltnise
            if isinstance(item, dict) and item.get("type")
        ]

    if "bereich" not in record and isinstance(record.get("aufgabenbereich"), list):
        record["bereich"] = record["aufgabenbereich"]

    employment_type = record.get("teilzeitVollzeit")
    if "beschaeftigungsart" not in record and employment_type:
        record["beschaeftigungsart"] = {
            "VOLLZEIT": "Vollzeit",
            "TEILZEIT": "Teilzeit",
            "BEIDES_MOEGLICH": "Vollzeit oder Teilzeit",
        }.get(employment_type, employment_type)

    einsatzorte = record.get("einsatzorte")
    if "dienstort" not in record and isinstance(einsatzorte, list):
        record["dienstort"] = next(
            (item.get("ort") for item in einsatzorte if isinstance(item, dict) and item.get("ort")),
            None,
        )

    field_mapping = {
        "datumVeroeffentlicht": "veroeffentlicht_am",
        "datumBewerbungsfrist": "bewerbungsfrist",
        "anzahlPosten": "anzahl_posten",
        "behoerde": "dienststelle",
        "behoerdeUrl": "url",
    }
    for source, target in field_mapping.items():
        if target not in record and source in record:
            record[target] = record[source]

    return record


def _to_jsonl_records(payload):
    documents = _extract_documents(_remove_tbehoerde(payload))
    records = {}

    for position, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"Dokument {position} ist kein JSON-Objekt.")

        record = _normalize_api_record(dict(document))
        doc_id = (
            record.get("_id")
            or record.get("id")
            or record.get("stellenId")
            or record.get("stellen_id")
            or record.get("kennung")
            or f"job_{position}"
        )
        record["_id"] = str(doc_id)

        if not isinstance(record.get("title"), str):
            record["title"] = str(
                record.get("bezeichnung")
                or record.get("stellenbezeichnung")
                or record.get("kennung")
                or ""
            )
        if not isinstance(record.get("text"), str):
            description = (
                record.get("stellenbeschreibung")
                or record.get("beschreibung")
                or record.get("description")
                or record["title"]
                or ""
            )
            record["text"] = _html_to_text(description) or record["title"]

        if record["_id"] in records:
            raise ValueError(f"Doppelte Dokument-ID im Interamt-Korpus: {record['_id']}")
        records[record["_id"]] = record

    return records


def fetch_and_store_corpus(destination: Path) -> int:
    """Lädt den Korpus, bereinigt ihn und ersetzt die lokale Datei atomar."""
    api_key = os.getenv(INTERAMT_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{INTERAMT_API_KEY_ENV} ist nicht gesetzt; der Interamt-Korpus "
            "kann nicht geladen werden."
        )

    request = Request(
        INTERAMT_API_URL,
        headers={"X-API-Key": api_key, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=INTERAMT_FETCH_TIMEOUT) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Interamt-API konnte nicht abgerufen werden: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Interamt-API hat kein gültiges JSON geliefert.") from error

    records = _to_jsonl_records(payload)
    if not records:
        raise RuntimeError("Interamt-API hat keine Dokumente geliefert.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        for record in records.values():
            temporary_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.info("Interamt-Korpus geladen: %d Dokumente nach %s", len(records), destination)
    return len(records)