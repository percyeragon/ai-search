"""
Rechtschreibkorrektur für Suchanfragen.

Kombiniert zwei Wörterbuch-Quellen für SymSpell:
1. Domänen-Vokabular aus dem eigenen Korpus (title, Aufgabenbereich,
   Fachrichtung, description) — hoch gewichtet, damit Fachbegriffe wie
   "Sachbearbeiter" oder "Dienstverhältnis" nicht überschrieben werden.
2. Ein allgemeines deutsches Frequenzwörterbuch (de_50k von
   hermitdave/FrequencyWords, einmalig heruntergeladen und lokal unter
   models/ gecacht), für Alltagswörter außerhalb der Stellenanzeigen.

Nutzt SymSpell.lookup_compound(), das mehrwortige Anfragen als Ganzes
korrigiert (inkl. Segmentierung von zusammengeschriebenen Wörtern) statt
Token für Token.
"""

import logging
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Tuple

from symspellpy import SymSpell

logger = logging.getLogger("spell_correction")

_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")

_GENERAL_DICT_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
    "master/content/2018/de/de_50k.txt"
)
_GENERAL_DICT_CACHE = Path("models/de_50k.txt")

# Domänen-Begriffe bekommen dieses Basisgewicht relativ zur Häufigkeit
# im allgemeinen Wörterbuch, damit sie bei Ties immer gewinnen.
_DOMAIN_WEIGHT = 2_000_000

# Optional: hier die gleichen kurzen Fachbegriffe eintragen, die du auch
# in transform_dtag_to_beir.py als SHORT_TECH_TERMS pflegst (SAP, SQL,
# Java, ...), damit sie garantiert nicht "korrigiert" werden.
SHORT_TECH_TERMS: set[str] = set()


def _ensure_general_dictionary(path: Path = _GENERAL_DICT_CACHE) -> Path:
    """Lädt das allgemeine dt. Frequenzwörterbuch einmalig herunter und cached es lokal."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Lade allgemeines deutsches Frequenzwörterbuch von %s ...", _GENERAL_DICT_URL)
    urllib.request.urlretrieve(_GENERAL_DICT_URL, path)
    return path


def _extract_domain_vocab(
    corpus: dict,
    fields: Iterable[str] = ("title", "Aufgabenbereich", "Fachrichtung", "description"),
) -> Counter:
    """Baut ein Vokabular-Counter aus den relevanten Textfeldern des Korpus."""
    counter: Counter = Counter()
    for doc in corpus.values():
        for field in fields:
            text = doc.get(field)
            if not text:
                continue
            counter.update(t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2)
    return counter


class SpellCorrector:
    """Korrigiert Rechtschreibfehler in Suchanfragen via SymSpell."""

    def __init__(self, corpus: dict, max_edit_distance: int = 2, prefix_length: int = 7):
        self._sym_spell = SymSpell(
            max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length
        )

        general_dict_path = _ensure_general_dictionary()
        self._sym_spell.load_dictionary(
            str(general_dict_path), term_index=0, count_index=1, separator=" "
        )

        domain_vocab = _extract_domain_vocab(corpus)
        if domain_vocab:
            max_domain_freq = max(domain_vocab.values())
            for term, freq in domain_vocab.items():
                boosted_freq = _DOMAIN_WEIGHT + int((freq / max_domain_freq) * _DOMAIN_WEIGHT)
                self._sym_spell.create_dictionary_entry(term, boosted_freq)

        for term in SHORT_TECH_TERMS:
            self._sym_spell.create_dictionary_entry(term.lower(), _DOMAIN_WEIGHT * 2)

        logger.info(
            "SpellCorrector initialisiert (%d Wörterbucheinträge, davon %d Domänenbegriffe).",
            len(self._sym_spell.words),
            len(domain_vocab),
        )

    def correct(self, query: str, max_edit_distance: int = 2) -> Tuple[str, bool]:
        """
        Korrigiert eine komplette Suchanfrage.

        Gibt (korrigierte_query, wurde_geaendert) zurück. Zahlen, PLZ und
        Sonderzeichen bleiben unangetastet (ignore_non_words=True); die
        Original-Groß-/Kleinschreibung wird per transfer_casing erhalten.
        """
        if not query or not query.strip():
            return query, False

        try:
            suggestions = self._sym_spell.lookup_compound(
                query,
                max_edit_distance=max_edit_distance,
                ignore_non_words=True,
                transfer_casing=True,
            )
        except IndexError:
            logger.warning("Spell correction skipped for unsupported query: %r", query)
            return query, False
        if not suggestions:
            return query, False

        corrected = suggestions[0].term
        changed = corrected.strip() != query.strip()
        return corrected, changed