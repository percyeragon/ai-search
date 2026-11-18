"""
Normalisierung unterschiedlicher Schreibweisen desselben Begriffs.

Anders als AbbreviationExpander (kurz -> lang) geht es hier um
GLEICHWERTIGE Varianten eines Begriffs, die auf eine kanonische Form
abgebildet werden -- z.B. Gender-Schreibweisen (Mitarbeiter/in,
Mitarbeiter:in, MitarbeiterIn), ß/ss-Varianten oder Kompositum-Schreib-
weisen (Homeoffice / Home-Office / Home Office).

WICHTIG: Damit das beim Retrieval etwas bringt, muss dieselbe
Normalisierung sowohl auf die Query ALS AUCH beim Indexieren des
Korpus (BM25-Tokenisierung, Dense-Embedding-Input) angewendet werden.
Sonst matcht "Mitarbeiter:in" in der Query nie gegen "MitarbeiterIn"
im Dokument. Siehe normalize_text() weiter unten -- die wird an
BM25Retriever/DenseRetriever als Preprocessing-Hook übergeben.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Pattern, Tuple


class SpellingVariantNormalizer:
    def __init__(self, custom_path: Path = None):
        """
        custom_path: optionale JSON-Datei {"kanonische_form": ["variante1", ...], ...},
        um projektspezifische Varianten zu ergänzen/überschreiben.
        """
        # Reihenfolge wichtig: generische Muster (Regex) zuerst, dann
        # feste Wortlisten. Regeln sind bewusst als *Funktionen*
        # modelliert (nicht nur als Dict), weil Gender-Formen und
        # Komposita sich nicht mit einer festen Ersetzungsliste
        # abdecken lassen (zu viele mögliche Grundwörter).
        self._pattern_rules: List[Tuple[Pattern, str]] = [
            # Gender-Suffixe (Binnen-I, /in, :in, *in, _in) -> kanonisch auf
            # die neutrale Grundform ohne Suffix normalisieren.
            # "Sachbearbeiter/in" / "Sachbearbeiter:in" / "Sachbearbeiter*in"
            # / "SachbearbeiterIn" / "Sachbearbeiter_in" -> "sachbearbeiter"
            (re.compile(r'([a-zäöüß]+)(?:/in|:in|\*in|_in|In)\b'), r'\1'),
            # ß -> ss (einheitlich auf eine Form normalisieren; welche
            # Richtung ist letztlich egal, Hauptsache konsistent)
            (re.compile(r'ß'), 'ss'),
            # Bindestrich-Komposita ohne Leerzeichen dazwischen:
            # "Home-Office" / "Home Office" -> "homeoffice"
            # (nur für bekannte Begriffe unten, NICHT generisch für alle
            # Bindestriche, da z.B. "Sachbearbeiter-in" (Ort) anders zu
            # behandeln wäre als Komposita-Trennung)
        ]

        # Feste Synonym-Gruppen: alle Varianten -> kanonische Form (lowercase).
        # Wird NACH den generischen Pattern-Regeln angewendet.
        self.canonical_map: Dict[str, str] = {
            "home-office": "homeoffice",
            "home office": "homeoffice",
            "remote-arbeit": "remotearbeit",
            "remote arbeit": "remotearbeit",
            "mobiles arbeiten": "homeoffice",  # fachlich synonym im Kontext Stellenausschreibung
            "e-mail": "email",
            "gleit-zeit": "gleitzeit",
            "gleit zeit": "gleitzeit",
            "voll-zeit": "vollzeit",
            "voll zeit": "vollzeit",
            "teil-zeit": "teilzeit",
            "teil zeit": "teilzeit",
            "quer-einsteiger": "quereinsteiger",
            "quer einsteiger": "quereinsteiger",
            "berufs-einsteiger": "berufseinsteiger",
            "berufs einsteiger": "berufseinsteiger",
        }

        if custom_path and custom_path.exists():
            with open(custom_path, "r", encoding="utf-8") as f:
                custom_groups: Dict[str, List[str]] = json.load(f)
            for canonical, variants in custom_groups.items():
                for v in variants:
                    self.canonical_map[v.lower()] = canonical.lower()

        # Längste Varianten zuerst ersetzen, damit z.B. "home office"
        # (2 Wörter) vor einem eventuellen Einzelwort-Treffer greift.
        self._sorted_variants = sorted(self.canonical_map.keys(), key=len, reverse=True)
        self._variant_pattern = re.compile(
            r'(?<![a-zäöüß])(' + '|'.join(re.escape(v) for v in self._sorted_variants) + r')(?![a-zäöüß])',
            re.IGNORECASE,
        )

    def normalize_text(self, text: str) -> str:
        """
        Kanonisiert einen Text vollständig. Diese Funktion MUSS identisch
        sowohl auf Queries als auch beim Korpus-Indexing angewendet werden
        (siehe Modulkommentar oben).
        """
        result = text
        for pattern, repl in self._pattern_rules:
            result = pattern.sub(repl, result)

        def _replace(m: re.Match) -> str:
            return self.canonical_map[m.group(1).lower()]

        result = self._variant_pattern.sub(_replace, result.lower())
        return result

    def expand_query(self, query: str) -> Tuple[str, bool]:
        """
        Für den Such-Endpoint: hängt die normalisierte Form zusätzlich an
        die Original-Query an (statt zu ersetzen), damit auch dann etwas
        matcht, wenn der Korpus NICHT durchgängig normalisiert indexiert
        wurde (z.B. Übergangsphase, bevor BM25/Dense neu aufgebaut sind).
        """
        normalized = self.normalize_text(query)
        if normalized == query.lower():
            return query, False
        return f"{query} {normalized}", True