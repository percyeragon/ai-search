"""
Erkennung und Auflösung von Abkürzungen in Suchqueries.

Läuft nach der Rechtschreibkorrektur und vor dem eigentlichen Retrieval.
Erkannte Abkürzungen werden NICHT ersetzt, sondern die ausgeschriebene(n)
Form(en) werden an die Query angehängt -- so matcht BM25/Dense sowohl
gegen die Abkürzung als auch gegen den ausgeschriebenen Begriff.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Zeichen, die als "Wortgrenze" um eine Abkürzung herum gelten.
# Bewusst KEIN \b verwendet, weil das bei Abkürzungen mit Punkt/Slash
# am Ende (z.B. "z.B.", "m/w/d") nicht zuverlässig funktioniert.
_BOUNDARY_CHARS = r'\s,;:!?()\[\]{}"\'/'

# Format: "Abkürzung": {"expansions": [...], "case_sensitive": bool}
# case_sensitive=True für kurze Großbuchstaben-Kürzel (SB, IT, EG, ...),
# um Kollisionen mit normalen Wörtern zu vermeiden.
DEFAULT_ABBREVIATIONS: Dict[str, dict] = {
    "TVöD": {"expansions": ["Tarifvertrag für den öffentlichen Dienst"], "case_sensitive": False},
    "TV-L": {"expansions": ["Tarifvertrag der Länder"], "case_sensitive": False},
    "TV-H": {"expansions": ["Tarifvertrag Hessen"], "case_sensitive": False},
    "BAT": {"expansions": ["Bundesangestelltentarifvertrag"], "case_sensitive": False},
    "EG": {"expansions": ["Entgeltgruppe"], "case_sensitive": True},
    "ÖD": {"expansions": ["öffentlicher Dienst"], "case_sensitive": True},
    "VZÄ": {"expansions": ["Vollzeitäquivalent"], "case_sensitive": True},
    "m/w/d": {"expansions": ["männlich weiblich divers"], "case_sensitive": False},
    "w/m/d": {"expansions": ["weiblich männlich divers"], "case_sensitive": False},
    "SB": {"expansions": ["Sachbearbeiter", "Sachbearbeiterin"], "case_sensitive": True},
    "FK": {"expansions": ["Führungskraft"], "case_sensitive": True},
    "IT": {"expansions": ["Informationstechnologie", "Informationstechnik"], "case_sensitive": True},
    "EDV": {"expansions": ["elektronische Datenverarbeitung"], "case_sensitive": True},
    "ADV": {"expansions": ["automatisierte Datenverarbeitung"], "case_sensitive": True},
    "BA": {"expansions": ["Bachelor"], "case_sensitive": True},
    "MA": {"expansions": ["Master"], "case_sensitive": True},
    "Dipl.-Ing.": {"expansions": ["Diplom-Ingenieur"], "case_sensitive": False},
    "Dipl.-Inf.": {"expansions": ["Diplom-Informatiker"], "case_sensitive": False},
    "z.B.": {"expansions": ["zum Beispiel"], "case_sensitive": False},
    "ggf.": {"expansions": ["gegebenenfalls"], "case_sensitive": False},
    "bzgl.": {"expansions": ["bezüglich"], "case_sensitive": False},
    "inkl.": {"expansions": ["inklusive"], "case_sensitive": False},
    "exkl.": {"expansions": ["exklusive"], "case_sensitive": False},
    "Abt.": {"expansions": ["Abteilung"], "case_sensitive": False},
    "Ref.": {"expansions": ["Referat"], "case_sensitive": False},
    "Dez.": {"expansions": ["Dezernat"], "case_sensitive": False},
    "Verw.": {"expansions": ["Verwaltung"], "case_sensitive": False},
    "TZ": {"expansions": ["Teilzeit"], "case_sensitive": True},
    "VZ": {"expansions": ["Vollzeit"], "case_sensitive": True},
    "unbefr.": {"expansions": ["unbefristet"], "case_sensitive": False},
    "befr.": {"expansions": ["befristet"], "case_sensitive": False},
}


class AbbreviationExpander:
    def __init__(self, custom_path: Optional[Path] = None):
        """
        custom_path: optionale JSON-Datei im selben Format wie
        DEFAULT_ABBREVIATIONS, um projektspezifische Kürzel zu ergänzen
        oder zu überschreiben, ohne den Code anzufassen.
        """
        self.abbreviations = dict(DEFAULT_ABBREVIATIONS)
        if custom_path and custom_path.exists():
            with open(custom_path, "r", encoding="utf-8") as f:
                self.abbreviations.update(json.load(f))

        self._compiled: List[Tuple[re.Pattern, List[str]]] = []
        for abbr, cfg in self.abbreviations.items():
            flags = 0 if cfg.get("case_sensitive", False) else re.IGNORECASE
            pattern = re.compile(
                rf'(?:(?<=^)|(?<=[{_BOUNDARY_CHARS}]))'
                + re.escape(abbr)
                + rf'(?=$|[{_BOUNDARY_CHARS}])',
                flags,
            )
            self._compiled.append((pattern, cfg["expansions"]))

        # Längere/spezifischere Kürzel zuerst prüfen (z.B. "Dipl.-Ing."
        # sollte nicht zusätzlich durch ein generisches "Ing."-Kürzel matchen).
        self._compiled.sort(key=lambda t: -t[0].pattern.count(re.escape("")))

    def expand(self, query: str) -> Tuple[str, bool]:
        """
        Gibt (erweiterte_query, wurde_erweitert) zurück. Erkannte
        Abkürzungen werden nicht ersetzt, sondern deren Auflösung(en)
        werden an die Query angehängt.
        """
        additions: List[str] = []
        seen = set()
        for pattern, expansions in self._compiled:
            if pattern.search(query):
                for exp in expansions:
                    if exp not in seen:
                        additions.append(exp)
                        seen.add(exp)

        if not additions:
            return query, False
        return f"{query} {' '.join(additions)}", True