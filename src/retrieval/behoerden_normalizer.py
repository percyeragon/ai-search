"""
Behördensprache/Amtsdeutsch <-> Klartext Normalisierung (Phrasen-Ebene).

Ergänzt den bestehenden AbbreviationExpander (TVöD, EG, m/w/d, ...) um
Phrasen-Synonyme: Begriffe wie "Tätigkeitsmerkmale", "Sachbearbeitung",
"Eingruppierung" werden um Klartext-Äquivalente ergänzt (und umgekehrt),
damit ein Bewerberprofil in Alltagssprache ("Ich bearbeite Anträge") auch
gegen Amtsdeutsch-Ausschreibungen ("Sachbearbeitung", "Tätigkeitsmerkmale")
matcht.

Wird -- wie AbbreviationExpander/SpellingVariantNormalizer -- NUR auf die
Query angewendet. Der Korpus bleibt unverändert; die Query wird um Synonyme
aus derselben Gruppe erweitert (bidirektional), damit BM25/Dense trotz
unterschiedlichem Register einen Match finden.

Interface ist bewusst identisch zu SpellingVariantNormalizer gehalten:
    expand_query(text) -> (expanded_text, was_expanded)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Synonymgruppen: alle Begriffe einer Gruppe gelten als äquivalent.
# TODO: gegen Interamt-Korpus abgleichen und aus BEIR-Fehleranalyse
# (hohe Relevanz, niedriger BM25-Score -> Vokabular-Mismatch-Kandidat)
# iterativ erweitern. Neue Gruppen ohne Codeänderung über
# config/behoerdensprache.json nachladbar (siehe custom_path).
SYNONYM_GROUPS: List[List[str]] = [
    ["tätigkeitsmerkmale", "aufgabenanforderungen", "aufgabenprofil"],
    ["tätigkeitsschwerpunkt", "hauptaufgabe", "schwerpunktaufgabe"],
    ["sachbearbeitung", "sachbearbeiter", "sachbearbeiterin",
     "bearbeitung von vorgängen", "anträge bearbeiten"],
    ["eingruppierung", "einstufung", "gehaltseinstufung"],
    ["höherwertige tätigkeit", "anspruchsvollere aufgabe", "verantwortungsvollere tätigkeit"],
    ["einschlägige berufserfahrung", "relevante berufserfahrung", "passende vorerfahrung"],
    ["gehobener dienst", "gehobenen dienst", "bachelor-niveau", "fachhochschulabschluss erforderlich"],
    ["höherer dienst", "höheren dienst", "master-niveau", "wissenschaftliches hochschulstudium"],
    ["mittlerer dienst", "mittleren dienst", "abgeschlossene berufsausbildung"],
    ["einfacher dienst", "einfachen dienst", "keine ausbildung erforderlich", "angelernte tätigkeit"],
    ["verwaltungsfachangestellte", "verwaltungsfachangestellter",
     "verwaltungsfachkraft", "verwaltungsangestellte"],
    ["dienstherr", "arbeitgeber"],
    ["dienststelle", "amt", "arbeitsstelle", "behörde"],
    ["dienstverhältnis", "arbeitsverhältnis", "beschäftigungsverhältnis"],
    ["arbeitsvorgang", "arbeitsaufgabe", "vorgang"],
    ["stellenbewertung", "stellenbeschreibung", "aufgabenbewertung"],
    ["personalrat", "betriebsrat", "arbeitnehmervertretung"],
    ["fachliche einarbeitung", "einarbeitung", "schulung"],
    ["widerspruch", "einspruch", "beschwerde"],
    ["antragsteller", "antragstellerin", "bewerber", "bewerberin"],
    ["verbeamtung", "beamtenstatus", "verbeamtet werden"],
    ["referendariat", "vorbereitungsdienst", "ausbildung im öffentlichen dienst"],
]


class BehoerdenNormalizer:
    """
    Erweitert Query-Text um Amtsdeutsch<->Klartext Synonyme.
    Interface analog zu AbbreviationExpander/SpellingVariantNormalizer.
    """

    def __init__(self, custom_path: Union[str, Path, None] = None):
        self.groups: List[List[str]] = [list(g) for g in SYNONYM_GROUPS]
        if custom_path is not None:
            self._load_extra(Path(custom_path))

        # Lookup: Phrase (lowercase) -> Gruppen-Index
        self._phrase_to_group: Dict[str, int] = {}
        for idx, group in enumerate(self.groups):
            for phrase in group:
                self._phrase_to_group[phrase.lower()] = idx

        # Absteigend nach Länge sortiert, damit z.B. "gehobener dienst"
        # vor einem generischeren Teilstring geprüft wird.
        self._sorted_phrases = sorted(self._phrase_to_group.keys(), key=len, reverse=True)

    def _load_extra(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Zusätzliche Behördensprache-Konfiguration nicht gefunden: %s", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                extra_groups = json.load(f)
            if isinstance(extra_groups, list):
                self.groups.extend(extra_groups)
                logger.info(
                    "Behördensprache-Synonyme erweitert aus %s (+%d Gruppen)",
                    path, len(extra_groups),
                )
            else:
                logger.error("Ungültiges Format in %s: erwarte Liste von Synonymgruppen", path)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Fehler beim Laden von %s: %s", path, e)

    def expand_query(self, text: str) -> Tuple[str, bool]:
        """
        Scannt text nach bekannten Amtsdeutsch/Klartext-Phrasen und hängt
        alle übrigen Mitglieder derselben Gruppe an.
        """
        if not text:
            return text, False

        text_lower = text.lower()
        matched_groups: Set[int] = set()

        for phrase in self._sorted_phrases:
            if phrase in text_lower:
                matched_groups.add(self._phrase_to_group[phrase])

        if not matched_groups:
            return text, False

        additions = []
        for idx in matched_groups:
            for phrase in self.groups[idx]:
                if phrase not in text_lower and phrase not in additions:
                    additions.append(phrase)

        if not additions:
            return text, False

        return text + " " + " ".join(additions), True


if __name__ == "__main__":
    normalizer = BehoerdenNormalizer()

    query = "Ich habe Erfahrung in der Bearbeitung von Anträgen im gehobenen Dienst."
    expanded, changed = normalizer.expand_query(query)
    print(f"changed={changed}")
    print(expanded)