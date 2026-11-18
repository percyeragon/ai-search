"""
Deutsche Kompositum-Zerlegung (Decompounding) für die Query-Preprocessing-Kette.

Motivation: BM25 arbeitet tokenbasiert. Eine Query wie "IT Sicherheit"
matcht ein Dokument-Token "Sicherheitsbeauftragter" nicht, obwohl es
semantisch genau passt -- das Kompositum ist für den Tokenizer ein
einziges, unteilbares Wort. CompoundSplitter zerlegt bekannte
Bestandteile aus zusammengesetzten Wörtern heraus und ERGÄNZT die Query
um diese Teile (das Original bleibt erhalten, damit exakte Matches nicht
verloren gehen).

Folgt demselben Interface wie AbbreviationExpander/SpellingVariantNormalizer/
BehoerdenNormalizer: expand_query(text) -> (expanded_text, was_expanded).

Wörterbuch-Strategie: Statt eines externen, domänenfremden deutschen
Wörterbuchs wird das Vokabular direkt aus dem Korpus (corpus.jsonl)
extrahiert. Das hat zwei Vorteile:
  1) keine zusätzliche Abhängigkeit / kein Offline-Wörterbuch-Download
  2) das Vokabular ist automatisch domänenpassend (Verwaltungs- und
     Stellenausschreibungs-Sprache), was Fehlsplits reduziert, die bei
     allgemeinsprachlichen Wörterbüchern häufig sind (z.B. "Rathaus"
     würde von einem generischen Splitter u.U. in "Rat" + "Haus" zerlegt,
     was hier unproblematisch ist, weil "Rathaus" ohnehin schon im
     Korpus-Vokabular als Ganzes vorkommt und nicht mehr gesplittet wird)

Optional kann zusätzlich ein manuell kuratiertes Wörterbuch (JSON-Liste)
gemergt werden, z.B. für bekannte Bestandteile, die im Korpus zu selten
vorkommen, um die Frequenzschwelle zu erreichen.

WICHTIG -- Reichweite dieser Komponente: Sie splittet nur die QUERY.
Für den umgekehrten Fall (Query "IT" soll ein Dokument-Token
"ITSicherheitsbeauftragter" treffen) müsste dieselbe Logik zusätzlich
beim Indexing angewendet werden (z.B. als zusätzlicher Tokenizer-Schritt
in BM25Retriever). split_text() ist bewusst als eigenständige,
wiederverwendbare Methode gehalten, damit sie später auch dort
eingehängt werden kann, ohne die Klasse zu duplizieren.

BUGFIX (5): Der Vokabular-Aufbau filterte bisher nur nach einer
MINDEST-Gesamthäufigkeit (`min_frequency`, Default 2). In einem Corpus aus
öffentlichen Stellenausschreibungen kommen hochgenerische Verwaltungs-
Morpheme wie "Land", "Amt", "Stadt", "Bau", "Dienst" aber nicht zweimal,
sondern hunderte Male vor (praktisch jede Behörde heißt "Land X", "Amt für
Y", "Stadt Z" ...). Wurde ein Kompositum über eines dieser Fragmente
gesplittet, hängte der Splitter sie an die Query an -- und weil diese
Fragmente in einem riesigen Anteil ALLER Dokumente vorkommen, matchten sie
praktisch jedes Dokument gleichermaßen. BM25 summiert Scores pro Query-
Term: ein extrem generischer zusätzlicher Term addiert dann Rauschen auf
fast alle Dokumente, statt zu differenzieren, und verwässert dadurch das
Ranking -- Dokumente, die eigentlich nur über generische Verwaltungswörter
"matchen", rutschen nach oben und verdrängen echte Treffer aus den Top-k.
Das erklärte den beobachteten Effekt, dass Preprocessing (`pp=True`) den
gemessenen Recall/nDCG über praktisch alle Retrieval-Basen hinweg (BM25,
Dense, Alpha, XGB) verschlechterte, statt ihn zu verbessern.

Der Fix ergänzt die Mindesthäufigkeit um eine Dokumentfrequenz-Obergrenze
(`max_doc_ratio`): ein Wort wird nur dann als gültiger Split-Bestandteil
ins Vokabular aufgenommen, wenn es zusätzlich in höchstens einem
bestimmten Anteil ALLER Dokumente vorkommt. Hochgenerische Morpheme fallen
dadurch heraus, während spezifische Fachbegriffe (die nur in einem
thematisch begrenzten Ausschnitt des Corpus vorkommen) erhalten bleiben.
"""

import json
import logging
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("compound_splitter")

# Deutsche Fugenelemente (Verbindungslaute zwischen Kompositum-Teilen),
# absteigend nach Häufigkeit grob geordnet. Leerstring zuerst, damit ein
# fugenloser Split bevorzugt wird, wenn beides möglich ist.
FUGENELEMENTE: Tuple[str, ...] = ("", "s", "es", "n", "en", "e", "ns", "er")

_WORD_PATTERN = re.compile(r"[A-Za-zÄÖÜäöüß]+")


class CompoundSplitter:
    """
    Zerlegt deutsche Komposita rekursiv in bekannte Bestandteile, basierend
    auf einem aus dem Korpus abgeleiteten Vokabular.

    Args:
        corpus: dict[doc_id, doc] wie in rest_service.py (PC.corpus) --
            das "text"-Feld jedes Dokuments wird zur Vokabular-Extraktion
            genutzt.
        custom_path: optionaler Pfad zu einer JSON-Datei mit zusätzlichen
            bekannten Wörtern, Format: {"words": ["wort1", "wort2", ...]}
        min_word_length: minimale Länge eines Teilworts, damit es als
            gültiger Split-Bestandteil zählt (verhindert Sinnlos-Splits
            wie "e" + "beauftragter")
        min_frequency: minimale Gesamthäufigkeit im Korpus, damit ein Wort
            ins Vokabular aufgenommen wird (filtert Tippfehler/Einzelvorkommen)
        max_doc_ratio: BUGFIX (5) -- Obergrenze für den Anteil der
            Dokumente, in denen ein Wort vorkommen darf, damit es als
            Split-Bestandteil zugelassen wird. Verhindert, dass
            hochgenerische Verwaltungs-Morpheme ("land", "amt", "stadt",
            "bau", "dienst" ...), die in einem Großteil aller
            Stellenausschreibungen auftauchen, als Split-Fragmente in die
            Query injiziert werden und dort praktisch jedes Dokument
            gleichermaßen treffen (Ranking-Rauschen statt Signal).
        max_split_depth: maximale Rekursionstiefe beim Zerlegen (verhindert
            pathologische Laufzeiten bei sehr langen Token)
        min_query_word_length: Query-Token unterhalb dieser Länge werden
            gar nicht erst zum Splitten versucht (kurze Wörter sind selten
            Komposita und das Risiko von Fehlsplits steigt)
    """

    def __init__(
        self,
        corpus: dict,
        custom_path: Optional[Path] = None,
        min_word_length: int = 4,
        min_frequency: int = 2,
        max_doc_ratio: float = 0.15,
        max_split_depth: int = 4,
        min_query_word_length: int = 8,
    ):
        self.min_word_length = min_word_length
        self.max_split_depth = max_split_depth
        self.min_query_word_length = min_query_word_length

        self._known_words: set = self._build_vocabulary(corpus, min_frequency, max_doc_ratio)

        if custom_path is not None and Path(custom_path).exists():
            extra = self._load_custom_words(custom_path)
            logger.info("Lade %d zusätzliche Wörter aus %s ...", len(extra), custom_path)
            self._known_words |= extra

        logger.info(
            "CompoundSplitter initialisiert: %d Vokabeln (min_frequency=%d, "
            "max_doc_ratio=%.2f, min_word_length=%d).",
            len(self._known_words), min_frequency, max_doc_ratio, min_word_length,
        )

        # Rekursions-Cache pro Instanz (Vokabular ändert sich nur bei Reload,
        # dann wird ohnehin eine neue Instanz gebaut -- siehe rest_service.py
        # build_pipeline_components()).
        self._split_cache: Dict[str, Optional[List[str]]] = {}

    # -----------------------------------------------------------------
    # Vokabular-Aufbau
    # -----------------------------------------------------------------

    @staticmethod
    def _load_custom_words(path: Path) -> set:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {w.lower() for w in data.get("words", []) if len(w) >= 2}

    def _build_vocabulary(self, corpus: dict, min_frequency: int, max_doc_ratio: float) -> set:
        """
        BUGFIX (5): zusätzlich zur bisherigen Gesamthäufigkeit (`term_counts`)
        wird jetzt auch gezählt, in wie vielen EINZELNEN Dokumenten ein Wort
        vorkommt (`doc_counts`, pro Dokument je Wort maximal einmal gezählt).
        Ein Wort muss beide Kriterien erfüllen, um ins Split-Vokabular
        aufgenommen zu werden:
          - term_counts[word] >= min_frequency  (wie bisher: kein Einzel-
            vorkommen/Tippfehler)
          - doc_counts[word] / n_docs <= max_doc_ratio  (NEU: kein
            hochgenerisches Wort, das ohnehin in fast jedem Dokument steht
            und daher kein Unterscheidungssignal liefert)
        """
        term_counts: Counter = Counter()
        doc_counts: Counter = Counter()
        n_docs = max(len(corpus), 1)

        for doc in corpus.values():
            text = doc.get("text")
            if not text:
                continue
            tokens_in_doc = set()
            for token in _WORD_PATTERN.findall(text):
                if len(token) >= self.min_word_length:
                    token_lower = token.lower()
                    term_counts[token_lower] += 1
                    tokens_in_doc.add(token_lower)
            for token_lower in tokens_in_doc:
                doc_counts[token_lower] += 1

        vocabulary = {
            word for word, freq in term_counts.items()
            if freq >= min_frequency
            and (doc_counts[word] / n_docs) <= max_doc_ratio
        }
        return vocabulary

    # -----------------------------------------------------------------
    # Split-Logik
    # -----------------------------------------------------------------

    def _find_all_splits(self, word: str, depth: int = 0) -> List[List[str]]:
        """Sammelt alle gültigen Zerlegungen von `word` in bekannte Teilwörter."""
        results: List[List[str]] = []

        if word in self._known_words and len(word) >= self.min_word_length:
            results.append([word])

        if depth >= self.max_split_depth:
            return results

        for i in range(self.min_word_length, len(word) - self.min_word_length + 1):
            prefix = word[:i]
            if prefix not in self._known_words:
                continue
            for fuge in FUGENELEMENTE:
                rest_start = i + len(fuge)
                if rest_start >= len(word):
                    continue
                if fuge and word[i:rest_start] != fuge:
                    continue
                rest = word[rest_start:]
                if len(rest) < self.min_word_length:
                    continue
                for sub_split in self._find_all_splits(rest, depth + 1):
                    results.append([prefix] + sub_split)

        return results

    @staticmethod
    def _score_split(split: List[str]) -> Tuple[int, int]:
        """
        Bewertet eine Zerlegung: weniger Teile ist besser (vermeidet
        Über-Segmentierung), bei Gleichstand ist ein höheres Minimum der
        Teilwort-Längen besser (vermeidet winzige Fragment-Wörter).
        Wird als sort-key mit aufsteigender Sortierung benutzt, daher
        die negative Mindestlänge.
        """
        return (len(split), -min(len(part) for part in split))

    def split_compound(self, word: str) -> Optional[List[str]]:
        """
        Versucht, ein einzelnes Wort in Kompositum-Bestandteile zu zerlegen.
        Gibt None zurück, wenn das Wort bereits ein bekanntes Einzelwort
        ist oder keine gültige Zerlegung gefunden wurde (dann macht ein
        Split ohnehin keinen Sinn).
        """
        word_lower = word.lower()

        if word_lower in self._split_cache:
            cached = self._split_cache[word_lower]
            return list(cached) if cached else None

        result: Optional[List[str]] = None

        # Bereits ein bekanntes Einzelwort -> kein Split nötig, das Wort
        # matcht im Index ohnehin direkt.
        if word_lower not in self._known_words and len(word_lower) >= self.min_word_length * 2:
            candidates = self._find_all_splits(word_lower)
            # Nur "echte" Zerlegungen (mehr als ein Teil) sind für uns
            # interessant -- ein Ein-Teil-"Split" bedeutet nur, dass das
            # Wort selbst im Vokabular ist, was der obige Fall aber schon
            # ausschließt (word_lower not in self._known_words).
            multi_part = [c for c in candidates if len(c) > 1]
            if multi_part:
                multi_part.sort(key=self._score_split)
                result = multi_part[0]

        self._split_cache[word_lower] = result
        return list(result) if result else None

    def split_text(self, text: str) -> List[str]:
        """
        Allgemeine Hilfsmethode: zerlegt jedes hinreichend lange Wort in
        `text` und gibt eine flache Liste aller zusätzlich gewonnenen
        Teilwörter zurück (ohne Duplikate, ohne die Originalwörter).
        Wiederverwendbar sowohl für Query-Preprocessing (expand_query)
        als auch potenziell für Indexing-seitige Tokenisierung.
        """
        extra_parts: List[str] = []
        seen = set()
        for token in _WORD_PATTERN.findall(text):
            if len(token) < self.min_query_word_length:
                continue
            parts = self.split_compound(token)
            if not parts:
                continue
            for part in parts:
                if part not in seen and part.lower() != token.lower():
                    seen.add(part)
                    extra_parts.append(part)
        return extra_parts

    # -----------------------------------------------------------------
    # Öffentliches Interface (konsistent mit den anderen Query-Preprocessing-
    # Komponenten in rest_service.py)
    # -----------------------------------------------------------------

    def expand_query(self, text: str) -> Tuple[str, bool]:
        """
        Erweitert die Query um die Bestandteile erkannter Komposita.
        Das Original bleibt vollständig erhalten (Bestandteile werden nur
        angehängt), damit exakte Kompositum-Matches nicht verloren gehen
        und nur zusätzliches Retrieval-Signal entsteht.

        Beispiel:
            "Sachbearbeiterin Bewerbungsfrist"
            -> "Sachbearbeiterin Bewerbungsfrist sach bearbeiterin
                bewerbung frist"
        """
        extra_parts = self.split_text(text)
        if not extra_parts:
            return text, False

        expanded = text + " " + " ".join(extra_parts)
        logger.info("Kompositum-Zerlegung: '%s' -> '%s'", text, expanded)
        return expanded, True