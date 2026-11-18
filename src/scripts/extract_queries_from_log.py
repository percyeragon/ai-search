#!/usr/bin/env python3
"""Extract queries from an Interamt search logfile and write a JSONL file.

Produces lines like {"_id": ..., "text": ..., "count": N, "params": {...}}
by default `params` is omitted unless `--include-params` is set.

Only lines with a non-empty `suchText` are considered (filter-only searches
without free text are ignored because they provide no textual signal).

Usage:
    python3 extract_queries_from_log.py --input gesamt.log --output queries.jsonl
    python3 extract_queries_from_log.py --input - --output queries.jsonl  # read stdin
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

LOG_LINE_RE = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ DEBUG \S+ \[(\S+)\] '
    r'Stellensuche \[(.*?)\] -> (\d+) Treffer\s*$'
)
PARAM_RE = re.compile(r"(\w+)=('(?:[^']*)'|\S+?)(?:,\s|$)")

# =====================================================
# Enum -> lesbares Label (siehe transform_dtag_to_beir.py fuer die
# entsprechenden Gegenstuecke auf Job-/Corpus-Seite)
# =====================================================

DIENSTVERHAELTNIS_LABELS = {
    'ARBEITNEHMER': 'Angestellter',
    'BEAMTER': 'Beamter',
    'AUSBILDUNGPRAKTIKUMDUALES_STUDIUM': 'Ausbildung, Praktikum oder duales Studium',
    'RUHESTAND': 'Ruhestand',
}

LAUFBAHNGRUPPE_LABELS = {
    'EINFACHER_DIENST': 'einfacher Dienst',
    'MITTLERER_DIENST': 'mittlerer Dienst',
    'GEHOBENER_DIENST': 'gehobener Dienst',
    'HOEHERER_DIENST': 'höherer Dienst',
    'ERSTES_EINSTIEGSAMT_ERSTE_LAUFBAHNGRUPPE': 'erstes Einstiegsamt, erste Laufbahngruppe',
    'ZWEITES_EINSTIEGSAMT_ERSTE_LAUFBAHNGRUPPE': 'zweites Einstiegsamt, erste Laufbahngruppe',
    'ERSTES_EINSTIEGSAMT_ZWEITE_LAUFBAHNGRUPPE': 'erstes Einstiegsamt, zweite Laufbahngruppe',
    'ZWEITES_EINSTIEGSAMT_ZWEITE_LAUFBAHNGRUPPE': 'zweites Einstiegsamt, zweite Laufbahngruppe',
    'LAUFBAHNGRUPPE_EINS': 'Laufbahngruppe eins',
    'LAUFBAHNGRUPPE_ZWEI': 'Laufbahngruppe zwei',
    'ERSTE_QUALIFIKATIONS_EBENE': 'erste Qualifikationsebene',
    'ZWEITE_QUALIFIKATIONS_EBENE': 'zweite Qualifikationsebene',
    'DRITTE_QUALIFIKATIONS_EBENE': 'dritte Qualifikationsebene',
    'VIERTE_QUALIFIKATIONS_EBENE': 'vierte Qualifikationsebene',
    'GRUPPE_A_UEBERGREIFEND': 'Gruppe A, übergreifend',
    'GRUPPEN_B': 'Gruppe B',
}

QUALIFIKATION_LABELS = {
    'SCHUELER': 'Schulabschluss (Schüler)',
    'QUALIFIZIERENDER_HAUPT_BZW_MITTELSCHULABSCHLUSS': 'qualifizierender Haupt- bzw. Mittelschulabschluss',
    'MITTLERE_REIFEMSA': 'mittlere Reife',
    'MITTLERE_REIFEMSA__BERUFSAUSBILDUNG': 'mittlere Reife mit Berufsausbildung',
    'FACHHOCHSCHULREIFE': 'Fachhochschulreife',
    'ABITUR': 'Abitur',
    'BACHELOR': 'Bachelor',
    'MASTER': 'Master',
}

TEILZEIT_LABELS = {
    'VOLLZEIT': 'Vollzeit',
    'TEILZEIT': 'Teilzeit',
    'BEIDES_MOEGLICH': 'Vollzeit oder Teilzeit',
}

DIENSTORT_LABELS = {
    'HYBRID': 'hybrid',
    'HOME_OFFICE': 'Home-Office',
    'VOR_ORT': 'vor Ort',
}

AUSBILDUNGEN_LABELS = {
    'DUALES_STUDIUM': 'duales Studium',
    'PRAKTIKUM': 'Praktikum',
    'AUSZUBILDENDER': 'Ausbildung',
    'BEAMTENANWAERTER': 'Beamtenanwärter',
}

BESCHAEFTIGUNGSDAUER_LABELS = {
    'UNBEFRISTET': 'unbefristet',
    'BEFRISTET': 'befristet',
    'BEFRISTET_OPTION_UNBEFRISTET': 'befristet mit Übernahmeoption',
}

PLZ_PREFIX_CITY = {
    '01': 'Dresden', '02': 'Bautzen', '03': 'Cottbus', '04': 'Leipzig',
    '06': 'Halle', '07': 'Gera', '08': 'Zwickau', '09': 'Chemnitz',
    '10': 'Berlin', '12': 'Berlin', '13': 'Berlin', '14': 'Berlin/Potsdam',
    '15': 'Frankfurt (Oder)', '16': 'Neuruppin', '17': 'Neubrandenburg',
    '18': 'Rostock', '19': 'Schwerin', '20': 'Hamburg', '21': 'Hamburg',
    '22': 'Hamburg', '23': 'Lübeck', '24': 'Kiel', '25': 'Itzehoe',
    '26': 'Oldenburg', '27': 'Bremerhaven', '28': 'Bremen', '29': 'Lüneburg',
    '30': 'Hannover', '31': 'Hildesheim', '32': 'Herford', '33': 'Bielefeld',
    '34': 'Kassel', '35': 'Gießen', '36': 'Fulda', '37': 'Göttingen',
    '38': 'Braunschweig', '39': 'Magdeburg', '40': 'Düsseldorf', '41': 'Mönchengladbach',
    '42': 'Wuppertal', '44': 'Dortmund', '45': 'Essen', '46': 'Oberhausen',
    '47': 'Duisburg', '48': 'Münster', '49': 'Osnabrück', '50': 'Köln',
    '51': 'Köln', '52': 'Aachen', '53': 'Bonn', '54': 'Trier',
    '55': 'Mainz', '56': 'Koblenz', '57': 'Siegen', '58': 'Hagen',
    '59': 'Hamm', '60': 'Frankfurt am Main', '61': 'Bad Homburg', '63': 'Offenbach',
    '64': 'Darmstadt', '65': 'Wiesbaden', '66': 'Saarbrücken', '67': 'Ludwigshafen',
    '68': 'Mannheim', '69': 'Heidelberg', '70': 'Stuttgart', '71': 'Ludwigsburg',
    '72': 'Tübingen', '73': 'Göppingen', '74': 'Heilbronn', '75': 'Pforzheim',
    '76': 'Karlsruhe', '77': 'Offenburg', '78': 'Villingen-Schwenningen', '79': 'Freiburg im Breisgau',
    '80': 'München', '81': 'München', '82': 'München', '83': 'Rosenheim',
    '84': 'Landshut', '85': 'München', '86': 'Augsburg', '87': 'Kempten',
    '88': 'Ravensburg', '89': 'Ulm', '90': 'Nürnberg', '91': 'Nürnberg',
    '92': 'Amberg', '93': 'Regensburg', '94': 'Passau', '95': 'Bayreuth',
    '96': 'Bamberg', '97': 'Würzburg', '98': 'Suhl', '99': 'Erfurt',
}


def plz_to_city(plz: str) -> Optional[str]:
    return PLZ_PREFIX_CITY.get(plz[:2]) if plz and len(plz.strip()) >= 2 else None


def multi_label(raw: str, mapping: Dict[str, str], joiner: str = ' oder ') -> str:
    parts = raw.split('|')
    labels = [mapping.get(p, p.replace('_', ' ').lower()) for p in parts]
    return joiner.join(labels)


def cluster_to_text(raw: str) -> str:
    return raw.replace('_', ' ').lower().replace('|', ' oder ')


def parse_log(input_source: Iterable[str]) -> List[Dict[str, str]]:
    """Parse lines from a logfile iterable and return a list of param dicts.

    Accepts any iterable of lines (file, stdin, list of strings).
    """
    rows: List[Dict[str, str]] = []
    for raw_line in input_source:
        line = raw_line.rstrip('\n')
        if not line.strip():
            continue
        m = LOG_LINE_RE.match(line)
        if not m:
            continue
        _ts, _thread, params_str, treffer = m.groups()
        if 'id=' in params_str:
            # Job detail fetch by concrete job ID, not a search
            continue
        params: Dict[str, str] = {}
        for pm in PARAM_RE.finditer(params_str):
            key, val = pm.groups()
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            params[key] = val
        try:
            params['treffer'] = int(treffer)
        except Exception:
            params['treffer'] = treffer
        rows.append(params)
    return rows


def build_text(row: Dict[str, str]) -> str:
    parts: List[str] = []

    such = (row.get('suchText') or '').strip()
    if such:
        parts.append(f'Ich suche nach "{such}".')

    if row.get('fachrichtungCluster'):
        parts.append(f"Fachrichtung: {cluster_to_text(row['fachrichtungCluster'])}.")

    if row.get('dienstverhaeltnisse'):
        parts.append(f"Beschäftigungsverhältnis: {multi_label(row['dienstverhaeltnisse'], DIENSTVERHAELTNIS_LABELS)}.")

    if row.get('ausbildungen'):
        parts.append(f"Gesucht: {multi_label(row['ausbildungen'], AUSBILDUNGEN_LABELS)}.")

    if row.get('laufbahnGruppe'):
        parts.append(f"Laufbahngruppe: {LAUFBAHNGRUPPE_LABELS.get(row['laufbahnGruppe'], row['laufbahnGruppe'])}.")

    qvon, qbis = row.get('qualifikationVon', ''), row.get('qualifikationBis', '')
    if qvon or qbis:
        von = QUALIFIKATION_LABELS.get(qvon, qvon)
        bis = QUALIFIKATION_LABELS.get(qbis, qbis)
        if von and bis and von != bis:
            parts.append(f"Qualifikation zwischen {von} und {bis}.")
        elif von or bis:
            parts.append(f"Qualifikation: {von or bis}.")

    bvon, bbis = row.get('besoldungVon', ''), row.get('besoldungBis', '')
    if bvon or bbis:
        if bvon and bbis:
            parts.append(f"Besoldungsgruppe von {bvon} bis {bbis}.")
        else:
            parts.append(f"Besoldungsgruppe {bvon or bbis}.")

    if row.get('teilzeit'):
        parts.append(f"Arbeitszeit: {TEILZEIT_LABELS.get(row['teilzeit'], row['teilzeit'])}.")

    if row.get('beschaeftigungsdauer'):
        parts.append(f"Beschäftigung: {BESCHAEFTIGUNGSDAUER_LABELS.get(row['beschaeftigungsdauer'], row['beschaeftigungsdauer'])}.")

    if row.get('dienstort'):
        parts.append(f"Arbeitsform: {multi_label(row['dienstort'], DIENSTORT_LABELS)}.")

    plz = (row.get('plz') or '').strip()
    entf = (row.get('maxEntfernung') or '').strip()
    if plz:
        city = plz_to_city(plz)
        if entf and entf != '1000km':
            if city:
                parts.append(f"Im Umkreis von {entf} um {city} (PLZ {plz}).")
            else:
                parts.append(f"Im Umkreis von {entf} um PLZ {plz}.")
        else:
            if city:
                parts.append(f"In der Nähe von {city} (PLZ {plz}).")
            else:
                parts.append(f"PLZ {plz}.")

    veroeff = (row.get('veroeffentlichtSeit') or '').strip()
    if veroeff and veroeff != '2000-01-01':
        parts.append(f"Veröffentlicht seit {veroeff}.")

    return ' '.join(parts)


def dedupe_key(row: Dict[str, str]) -> Tuple[str, ...]:
    """Eindeutige Kombination aus allen Suchparametern (ohne treffer)."""
    return tuple(sorted((k, v) for k, v in row.items() if k != 'treffer'))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, help='Pfad zum Logfile (z.B. gesamt.log)')
    ap.add_argument('--output', required=True, help='Pfad zur Ausgabe-queries.jsonl')
    ap.add_argument(
        '--min-count', type=int, default=1,
        help='Mindestanzahl an Vorkommen im Log, damit eine Query-Kombination '
             'aufgenommen wird (Default: 1 = alle, auch einmalige)'
    )
    ap.add_argument('--include-params', action='store_true', help='Include original search params in output JSON')
    ap.add_argument('--min-treffer', type=int, default=0, help='Ignore searches with fewer hits than this (default: 0)')
    ap.add_argument('--verbose', action='store_true', help='Verbose logging to stderr')
    args = ap.parse_args()

    input_path = Path(args.input)
    # Support reading from stdin when input is '-'
    if args.input == '-':
        if args.verbose:
            print("Lese von stdin ...", file=sys.stderr)
        lines = sys.stdin
    else:
        if not input_path.exists():
            sys.exit(f"FEHLER: Input-Datei nicht gefunden: {input_path}")
        if args.verbose:
            print(f"Lese {input_path} ...", file=sys.stderr)
        lines = open(input_path, encoding='utf-8', errors='replace')

    try:
        rows = parse_log(lines)
    finally:
        if args.input != '-':
            try:
                lines.close()
            except Exception:
                pass

    if args.verbose:
        print(f"Such-Zeilen (ohne Jobdetail-Abrufe): {len(rows)}", file=sys.stderr)

    # Only rows with real free text (suchText not empty)
    with_text = [r for r in rows if (r.get('suchText') or '').strip()]
    if args.verbose:
        print(f"Davon mit nicht-leerem suchText: {len(with_text)}", file=sys.stderr)

    # Optionally filter by min_treffer
    if args.min_treffer > 0:
        before = len(with_text)
        with_text = [r for r in with_text if int(r.get('treffer') or 0) >= args.min_treffer]
        if args.verbose:
            print(f"Filtered by min_treffer={args.min_treffer}: {before} -> {len(with_text)}", file=sys.stderr)

    # Count unique parameter combinations
    counts = Counter(dedupe_key(r) for r in with_text)
    unique_rows: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for r in with_text:
        unique_rows[dedupe_key(r)] = r

    kept = [(k, v) for k, v in unique_rows.items() if counts[k] >= args.min_count]
    kept.sort(key=lambda kv: (-counts[kv[0]], kv[0]))
    if args.verbose:
        print(f"Eindeutige Kombinationen (>= {args.min_count}x vorgekommen): {len(kept)}", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (key, row) in enumerate(kept, 1):
            qid_base = f"log_{i:05d}"
            # Clean query: remove surrounding/embedded quotes and trim
            raw_search = (row.get('suchText') or '').strip()
            query = raw_search.replace('"', '').replace("'", '').strip()

            # Build metadata including Ort (city) and Entfernung (distance)
            plz = (row.get('plz') or '').strip()
            city = plz_to_city(plz)
            entf = (row.get('maxEntfernung') or '').strip() or None

            metadata = {
                'postal_code': plz or None,
                'location': city,
                'radius': entf,
                'num_hits': row.get('treffer'),
                'specialization': row.get('fachrichtungCluster'),
                'employment_type': row.get('dienstverhaeltnisse'),
                'career_group': row.get('laufbahnGruppe'),
                'qualification_min': row.get('qualifikationVon'),
                'qualification_max': row.get('qualifikationBis'),
                'salary_min': row.get('besoldungVon'),
                'salary_max': row.get('besoldungBis'),
                'work_time': row.get('teilzeit'),
                'contract_duration': row.get('beschaeftigungsdauer'),
                'workplace': row.get('dienstort'),
                'published_since': row.get('veroeffentlichtSeit'),
            }

            repeats = int(counts[key]) if counts[key] and int(counts[key]) > 0 else 1
            for j in range(repeats):
                out_id = qid_base if repeats == 1 else f"{qid_base}_{j+1:02d}"
                out = {'_id': out_id, 'text': query or build_text(row), 'metadata': metadata}
                if args.include_params:
                    out['params'] = row
                f.write(json.dumps(out, ensure_ascii=False) + '\n')

    if args.verbose:
        print(f"Geschrieben: {output_path} ({len(kept)} Queries)", file=sys.stderr)


if __name__ == '__main__':
    main()