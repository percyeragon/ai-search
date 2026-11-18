"""
Erzeugt aus einem Benchmark-Lauf (src.scripts.benchmark_latency) einen PDF-Bericht.

Aufruf (aus dem Repo-Root):
    .env/Scripts/python -m src.scripts.benchmark_report results/benchmark/20261118_164249
    .env/Scripts/python -m src.scripts.benchmark_report <run_dir> --output bericht.pdf
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# =====================================================
# DESIGN-TOKENS (Referenzpalette, Light-Mode / Druck)
# =====================================================

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e6e5e1"
RULE = "#d4d3ce"
ACCENT = "#2a78d6"

# Kategoriale Slots in fester Reihenfolge (nie zyklisch).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Stufen-Gruppen für den gestapelten Balken. Reihenfolge = Ablauf in der Pipeline.
STAGE_GROUPS = [
    ("Preprocessing", SERIES[0]),
    ("Filter & Geo", SERIES[1]),
    ("Query Expansion", SERIES[2]),
    ("BM25 & Fusion", SERIES[3]),
    ("Dense: Query-Encoding", SERIES[6]),
    ("Dense: FAISS-Suche", SERIES[4]),
    ("Cross-Encoder", SERIES[7]),
    ("Ergebnisaufbau & Rest", "#a9a8a2"),
]

SCENARIO_LABELS = {
    "bm25:no_preproc": "bm25 (ohne Preprocessing)",
    "rrf:geo_filter": "rrf + Geo-Filter",
    "rrf:cat_filter": "rrf + Kategorie-Filter",
    "rrf:geo_filter+sort_distance": "rrf + Geo-Filter + Sort. Distanz",
}

SCENARIO_DESCRIPTIONS = [
    ("bm25", "Stichwortsuche (BM25) mit vollem Query-Preprocessing"),
    ("bm25 (ohne Preprocessing)", "BM25, alle Preprocessing-Schritte deaktiviert"),
    ("dense", "Semantische Vektorsuche (bge-m3 + FAISS)"),
    ("rrf", "BM25 + Dense, Reciprocal Rank Fusion"),
    ("alpha", "BM25 + Dense, gewichtete Min-Max-Fusion (α = 0,2)"),
    ("bm25+qe", "BM25 mit Query Expansion"),
    ("dense+ce / rrf+ce", "wie oben, zusätzlich Cross-Encoder-Reranking (Top 20)"),
    ("rrf+qe+ce", "volle Kette: Query Expansion, RRF, Cross-Encoder"),
    ("rrf + Geo-Filter", "RRF, Umkreis 50 km um PLZ 10115 (Berlin)"),
    ("rrf + Kategorie-Filter", "RRF, Filter beschaeftigungsart = Vollzeit"),
    ("rrf + Geo + Sort. Distanz", "wie Geo-Filter, Ergebnisse nach Entfernung sortiert"),
]

STARTUP_LABELS = {
    "corpus_load": "Korpus laden",
    "spell_corrector": "Rechtschreibkorrektur (SymSpell)",
    "abbreviation_expander": "Abkürzungserkennung",
    "behoerden_normalizer": "Behördensprache",
    "variant_normalizer": "Schreibvarianten",
    "compound_splitter": "Komposita-Splitter",
    "bm25_index": "BM25-Index",
    "dense_model+index": "Dense: Modell + FAISS-Index",
    "query_expander": "Query Expansion",
    "cross_encoder_model": "Cross-Encoder-Modell",
    "xgb_model": "XGB-Modell",
    "geo_cache": "Geo-Cache",
    "total": "Gesamt",
}


def label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


# =====================================================
# DATEN AUFBEREITEN
# =====================================================

def stage_mean(summary: dict, stage: str) -> float:
    return summary["stages"].get(stage, {}).get("mean", 0.0)


def group_breakdown(summary: dict) -> list[float]:
    pre = sum(stage_mean(summary, s) for s in summary["stages"] if s.startswith("pre."))
    filt = stage_mean(summary, "filter") + stage_mean(summary, "geo_distances")
    qe = stage_mean(summary, "query_expansion")
    enc = stage_mean(summary, "dense.query_encode")
    faiss = stage_mean(summary, "dense.faiss_search")
    retrieval_rest = max(stage_mean(summary, "retrieval_fusion") - enc - faiss, 0.0)
    ce = stage_mean(summary, "cross_encoder")
    rest = sum(stage_mean(summary, s) for s in ("result_building", "sort", "other"))
    return [pre, filt, qe, retrieval_rest, enc, faiss, ce, rest]


# =====================================================
# DIAGRAMME
# =====================================================

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def set_log_axis(ax, values, pad_lo=0.7, pad_hi=1.5):
    lo, hi = min(values) * pad_lo, max(values) * pad_hi
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000) if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.xaxis.set_minor_locator(NullLocator())


def chart_breakdown(summaries: list[dict], path: Path):
    order = sorted(summaries, key=lambda s: s["stages"]["total"]["mean"])
    names = [label(s["scenario"]) for s in order]
    data = [group_breakdown(s) for s in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    y = range(len(order))
    left = [0.0] * len(order)
    for gi, (gname, color) in enumerate(STAGE_GROUPS):
        widths = [row[gi] for row in data]
        if not any(w > 0.05 for w in widths):
            continue
        ax.barh(
            y, widths, left=left, height=0.62, color=color, label=gname,
            edgecolor=SURFACE, linewidth=1.0,
        )
        left = [l + w for l, w in zip(left, widths)]

    x_max = max(s["stages"]["total"]["mean"] for s in order)
    for yi, s in zip(y, order):
        total = s["stages"]["total"]["mean"]
        ax.text(total + x_max * 0.012, yi, f"{fmt(total)} ms", va="center", fontsize=7.5, color=TEXT_PRIMARY)

    ax.set_yticks(list(y))
    ax.set_yticklabels(names, color=TEXT_PRIMARY)
    ax.set_xlabel("Mittlere Latenz pro Anfrage [ms]", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, x_max * 1.14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.42, -0.12), ncol=4, frameon=False,
        fontsize=7.5, labelcolor=TEXT_PRIMARY, handlelength=1.0, columnspacing=1.2,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def chart_percentiles(summaries: list[dict], path: Path):
    order = sorted(summaries, key=lambda s: s["stages"]["total"]["p50"])
    names = [label(s["scenario"]) for s in order]
    p50 = [s["stages"]["total"]["p50"] for s in order]
    p95 = [s["stages"]["total"]["p95"] for s in order]
    p99 = [s["stages"]["total"]["p99"] for s in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    y = list(range(len(order)))

    for yi, a, c in zip(y, p50, p99):
        ax.plot([a, c], [yi, yi], color=RULE, linewidth=2, solid_capstyle="round", zorder=1)
    for vals, color, name, marker in (
        (p50, SERIES[0], "p50 (Median)", "o"),
        (p95, SERIES[1], "p95", "D"),
        (p99, SERIES[2], "p99", "s"),
    ):
        ax.scatter(vals, y, s=34, color=color, label=name, marker=marker,
                   edgecolors=SURFACE, linewidths=1.2, zorder=3)

    for yi, v in zip(y, p95):
        ax.text(v, yi + 0.32, fmt(v), ha="center", va="bottom", fontsize=6.5, color=TEXT_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(names, color=TEXT_PRIMARY)
    ax.set_ylim(-0.6, len(order) - 0.3)
    if max(p99) / max(min(p50), 0.1) > 50:
        set_log_axis(ax, p50 + p99, pad_hi=1.3)
        ax.set_xlabel("Latenz pro Anfrage [ms], logarithmische Skala", fontsize=8, color=TEXT_SECONDARY)
    else:
        ax.set_xlim(0, max(p99) * 1.08)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel("Latenz pro Anfrage [ms]", fontsize=8, color=TEXT_SECONDARY)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def device_label(meta: dict) -> str:
    return "GPU" if meta.get("gpu") and meta.get("device") == "cuda" else "CPU"


def chart_compare(summaries, other, this_name, other_name, path: Path):
    """Dumbbell auf log-Skala: gleiche Szenarien, zwei Geräte."""
    O = {s["scenario"]: s for s in other}
    common = [s for s in summaries if s["scenario"] in O]
    order = sorted(common, key=lambda s: s["stages"]["total"]["mean"])
    names = [label(s["scenario"]) for s in order]
    a = [s["stages"]["total"]["mean"] for s in order]
    b = [O[s["scenario"]]["stages"]["total"]["mean"] for s in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    y = list(range(len(order)))
    for yi, va, vb in zip(y, a, b):
        ax.plot([va, vb], [yi, yi], color=RULE, linewidth=2, solid_capstyle="round", zorder=1)
    ax.scatter(a, y, s=36, color=SERIES[0], label=this_name, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.scatter(b, y, s=36, color=SERIES[1], marker="D", label=other_name, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    for yi, va, vb in zip(y, a, b):
        hi = max(va, vb)
        ax.text(hi * 1.12, yi, f"×{fmt(max(va, vb) / min(va, vb))}", va="center", fontsize=6.8, color=TEXT_SECONDARY)
    set_log_axis(ax, a + b, pad_hi=2.2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=TEXT_PRIMARY)
    ax.set_xlabel("Mittlere Latenz pro Anfrage [ms], logarithmische Skala", fontsize=8, color=TEXT_SECONDARY)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# =====================================================
# PDF
# =====================================================

def register_fonts():
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(font_dir / "DejaVuSans-Oblique.ttf")))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily

    registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique")
    plt.rcParams["font.family"] = "DejaVu Sans"


def make_styles():
    base = dict(fontName="DejaVu", textColor=colors.HexColor(TEXT_PRIMARY), alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("title", fontSize=20, leading=25, fontName="DejaVu-Bold", **{k: v for k, v in base.items() if k != "fontName"}),
        "subtitle": ParagraphStyle("subtitle", fontSize=10, leading=14, textColor=colors.HexColor(TEXT_SECONDARY), fontName="DejaVu"),
        "h1": ParagraphStyle("h1", fontSize=13, leading=17, spaceBefore=10, spaceAfter=5, fontName="DejaVu-Bold", textColor=colors.HexColor(TEXT_PRIMARY)),
        "h2": ParagraphStyle("h2", fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=3, fontName="DejaVu-Bold", textColor=colors.HexColor(TEXT_PRIMARY)),
        "body": ParagraphStyle("body", fontSize=9, leading=13, spaceAfter=4, **base),
        "small": ParagraphStyle("small", fontSize=7.5, leading=10, textColor=colors.HexColor(TEXT_SECONDARY), fontName="DejaVu"),
        "caption": ParagraphStyle("caption", fontSize=7.5, leading=10, spaceBefore=2, spaceAfter=8, textColor=colors.HexColor(TEXT_SECONDARY), fontName="DejaVu-Oblique"),
        "cell": ParagraphStyle("cell", fontSize=8, leading=10, **base),
        "th": ParagraphStyle("th", fontSize=7.5, leading=9, fontName="DejaVu-Bold", textColor=colors.HexColor(TEXT_SECONDARY), alignment=2),
        "th_left": ParagraphStyle("th_left", fontSize=7.5, leading=9, fontName="DejaVu-Bold", textColor=colors.HexColor(TEXT_SECONDARY)),
        "cell_bold": ParagraphStyle("cell_bold", fontSize=8, leading=10, fontName="DejaVu-Bold", textColor=colors.HexColor(TEXT_PRIMARY)),
        "kpi_value": ParagraphStyle("kpi_value", fontSize=17, leading=20, fontName="DejaVu-Bold", textColor=colors.HexColor(TEXT_PRIMARY)),
        "kpi_label": ParagraphStyle("kpi_label", fontSize=7.5, leading=10, textColor=colors.HexColor(TEXT_SECONDARY), fontName="DejaVu"),
    }


def table_style(header_rows=1, numeric_from_col=1):
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "DejaVu-Bold"),
        ("FONTNAME", (0, header_rows), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.HexColor(TEXT_SECONDARY)),
        ("TEXTCOLOR", (0, header_rows), (-1, -1), colors.HexColor(TEXT_PRIMARY)),
        ("ALIGN", (numeric_from_col, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, colors.HexColor(TEXT_SECONDARY)),
        ("LINEBELOW", (0, header_rows), (-1, -2), 0.3, colors.HexColor(GRID)),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ])


def bullets(items, style):
    return ListFlowable(
        [ListItem(Paragraph(t, style), leftIndent=10, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8,
        bulletColor=colors.HexColor(TEXT_SECONDARY),
    )


def kpi_row(kpis, st, width):
    cells = [[Paragraph(v, st["kpi_value"]) for v, _ in kpis], [Paragraph(l, st["kpi_label"]) for _, l in kpis]]
    t = Table(cells, colWidths=[width / len(kpis)] * len(kpis))
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 2, colors.HexColor(ACCENT)),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def by_name(summaries):
    return {s["scenario"]: s for s in summaries}


def build_pdf(run_dir: Path, output: Path, cpu_name: str | None, compare_dir: Path | None = None):
    data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    meta, startup, summaries = data["meta"], data["startup_seconds"], data["scenarios"]
    S = by_name(summaries)

    register_fonts()
    st = make_styles()

    chart_dir = run_dir / "charts"
    chart_dir.mkdir(exist_ok=True)
    chart_breakdown(summaries, chart_dir / "breakdown.png")
    chart_percentiles(summaries, chart_dir / "percentiles.png")

    page_w, page_h = A4
    margin = 18 * mm
    content_w = page_w - 2 * margin - 12  # Frame-Padding (2 x 6 pt)

    run_date = datetime.fromisoformat(meta["timestamp"])

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(SURFACE))
        canvas.rect(0, 0, page_w, page_h, stroke=0, fill=1)
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(colors.HexColor(TEXT_MUTED))
        canvas.drawString(margin, 10 * mm, f"ai-suche · Laufzeittest Suchpipeline · {run_date:%d.%m.%Y}")
        canvas.drawRightString(page_w - margin, 10 * mm, f"Seite {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=margin, rightMargin=margin,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title="Laufzeittest Suchpipeline – ai-suche", author="ai-suche",
    )

    on_gpu = bool(meta.get("gpu")) and meta.get("device") == "cuda"

    def get(name, stage="total", stat="mean"):
        return S[name]["stages"][stage][stat] if name in S and stage in S[name]["stages"] else None

    story = []

    # ---------- Titel ----------
    story += [
        Paragraph("Laufzeittest der Suchpipeline", st["title"]),
        Spacer(1, 2),
        Paragraph(
            f"ai-suche · Messung vom {run_date:%d.%m.%Y, %H:%M} Uhr · "
            f"{meta['n_queries']} echte Suchanfragen · {thousands(meta['corpus_docs'])} Stellenausschreibungen",
            st["subtitle"],
        ),
        Spacer(1, 10),
    ]

    kpis = []
    if get("bm25"):
        kpis.append((f"{fmt(get('bm25'))} ms", "BM25 (mit Preprocessing), Mittelwert"))
    if get("rrf"):
        kpis.append((f"{fmt(get('rrf'))} ms", "Hybrid RRF, Mittelwert"))
    if get("rrf+ce"):
        kpis.append((f"{fmt(get('rrf+ce'))} ms", "RRF + Cross-Encoder, Mittelwert"))
    if get("rrf+qe+ce"):
        kpis.append((f"{fmt(get('rrf+qe+ce', stat='p99'))} ms", "p99 der teuersten Konfiguration"))
    story += [kpi_row(kpis, st, content_w), Spacer(1, 12)]

    # ---------- Kurzfazit ----------
    enc = get("dense", "dense.query_encode")
    faiss = get("dense", "dense.faiss_search")
    ce = get("rrf+ce", "cross_encoder")
    pre_bm25 = get("bm25")
    raw_bm25 = get("bm25:no_preproc")
    cat_f = get("rrf:cat_filter", "filter")
    geo_f = (get("rrf:geo_filter", "filter") or 0) + (get("rrf:geo_filter", "geo_distances") or 0) + (get("rrf:geo_filter", "other") or 0) - (get("rrf", "other") or 0)

    story.append(Paragraph("Kurzfazit", st["h1"]))
    max_p99 = max(s["stages"]["total"]["p99"] for s in summaries)
    slowest = max(summaries, key=lambda s: s["stages"]["total"]["mean"])
    if max_p99 < 100:
        headline = "<b>Alle Konfigurationen liegen unter 100 ms</b> (p99)."
    else:
        under = [label(s["scenario"]) for s in summaries if s["stages"]["total"]["p99"] < 100]
        headline = (
            f"<b>Die Spannweite reicht von {fmt(pre_bm25)} ms bis {fmt(slowest['stages']['total']['mean'])} ms</b> "
            f"im Mittel; unter 100 ms (p99) bleiben {len(under)} von {len(summaries)} Konfigurationen."
        )
    findings = [
        f"{headline} Die schnellste Variante (BM25) braucht im Mittel {fmt(pre_bm25)} ms, "
        f"die vollständige Kette mit Query Expansion, Hybrid-Fusion und Cross-Encoder "
        f"{fmt(get('rrf+qe+ce'))} ms.",
    ]
    if enc and faiss:
        findings.append(
            f"<b>{'Haupttreiber ist' if not ce or ce < enc else 'Ohne Cross-Encoder ist der Haupttreiber'} das Query-Encoding mit bge-m3</b>: rund {fmt(enc)} ms pro Anfrage, "
            f"während die eigentliche FAISS-Vektorsuche nur {fmt(faiss)} ms kostet. Jede Konfiguration "
            f"mit Dense-Anteil (dense, rrf, alpha) zahlt diesen Fixbetrag."
        )
    if ce and get("rrf"):
        factor = get("rrf+ce") / get("rrf")
        findings.append(
            f"<b>Cross-Encoder-Reranking kostet +{fmt(ce)} ms</b> für das Neubewerten von 20 "
            f"Kandidaten – die Latenz von RRF steigt damit um den Faktor {fmt(factor)}."
        )
    if pre_bm25 and raw_bm25:
        findings.append(
            f"<b>Query-Preprocessing ist günstig</b> (im Mittel unter 1 ms, p95 rund 3 ms), fällt "
            f"aber bei reinem BM25 ins Gewicht: {fmt(raw_bm25)} ms → {fmt(pre_bm25)} ms. "
            f"Fast die gesamte Zeit entfällt auf die Rechtschreibkorrektur."
        )
    if cat_f:
        findings.append(
            f"<b>Filter haben Optimierungspotenzial</b>: Der Kategorie-Filter kostet {fmt(cat_f)} ms "
            f"(Python-Schleife über alle Dokumente pro Anfrage), der Geo-Filter rund {fmt(geo_f)} ms, "
            f"weil Ortsauflösung und Distanzberechnung doppelt laufen."
        )
    story.append(bullets(findings, st["body"]))

    # ---------- Testaufbau ----------
    story.append(Paragraph("Testaufbau", st["h1"]))
    setup_rows = [
        ["Messobjekt", "rest_service.search() – derselbe Code wie der Endpoint POST /search"],
        ["Messart", "In-Process, sequentiell, ohne HTTP-Overhead; jede Stufe einzeln instrumentiert"],
        ["Anfragen", f"{meta['n_queries']} zufällig gezogene, eindeutige Suchanfragen aus dem Interamt-Log "
                     f"(Seed {meta['seed']}), {meta['warmup']} Warm-up-Anfragen vorab"],
        ["Caches", "Query-Embedding- und Cross-Encoder-Cache vor jedem Szenario geleert (kalte Anfragen)"
                   if not meta.get("keep_caches") else "Caches nach Warm-up beibehalten"],
        ["Parameter", f"top_k = {meta['top_k']}, candidate_k = {meta['candidate_k']}, Cross-Encoder auf Top 20"],
        ["Korpus", f"{thousands(meta['corpus_docs'])} Stellenausschreibungen (runtime-data/corpus.jsonl)"],
        ["Modelle", "BAAI/bge-m3 (Dense), BAAI/bge-reranker-v2-m3 (Cross-Encoder"
                    + (", FP16)" if on_gpu else ", FP32)")],
        ["Hardware", (f"GPU: {meta['gpu']}" if meta.get("gpu") else "nur CPU (GPU deaktiviert)")
                     + (f" · CPU: {cpu_name}" if cpu_name else "")],
        ["Software", f"Python {meta['python']}, PyTorch {meta['torch']}, Gerät: {meta['device']}"],
    ]
    t = Table(
        [[Paragraph(a, st["cell_bold"]), Paragraph(b, st["cell"])] for a, b in setup_rows],
        colWidths=[28 * mm, content_w - 28 * mm],
    )
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor(GRID)),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t)

    story.append(Paragraph("Gemessene Stufen (entsprechen der Service-Architektur)", st["h2"]))
    story.append(Paragraph(
        "Anfrage → <b>Preprocessing</b> (Rechtschreibkorrektur, Abkürzungen, Behördensprache, "
        "Schreibvarianten, Komposita) → <b>Filter</b> (Kategorien, Geo-Radius) → "
        "<b>Query Expansion</b> (optional) → <b>Retrieval &amp; Fusion</b> (BM25, Dense-Encoding, "
        "FAISS-Suche, RRF/Alpha) → <b>Cross-Encoder</b> (optional) → <b>Ergebnisaufbau</b> und Sortierung.",
        st["body"],
    ))

    # ---------- Ergebnisse ----------
    story.append(PageBreak())
    story.append(Paragraph("Wo die Zeit hingeht", st["h1"]))
    story.append(Paragraph(
        "Mittlere Latenz je Szenario, aufgeteilt auf die Stufen der Pipeline. "
        "Die Balken sind nach Gesamtlatenz sortiert.", st["body"],
    ))
    story.append(Image(str(chart_dir / "breakdown.png"), width=content_w, height=content_w * 4.1 / 7.2))
    story.append(Paragraph(
        "Abb. 1 – Mittlere Latenz pro Anfrage nach Pipeline-Stufe. „BM25 &amp; Fusion“ umfasst "
        "BM25-Scoring sowie das Zusammenführen der Ranglisten.", st["caption"],
    ))

    story.append(KeepTogether([Paragraph("Streuung: Median und Ausreißer", st["h1"]), Paragraph(
        "Median (p50), p95 und p99 der Gesamtlatenz. Ein großer Abstand zwischen p50 und p95 zeigt, "
        "dass einzelne Anfragen deutlich länger dauern – bei BM25 und Query Expansion typischerweise "
        "lange oder häufige Suchbegriffe.", st["body"],
    ),
        Image(str(chart_dir / "percentiles.png"), width=content_w, height=content_w * 3.5 / 7.2),
        Paragraph("Abb. 2 – Perzentile der Gesamtlatenz je Szenario (beschriftet: p95).", st["caption"])]))

    # ---------- Tabelle ----------
    story.append(PageBreak())
    story.append(Paragraph("Alle Ergebnisse im Überblick", st["h1"]))
    header = ["Mittel", "p50", "p95", "p99", "max", "Anfr./s", "Ø Treffer"]
    rows = [[Paragraph("Szenario", st["th_left"])] + [Paragraph(h, st["th"]) for h in header]]
    for s in sorted(summaries, key=lambda x: x["stages"]["total"]["mean"]):
        tot = s["stages"]["total"]
        rows.append([
            label(s["scenario"]), fmt(tot["mean"]), fmt(tot["p50"]), fmt(tot["p95"]),
            fmt(tot["p99"]), fmt(tot["max"]), fmt(s["throughput_qps_sequential"], 0), fmt(s["mean_results"]),
        ])
    t = Table(rows, colWidths=[54 * mm] + [(content_w - 54 * mm) / 7] * 7, repeatRows=1)
    t.setStyle(table_style())
    story.append(t)
    story.append(Paragraph(
        "Latenzen in Millisekunden. „Anfr./s“ = 1000 / Mittelwert, also Durchsatz eines einzelnen "
        "Prozesses bei sequentieller Verarbeitung. Der Geo-Filter liefert im Schnitt weniger als 10 "
        "Treffer, weil im Umkreis nicht für jede Anfrage genug passende Stellen existieren.",
        st["caption"],
    ))

    # Stufen-Detail
    story.append(Paragraph("Stufen im Detail (Mittelwert in ms)", st["h2"]))
    detail_cols = [
        ("Pre-<br/>proc.", lambda s: sum(stage_mean(s, k) for k in s["stages"] if k.startswith("pre."))),
        ("Filter/<br/>Geo", lambda s: stage_mean(s, "filter") + stage_mean(s, "geo_distances")),
        ("QE", lambda s: stage_mean(s, "query_expansion")),
        ("BM25", lambda s: stage_mean(s, "bm25.scoring")),
        ("Dense-<br/>Enc.", lambda s: stage_mean(s, "dense.query_encode")),
        ("FAISS", lambda s: stage_mean(s, "dense.faiss_search")),
        ("Cross-<br/>Enc.", lambda s: stage_mean(s, "cross_encoder")),
        ("Gesamt", lambda s: stage_mean(s, "total")),
    ]
    rows = [[Paragraph("Szenario", st["th_left"])] + [Paragraph(c, st["th"]) for c, _ in detail_cols]]
    for s in sorted(summaries, key=lambda x: x["stages"]["total"]["mean"]):
        rows.append([label(s["scenario"])] + [
            (fmt(fn(s)) if fn(s) >= 0.05 else "–") for _, fn in detail_cols
        ])
    t = Table(rows, colWidths=[54 * mm] + [(content_w - 54 * mm) / 8] * 8, repeatRows=1)
    ts = table_style()
    ts.add("FONTNAME", (-1, 1), (-1, -1), "DejaVu-Bold")
    t.setStyle(ts)
    story.append(t)
    story.append(Paragraph(
        "Dense-Enc. = Query-Encoding mit bge-m3, Cross-Enc. = Cross-Encoder. „BM25“ enthält bei Query Expansion auch den zusätzlichen BM25-Aufruf der Expansion. "
        "Nicht aufgeführt: Ergebnisaufbau und Sortierung (zusammen &lt; 0,5 ms).", st["caption"],
    ))

    # Szenarien-Legende
    story.append(KeepTogether([
        Paragraph("Szenarien", st["h2"]),
        Table(
            [[Paragraph(a, st["cell_bold"]), Paragraph(b, st["cell"])] for a, b in SCENARIO_DESCRIPTIONS],
            colWidths=[50 * mm, content_w - 50 * mm],
            style=TableStyle([
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor(GRID)),
                ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]),
        ),
    ]))

    # ---------- Optimierung / Startup / Grenzen ----------

    # ---------- Vergleich mit anderem Lauf ----------
    if compare_dir is not None:
        cmp = json.loads((compare_dir / "summary.json").read_text(encoding="utf-8"))
        this_name, other_name = device_label(meta), device_label(cmp["meta"])
        if this_name == other_name:
            this_name, other_name = "dieser Lauf", f"Lauf {compare_dir.name}"
        chart_compare(summaries, cmp["scenarios"], this_name, other_name, chart_dir / "compare.png")
        O = {s["scenario"]: s for s in cmp["scenarios"]}
        block = [
            Paragraph(f"Vergleich {this_name} vs. {other_name}", st["h1"]),
            Paragraph(
                f"Dieselben Szenarien und Anfragen, einmal mit {this_name} und einmal mit {other_name} "
                f"(Vergleichslauf: {compare_dir.as_posix()}). Der Faktor gibt an, um wie viel die "
                "langsamere Variante länger braucht.", st["body"],
            ),
            Image(str(chart_dir / "compare.png"), width=content_w, height=content_w * 3.6 / 7.2),
            Paragraph("Abb. 3 – Mittlere Latenz je Szenario im Gerätevergleich (log-Skala).", st["caption"]),
        ]
        rows = [[Paragraph("Szenario", st["th_left"])] + [Paragraph(h, st["th"]) for h in (
            f"{this_name}<br/>Mittel", f"{this_name}<br/>p95", f"{other_name}<br/>Mittel", f"{other_name}<br/>p95", "Faktor<br/>(Mittel)")]]
        for s in sorted(summaries, key=lambda x: x["stages"]["total"]["mean"]):
            o = O.get(s["scenario"])
            if not o:
                continue
            ta, oa = s["stages"]["total"], o["stages"]["total"]
            rows.append([label(s["scenario"]), fmt(ta["mean"]), fmt(ta["p95"]), fmt(oa["mean"]), fmt(oa["p95"]),
                         "×" + fmt(max(ta["mean"], oa["mean"]) / min(ta["mean"], oa["mean"]))])
        t = Table(rows, colWidths=[54 * mm] + [(content_w - 54 * mm) / 5] * 5, repeatRows=1)
        ts = table_style()
        ts.add("FONTNAME", (-1, 1), (-1, -1), "DejaVu-Bold")
        t.setStyle(ts)
        block.append(t)
        block.append(Paragraph("Latenzen in Millisekunden. Faktor = Mittelwert der langsameren / der schnelleren Variante.", st["caption"]))
        story.append(PageBreak())
        story.extend(block)

    story.append(Paragraph("Optimierungspotenzial", st["h1"]))
    opt = [
        "<b>Kategorie-Filter vorberechnen</b>: Statt pro Anfrage alle Dokumente in Python zu prüfen, "
        "beim Start einen invertierten Index (Feldwert → Menge von doc_ids) aufbauen; die Filterung "
        f"wird dann zur Mengen-Schnittmenge (geschätzt {fmt(cat_f)} ms → deutlich unter 1 ms).",
        "<b>Geo-Berechnung nur einmal ausführen</b>: get_allowed_doc_ids() und "
        "_compute_min_distances_km() lösen PLZ/Ort jeweils neu auf und berechnen die Distanzen doppelt. "
        "Einmal berechnen und weiterreichen spart einige Millisekunden.",
        "<b>Query-Encoding</b>: Der Query-Embedding-Cache greift bei wiederkehrenden Anfragen "
        "(in diesem Test bewusst geleert). Für kalte Anfragen helfen "
        + ("FP16 auf der GPU oder ein kleineres Encoder-Modell." if on_gpu else
           "eine GPU, ein kleineres Encoder-Modell oder ein quantisiertes ONNX-Modell; auf der CPU "
           "ist dies neben dem Cross-Encoder die kritische Stufe."),
        f"<b>Cross-Encoder</b> nur dort einsetzen, wo der Qualitätsgewinn die rund {fmt(ce or 0, 0)} ms "
        "rechtfertigt, oder die Kandidatenzahl (derzeit 20) reduzieren.",
    ]
    story.append(bullets(opt, st["body"]))

    story.append(Paragraph("Startzeiten des Dienstes", st["h1"]))
    story.append(Paragraph(
        "Zeit für den Aufbau der Komponenten beim Start (wie build_pipeline_components()). "
        "Gemessen bei vorhandenem Embedding-Cache; beim allerersten Start kommen Modell-Downloads "
        "und das einmalige Encoding des Korpus hinzu.", st["body"],
    ))
    rows = [[Paragraph("Komponente", st["th_left"]), Paragraph("Dauer [s]", st["th"])]]
    small = []
    for k, v in startup.items():
        if k == "total":
            continue
        if v < 0.1:
            small.append(STARTUP_LABELS.get(k, k))
            continue
        rows.append([STARTUP_LABELS.get(k, k), fmt(v, 2)])
    if small:
        rows.append([Paragraph("Übrige (je &lt; 0,1 s): " + ", ".join(small), st["cell"]), "&lt; 0,1".replace("&lt;", "<")])
    rows.append([STARTUP_LABELS["total"], fmt(startup["total"], 2)])
    t = Table(rows, colWidths=[110 * mm, 25 * mm], hAlign="LEFT")
    ts = table_style()
    ts.add("FONTNAME", (0, -1), (-1, -1), "DejaVu-Bold")
    ts.add("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.HexColor(TEXT_SECONDARY))
    t.setStyle(ts)
    story.append(t)
    if startup.get("dense_model+index", 0) > 60:
        story.append(Paragraph(
            "Hinweis: Der hohe Wert für das Dense-Modell in diesem Lauf stammt von einem einmaligen "
            "Nachladen der Modellgewichte (model.safetensors, ca. 2,2 GB) aus dem Hugging-Face-Hub. "
            "Im Offline-Betrieb (HF_HUB_OFFLINE=1, wie im docker-compose-Setup) lädt das Modell in "
            "wenigen Sekunden.", st["caption"],
        ))

    story.append(Paragraph("Einordnung und Grenzen", st["h1"]))
    limits = [
        ("<b>GPU statt CPU</b>: Gemessen wurde lokal mit GPU. Das docker-compose-Setup hat keine "
         "GPU-Konfiguration – dort laufen bge-m3 und der Cross-Encoder auf der CPU, Dense- und "
         "Cross-Encoder-Szenarien werden entsprechend langsamer sein.") if on_gpu else
        ("<b>Nur CPU</b>: Die GPU war deaktiviert, wie im docker-compose-Setup ohne "
         "GPU-Konfiguration. Die lokale CPU kann schneller oder langsamer sein als die des "
         "Zielservers; im Container begrenzen zudem CPU-Limits und das Speicherlimit (8 GB)."),
        "<b>Ohne HTTP und ohne Parallelität</b>: Netzwerk, Serialisierung und uvicorn-Overhead sind "
        "nicht enthalten; der Durchsatz gilt für einen Prozess mit nacheinander abgearbeiteten Anfragen.",
        "<b>Kalte Caches</b>: Wiederholte Anfragen sind im Betrieb durch Query-Embedding- und "
        "Cross-Encoder-Cache schneller als hier gemessen.",
        "<b>Embedding-Speicher</b>: Statt PostgreSQL/pgvector wurde ein lokaler Datei-Cache genutzt. "
        "Die Datenbank wird nur beim Start gelesen; zur Suchzeit wird in beiden Fällen der "
        "FAISS-Index im Speicher verwendet – die Suchlatenz ist davon nicht betroffen.",
        "<b>XGB-Fusion</b> fehlt, da lokal kein trainiertes Modell (models/xgb_fusion.json) vorlag.",
    ]
    story.append(bullets(limits, st["body"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Rohdaten: {run_dir.as_posix()}/ (summary.json, summary.csv, timings_raw.csv). "
        "Erzeugt mit src/scripts/benchmark_latency.py und src/scripts/benchmark_report.py.",
        st["small"],
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return output


def main():
    p = argparse.ArgumentParser(description="PDF-Bericht aus einem Benchmark-Lauf erzeugen.")
    p.add_argument("run_dir", help="Verzeichnis mit summary.json (results/benchmark/<timestamp>)")
    p.add_argument("--output", default=None, help="Ziel-PDF (Default: <run_dir>/laufzeittest_bericht.pdf)")
    p.add_argument("--cpu-name", default=os.getenv("BENCHMARK_CPU_NAME"), help="CPU-Bezeichnung für den Bericht")
    p.add_argument("--compare", default=None, help="Zweiter Lauf (Verzeichnis) für einen Vergleichsabschnitt, z.B. GPU vs. CPU")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "laufzeittest_bericht.pdf"
    build_pdf(run_dir, output, args.cpu_name, Path(args.compare) if args.compare else None)
    print(f"PDF geschrieben: {output}")


if __name__ == "__main__":
    main()
