#!/usr/bin/env python3
"""
Extract specified page ranges from multiple PDFs and merge into one PDF.
Page numbers in config are 1-based (as in the user's list).
"""

from pathlib import Path
from pypdf import PdfReader, PdfWriter

# Base directory (script location)
BASE = Path(__file__).resolve().parent

# (filename, list of (start_page, end_page_inclusive) in 1-based numbering)
# end_page_inclusive: last page to include
SOURCES = [
    # 1) Bishop — Pattern Recognition and Machine Learning
    (
        "Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf",
        [(4, 11), (12, 31), (32, 38), (38, 47), (137, 173), (303, 312)],
    ),
    # 2) Molnar — Interpretable Machine Learning (book)
    (
        "Molnar-interpretable-machine-learning_compressed.pdf",
        [(38, 53), (61, 78), (113, 125), (154, 162), (177, 188)],
    ),
    # 3) Molnar Dissertation (2010.093373)
    (
        "2010.093373.pdf",
        [(7, 12), (11, 20), (40, 80)],  # ~40-80 for PDP & PFI limitations
    ),
    # 4) Interpretable ML with Python — Masís
    (
        "Interpretable Machine Learning with Python (SafefilekU.com).pdf",
        [(1, 20), (150, 200)],  # intro + SHAP chapters (approx)
    ),
    # 5) Interpretable AI — Thampi
    (
        "Interpretable_AI_Building_explainable_machine_learning_systems_Manning.pdf",
        [(1, 40), (50, 100), (80, 120)],  # foundations, model-agnostic, local (overlap ok)
    ),
    # 6) Designing ML Systems — Chip Huyen (Ch 1-2, 4-6, monitoring; no exact pp.)
    (
        "Designing Machine Learning Systems.pdf",
        [(1, 80), (120, 280), (350, 461)],  # Ch1-2 approx, Ch4-6 approx, later/monitoring
    ),
    # 7) Applied ML Explainability — Ch1, Ch3, Ch6, Ch7, Ch10 (approx page ranges)
    (
        "Applied Machine Learning.pdf",
        [(1, 30), (40, 75), (100, 175), (220, 274)],
    ),
    # 8) Whitepaper — Cracking the Box / Interpreting ML Models
    (
        "Whitepaper-Interpreting-Machine-Learning-Models.pdf",
        [(4, 5), (5, 6), (7, 8), (11, 13)],
    ),
    # 9) SHAP Slides (Kubkowski) — using shap.pdf
    (
        "shap.pdf",
        [(2, 18)],  # 2-10 + 10-18 from user list
    ),
    # 10) SHAP Lecture (Gallic) — same or separate; content in shap.pdf covered above
    # 11) Orduz PyData
    (
        "orduz_pydata2021.pdf",
        [(2, 4), (10, 25)],
    ),
    # 12) Interpretable AI — Chapter 2 (White-Box)
    (
        "bookshelf_interpretableai_ch2.pdf",
        [(27, 32), (34, 39), (41, 48)],
    ),
    # 13) Molnar_Christoph.pdf — key sections (whole-doc ranges for Shapley, PDP, PFI)
    (
        "Molnar_Christoph.pdf",
        [(1, 50), (80, 150), (150, 220)],  # early, mid (PDP/PFI), model-agnostic
    ),
    # 14) shapley-values.pdf — entire document
    (
        "shapley-values.pdf",
        [(1, 999)],  # entire doc; script will clamp to max pages
    ),
    # 15) shap.pdf — already included above; skip duplicate or add tree explainer range
]

OUTPUT_PDF = BASE / "Anton_ML_Interpretability_Combined.pdf"


def extract_ranges(reader, ranges_1based):
    """Yield 0-based page indices for (start, end_inclusive) 1-based ranges."""
    n = len(reader.pages)
    for start, end in ranges_1based:
        s = max(1, start) - 1
        e = min(end, n)
        for i in range(s, e):
            yield i


def main():
    writer = PdfWriter()
    total_added = 0

    for filename, ranges in SOURCES:
        path = BASE / filename
        if not path.exists():
            print(f"SKIP (not found): {filename}")
            continue
        try:
            reader = PdfReader(path)
        except Exception as e:
            print(f"SKIP (read error): {filename} — {e}")
            continue
        n = len(reader.pages)
        # Clamp "entire doc" range (e.g. 1-999) to actual length
        clamped = []
        for a, b in ranges:
            if b >= 999:
                b = n
            clamped.append((a, b))
        indices = sorted(set(extract_ranges(reader, clamped)))
        if not indices:
            print(f"SKIP (no pages): {filename}")
            continue
        for i in indices:
            writer.add_page(reader.pages[i])
        total_added += len(indices)
        print(f"  + {filename}: pages {[r for r in clamped]} → {len(indices)} pages (total so far: {total_added})")

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PDF, "wb") as f:
        writer.write(f)
    print(f"\nWrote {total_added} pages to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
