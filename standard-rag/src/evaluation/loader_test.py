"""
Vergleich der verschiedenen Dokumentloader.

Untersucht werden:

1. Laufzeit
2. Anzahl der PDF-Seiten
3. Anzahl der visuell analysierten Seiten
4. Anzahl der Vision-API-Aufrufe
5. Größe des erzeugten Textes
6. Geschätzte Kosten

Die Datei gehört in:

evaluation/loader_test.py
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import fitz


# ==========================================================
# Projektordner für Imports verfügbar machen
# ==========================================================

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALUATION_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ==========================================================
# Eigene Module importieren
# ==========================================================

import config

from document_loader import load_pdf_by_type


# ==========================================================
# Hilfsfunktionen
# ==========================================================

def get_pdf_page_count(pdf_path: str) -> int:
    """
    Ermittelt die Anzahl der Seiten einer PDF-Datei.
    """
    with fitz.open(pdf_path) as document:
        return len(document)


def count_targeted_visual_pages(
    pdf_path: str,
    visual_keywords: list[str],
) -> int:
    """
    Zählt, wie viele PDF-Seiten mindestens eines der
    angegebenen Schlüsselwörter enthalten.

    Diese Seiten werden beim Loader 'targeted_visual'
    mit dem Vision-Modell analysiert.
    """
    normalized_keywords = [
        keyword.lower()
        for keyword in visual_keywords
    ]

    matching_pages = 0

    with fitz.open(pdf_path) as document:
        for page in document:
            page_text = page.get_text("text").lower()

            contains_keyword = any(
                keyword in page_text
                for keyword in normalized_keywords
            )

            if contains_keyword:
                matching_pages += 1

    return matching_pages


def get_visual_page_count(
    loader_type: str,
    pdf_path: str,
    page_count: int,
    visual_keywords: list[str],
) -> int:
    """
    Bestimmt die Anzahl der visuell analysierten Seiten
    abhängig vom verwendeten Loader.
    """
    if loader_type == "text":
        return 0

    if loader_type == "targeted_visual":
        return count_targeted_visual_pages(
            pdf_path=pdf_path,
            visual_keywords=visual_keywords,
        )

    if loader_type == "full_visual":
        return page_count

    raise ValueError(
        f"Unbekannter Loader: {loader_type}"
    )


def calculate_estimated_cost(
    api_calls: int,
    cost_per_vision_call: float,
) -> float:
    """
    Berechnet eine geschätzte Gesamtsumme anhand der
    Anzahl der Vision-Aufrufe.

    Wichtig:
    Dies ist keine exakte tokenbasierte Kostenberechnung.

    Die Berechnung lautet:

        API-Aufrufe * durchschnittliche Kosten pro Aufruf

    Der durchschnittliche Preis pro Vision-Aufruf wird
    in config.py eingetragen.
    """
    return api_calls * cost_per_vision_call


def write_loader_results(
    results: list[dict],
) -> Path:
    """
    Speichert die Ergebnisse als CSV-Datei.

    Der Dateiname enthält Datum und Uhrzeit, damit alte
    Ergebnisse nicht überschrieben werden.
    """
    results_dir = Path(
        getattr(
            config,
            "RESULTS_DIR",
            EVALUATION_DIR / "results",
        )
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_path = (
        results_dir
        / f"{timestamp}_document_loader_results.csv"
    )

    fieldnames = [
        "loader",
        "loader_bezeichnung",
        "pdf_datei",
        "pdf_seiten",
        "wiederholung",
        "laufzeit_sekunden",
        "visuell_analysierte_seiten",
        "vision_api_aufrufe",
        "text_zeichen",
        "text_woerter",
        "text_speicher_bytes",
        "text_speicher_kib",
        "geschaetzte_kosten",
        "waehrung",
    ]

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(results)

    return output_path


def print_result(result: dict) -> None:
    """
    Gibt das Ergebnis eines Loader-Durchlaufs
    übersichtlich in der Konsole aus.
    """
    print()
    print("-" * 60)
    print(
        f"Loader: "
        f"{result['loader_bezeichnung']}"
    )
    print(
        f"Wiederholung: "
        f"{result['wiederholung']}"
    )
    print(
        f"Laufzeit: "
        f"{result['laufzeit_sekunden']:.4f} Sekunden"
    )
    print(
        f"PDF-Seiten: "
        f"{result['pdf_seiten']}"
    )
    print(
        f"Visuell analysierte Seiten: "
        f"{result['visuell_analysierte_seiten']}"
    )
    print(
        f"Vision-API-Aufrufe: "
        f"{result['vision_api_aufrufe']}"
    )
    print(
        f"Erzeugte Textgröße: "
        f"{result['text_speicher_kib']:.2f} KiB"
    )
    print(
        f"Geschätzte Kosten: "
        f"{result['geschaetzte_kosten']:.6f} "
        f"{result['waehrung']}"
    )


def calculate_average_results(
    results: list[dict],
) -> list[dict]:
    """
    Berechnet für jeden Loader die durchschnittliche
    Laufzeit und die durchschnittliche Textgröße.

    Diese Werte werden nur in der Konsole angezeigt.
    Die CSV-Datei enthält weiterhin alle einzelnen
    Durchläufe.
    """
    loader_types = sorted(
        {
            result["loader"]
            for result in results
        }
    )

    average_results: list[dict] = []

    for loader_type in loader_types:
        loader_results = [
            result
            for result in results
            if result["loader"] == loader_type
        ]

        average_runtime = sum(
            result["laufzeit_sekunden"]
            for result in loader_results
        ) / len(loader_results)

        average_size = sum(
            result["text_speicher_kib"]
            for result in loader_results
        ) / len(loader_results)

        average_results.append(
            {
                "loader": loader_type,
                "loader_bezeichnung": (
                    loader_results[0][
                        "loader_bezeichnung"
                    ]
                ),
                "durchschnittliche_laufzeit": (
                    average_runtime
                ),
                "durchschnittliche_textgroesse": (
                    average_size
                ),
                "vision_api_aufrufe": (
                    loader_results[0][
                        "vision_api_aufrufe"
                    ]
                ),
                "geschaetzte_kosten": (
                    loader_results[0][
                        "geschaetzte_kosten"
                    ]
                ),
            }
        )

    return average_results


def print_average_results(
    average_results: list[dict],
) -> None:
    """
    Zeigt eine Zusammenfassung der durchschnittlichen
    Ergebnisse an.
    """
    print()
    print("=" * 75)
    print("Durchschnittliche Ergebnisse")
    print("=" * 75)

    for result in average_results:
        print()
        print(result["loader_bezeichnung"])
        print(
            f"  Durchschnittliche Laufzeit: "
            f"{result['durchschnittliche_laufzeit']:.4f} s"
        )
        print(
            f"  Durchschnittliche Textgröße: "
            f"{result['durchschnittliche_textgroesse']:.2f} KiB"
        )
        print(
            f"  Vision-API-Aufrufe: "
            f"{result['vision_api_aufrufe']}"
        )
        print(
            f"  Geschätzte Kosten: "
            f"{result['geschaetzte_kosten']:.6f}"
        )


# ==========================================================
# Hauptfunktion des Loader-Tests
# ==========================================================

def run_loader_test() -> None:
    """
    Testet alle in config.py eingetragenen Dokumentloader.

    Alle Loader erhalten dieselbe PDF-Datei.
    Dadurch können Laufzeit, Anzahl der API-Aufrufe,
    Textgröße und geschätzte Kosten verglichen werden.
    """
    print()
    print("=" * 60)
    print("Dokumentloader-Test")
    print("=" * 60)

    pdf_path = getattr(
        config,
        "PDF_PATH",
        "data/elektrotechnik_3.pdf",
    )

    document_loaders = getattr(
        config,
        "DOCUMENT_LOADERS",
        [
            "text",
            "targeted_visual",
            "full_visual",
        ],
    )

    loader_names = getattr(
        config,
        "DOCUMENT_LOADER_NAMES",
        {
            "text": "Nur direkt auslesbarer PDF-Text",
            "targeted_visual": (
                "PDF-Text mit gezielter visueller Analyse"
            ),
            "full_visual": (
                "PDF-Text mit vollständiger visueller Analyse"
            ),
        },
    )

    visual_keywords = getattr(
        config,
        "VISUAL_KEYWORDS",
        [
            "diagramm",
            "abbildung",
            "tabelle",
            "figure",
            "chart",
        ],
    )

    repetitions = int(
        getattr(
            config,
            "LOADER_TEST_REPETITIONS",
            1,
        )
    )

    cost_per_vision_call = float(
        getattr(
            config,
            "VISION_COST_PER_CALL",
            0.0,
        )
    )

    currency = getattr(
        config,
        "CURRENCY",
        "USD",
    )

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print()
        print(
            f"Fehler: Die PDF-Datei wurde nicht gefunden:"
        )
        print(pdf_file.resolve())
        print()
        print(
            "Prüfe den Wert PDF_PATH in "
            "evaluation/config.py."
        )
        return

    if repetitions <= 0:
        print(
            "LOADER_TEST_REPETITIONS muss "
            "größer als 0 sein."
        )
        return

    page_count = get_pdf_page_count(pdf_path)

    print(f"PDF-Datei: {pdf_path}")
    print(f"PDF-Seiten: {page_count}")
    print(f"Wiederholungen pro Loader: {repetitions}")

    all_results: list[dict] = []

    for loader_type in document_loaders:
        if loader_type not in {
            "text",
            "targeted_visual",
            "full_visual",
        }:
            print()
            print(
                f"Loader '{loader_type}' wird übersprungen, "
                "weil er nicht bekannt ist."
            )
            continue

        loader_name = loader_names.get(
            loader_type,
            loader_type,
        )

        visual_page_count = get_visual_page_count(
            loader_type=loader_type,
            pdf_path=pdf_path,
            page_count=page_count,
            visual_keywords=visual_keywords,
        )

        api_calls = visual_page_count

        estimated_cost = calculate_estimated_cost(
            api_calls=api_calls,
            cost_per_vision_call=(
                cost_per_vision_call
            ),
        )

        for repetition in range(
            1,
            repetitions + 1,
        ):
            print()
            print(
                f"Starte: {loader_name} "
                f"({repetition}/{repetitions})"
            )

            start_time = time.perf_counter()

            try:
                document_text = load_pdf_by_type(
                    pdf_path=pdf_path,
                    loader_type=loader_type,
                )
            except Exception as error:
                print()
                print(
                    f"Fehler beim Loader "
                    f"'{loader_type}': {error}"
                )
                continue

            runtime_seconds = (
                time.perf_counter() - start_time
            )

            text_size_bytes = len(
                document_text.encode("utf-8")
            )

            text_size_kib = (
                text_size_bytes / 1024
            )

            result = {
                "loader": loader_type,
                "loader_bezeichnung": loader_name,
                "pdf_datei": pdf_path,
                "pdf_seiten": page_count,
                "wiederholung": repetition,
                "laufzeit_sekunden": round(
                    runtime_seconds,
                    4,
                ),
                "visuell_analysierte_seiten": (
                    visual_page_count
                ),
                "vision_api_aufrufe": api_calls,
                "text_zeichen": len(document_text),
                "text_woerter": len(
                    document_text.split()
                ),
                "text_speicher_bytes": (
                    text_size_bytes
                ),
                "text_speicher_kib": round(
                    text_size_kib,
                    2,
                ),
                "geschaetzte_kosten": round(
                    estimated_cost,
                    6,
                ),
                "waehrung": currency,
            }

            all_results.append(result)
            print_result(result)

    if not all_results:
        print()
        print(
            "Es konnten keine Ergebnisse erzeugt werden."
        )
        return

    output_path = write_loader_results(
        all_results
    )

    average_results = calculate_average_results(
        all_results
    )

    print_average_results(
        average_results
    )

    print()
    print("=" * 60)
    print("Dokumentloader-Test abgeschlossen")
    print("=" * 60)
    print(
        f"Ergebnisse gespeichert unter:\n"
        f"{output_path.resolve()}"
    )

    if cost_per_vision_call == 0:
        print()
        print(
            "Hinweis: Die geschätzten Kosten stehen bei 0, "
            "weil VISION_COST_PER_CALL in config.py "
            "noch nicht eingetragen wurde."
        )
        print(
            "Ohne Änderung an describe_page_image() ist "
            "nur eine Kostenschätzung pro Vision-Aufruf "
            "möglich, keine exakte tokenbasierte Berechnung."
        )


# ==========================================================
# Direkter Start dieser Datei
# ==========================================================

if __name__ == "__main__":
    run_loader_test()
