"""
Evaluation von Chunk Size und Overlap.

Diese Datei enthält zwei getrennte Tests:

1. run_chunk_size_test()
   - verändert nur die Chunk-Größe
   - Overlap bleibt konstant

2. run_overlap_test()
   - verändert nur den Overlap
   - Chunk-Größe bleibt konstant

Untersucht werden:

- Anzahl der erzeugten Chunks
- durchschnittliche Chunk-Größe
- kleinster und größter Chunk
- Speicherbedarf der Chunk-Texte
- Retrieval-Trefferquote

Die Ergebnisse werden als CSV-Dateien im Ordner
evaluation/results gespeichert.
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# ==========================================================
# Projektordner für Imports verfügbar machen
# ==========================================================

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALUATION_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ==========================================================
# Projektmodule importieren
# ==========================================================

import config

from chunking import chunk_text
from document_loader import load_pdf_by_type
from embeddings import EmbeddingModel
from retrieval import retrieve_relevant_chunks
from vector_store import LocalVectorStore


# ==========================================================
# Pfad zu den Testfragen
# ==========================================================

TEST_QUESTIONS_PATH = (
    EVALUATION_DIR / "test_questions.json"
)


# ==========================================================
# Allgemeine Hilfsfunktionen
# ==========================================================

def load_test_questions() -> list[dict[str, Any]]:
    """
    Lädt die Testfragen aus test_questions.json.

    Erwartetes Format:

    [
        {
            "id": 1,
            "question": "Wie lautet das Ohmsche Gesetz?",
            "expected_keywords": [
                "spannung",
                "strom",
                "widerstand"
            ]
        }
    ]

    expected_keywords werden zur automatischen Bestimmung
    der Retrieval-Trefferquote verwendet.
    """
    if not TEST_QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            "Die Datei test_questions.json wurde nicht "
            f"gefunden:\n{TEST_QUESTIONS_PATH}"
        )

    with TEST_QUESTIONS_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "test_questions.json muss eine Liste enthalten."
        )

    valid_questions: list[dict[str, Any]] = []

    for question_data in questions:
        if not isinstance(question_data, dict):
            continue

        question = str(
            question_data.get("question", "")
        ).strip()

        if not question:
            continue

        expected_keywords = question_data.get(
            "expected_keywords",
            [],
        )

        if not isinstance(expected_keywords, list):
            expected_keywords = []

        cleaned_keywords = [
            str(keyword).strip()
            for keyword in expected_keywords
            if str(keyword).strip()
        ]

        valid_questions.append(
            {
                "id": question_data.get(
                    "id",
                    len(valid_questions) + 1,
                ),
                "question": question,
                "expected_keywords": cleaned_keywords,
            }
        )

    return valid_questions


def load_document() -> str:
    """
    Lädt das PDF mit dem aktuell in config.py gewählten
    Standard-Dokumentloader.
    """
    pdf_path = getattr(
        config,
        "PDF_PATH",
        "data/elektrotechnik_3.pdf",
    )

    loader_type = getattr(
        config,
        "DOCUMENT_LOADER",
        "text",
    )

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            "Die PDF-Datei wurde nicht gefunden:\n"
            f"{pdf_file.resolve()}"
        )

    print()
    print(f"PDF-Datei: {pdf_path}")
    print(f"Verwendeter Dokumentloader: {loader_type}")
    print("Dokument wird geladen ...")

    start_time = time.perf_counter()

    document_text = load_pdf_by_type(
        pdf_path=pdf_path,
        loader_type=loader_type,
    )

    runtime = time.perf_counter() - start_time

    if not document_text:
        raise ValueError(
            "Der Dokumentloader hat keinen Text geliefert."
        )

    print(
        f"Dokument geladen in {runtime:.4f} Sekunden."
    )

    return document_text


def calculate_chunk_statistics(
    chunks: list[str],
) -> dict[str, float | int]:
    """
    Berechnet statistische Kennzahlen der erzeugten Chunks.

    Die Chunk-Größe wird in Wörtern gemessen, da auch
    chunk_text() wortbasiert arbeitet.
    """
    if not chunks:
        return {
            "anzahl_chunks": 0,
            "woerter_gesamt": 0,
            "woerter_mittelwert": 0.0,
            "woerter_minimum": 0,
            "woerter_maximum": 0,
            "zeichen_gesamt": 0,
            "speicher_bytes": 0,
            "speicher_kib": 0.0,
        }

    word_counts = [
        len(chunk.split())
        for chunk in chunks
    ]

    character_count = sum(
        len(chunk)
        for chunk in chunks
    )

    storage_bytes = sum(
        len(chunk.encode("utf-8"))
        for chunk in chunks
    )

    return {
        "anzahl_chunks": len(chunks),
        "woerter_gesamt": sum(word_counts),
        "woerter_mittelwert": (
            sum(word_counts) / len(word_counts)
        ),
        "woerter_minimum": min(word_counts),
        "woerter_maximum": max(word_counts),
        "zeichen_gesamt": character_count,
        "speicher_bytes": storage_bytes,
        "speicher_kib": storage_bytes / 1024,
    }


def chunk_contains_keyword(
    chunk: str,
    expected_keywords: list[str],
) -> bool:
    """
    Prüft, ob mindestens eines der erwarteten
    Schlüsselwörter in einem Chunk vorkommt.
    """
    normalized_chunk = chunk.casefold()

    return any(
        keyword.casefold() in normalized_chunk
        for keyword in expected_keywords
    )


def retrieval_is_hit(
    retrieved_chunks: list[str],
    expected_keywords: list[str],
) -> bool:
    """
    Eine Retrieval-Anfrage gilt als Treffer, wenn
    mindestens eines der erwarteten Schlüsselwörter
    in mindestens einem abgerufenen Chunk vorkommt.
    """
    if not expected_keywords:
        return False

    return any(
        chunk_contains_keyword(
            chunk=chunk,
            expected_keywords=expected_keywords,
        )
        for chunk in retrieved_chunks
    )


def calculate_retrieval_hit_rate(
    chunks: list[str],
    test_questions: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    """
    Erstellt Embeddings für die Chunks und führt für jede
    Testfrage ein Retrieval durch.

    Rückgabe:

    - Anzahl auswertbarer Fragen
    - Anzahl Treffer
    - Trefferquote in Prozent
    - Detailergebnisse je Frage
    - Laufzeit für Embeddings
    - Laufzeit für Retrieval
    """
    questions_with_keywords = [
        question
        for question in test_questions
        if question["expected_keywords"]
    ]

    if not questions_with_keywords:
        return {
            "auswertbare_fragen": 0,
            "treffer": 0,
            "trefferquote_prozent": 0.0,
            "embedding_laufzeit_sekunden": 0.0,
            "retrieval_laufzeit_sekunden": 0.0,
            "details": [],
        }

    if not chunks:
        return {
            "auswertbare_fragen": (
                len(questions_with_keywords)
            ),
            "treffer": 0,
            "trefferquote_prozent": 0.0,
            "embedding_laufzeit_sekunden": 0.0,
            "retrieval_laufzeit_sekunden": 0.0,
            "details": [],
        }

    embedding_model = EmbeddingModel()

    print("Embeddings für Chunks werden erstellt ...")

    embedding_start = time.perf_counter()

    chunk_embeddings = embedding_model.embed_texts(
        chunks
    )

    embedding_runtime = (
        time.perf_counter() - embedding_start
    )

    vector_store = LocalVectorStore()

    vector_store.add(
        chunks=chunks,
        embeddings=chunk_embeddings,
    )

    hits = 0
    detail_results: list[dict[str, Any]] = []

    retrieval_start = time.perf_counter()

    for question_data in questions_with_keywords:
        question = question_data["question"]
        expected_keywords = question_data[
            "expected_keywords"
        ]

        retrieved_chunks = retrieve_relevant_chunks(
            query=question,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=top_k,
        )

        hit = retrieval_is_hit(
            retrieved_chunks=retrieved_chunks,
            expected_keywords=expected_keywords,
        )

        if hit:
            hits += 1

        detail_results.append(
            {
                "id": question_data["id"],
                "question": question,
                "expected_keywords": (
                    ", ".join(expected_keywords)
                ),
                "treffer": int(hit),
            }
        )

    retrieval_runtime = (
        time.perf_counter() - retrieval_start
    )

    hit_rate = (
        hits / len(questions_with_keywords) * 100
    )

    return {
        "auswertbare_fragen": (
            len(questions_with_keywords)
        ),
        "treffer": hits,
        "trefferquote_prozent": hit_rate,
        "embedding_laufzeit_sekunden": (
            embedding_runtime
        ),
        "retrieval_laufzeit_sekunden": (
            retrieval_runtime
        ),
        "details": detail_results,
    }


def create_timestamp() -> str:
    """
    Erstellt einen Zeitstempel für Ergebnisdateien.
    """
    return datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


def get_results_directory() -> Path:
    """
    Liefert den Ausgabeordner aus config.py.
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

    return results_dir


def write_summary_csv(
    filename: str,
    results: list[dict[str, Any]],
) -> Path:
    """
    Speichert die zusammengefassten Testergebnisse
    als CSV-Datei.
    """
    if not results:
        raise ValueError(
            "Es liegen keine Ergebnisse zum Speichern vor."
        )

    output_path = (
        get_results_directory() / filename
    )

    fieldnames = list(results[0].keys())

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


def write_detail_csv(
    filename: str,
    detail_results: list[dict[str, Any]],
) -> Path | None:
    """
    Speichert die Trefferentscheidung für jede einzelne
    Testfrage als zusätzliche CSV-Datei.
    """
    if not detail_results:
        return None

    output_path = (
        get_results_directory() / filename
    )

    fieldnames = list(
        detail_results[0].keys()
    )

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
        writer.writerows(detail_results)

    return output_path


def print_result(
    parameter_name: str,
    parameter_value: int,
    result: dict[str, Any],
) -> None:
    """
    Gibt ein einzelnes Testergebnis in der Konsole aus.
    """
    print()
    print("-" * 65)
    print(
        f"{parameter_name}: {parameter_value}"
    )
    print(
        f"Anzahl Chunks: "
        f"{result['anzahl_chunks']}"
    )
    print(
        f"Durchschnittliche Chunk-Größe: "
        f"{result['woerter_mittelwert']:.2f} Wörter"
    )
    print(
        f"Speicherbedarf: "
        f"{result['speicher_kib']:.2f} KiB"
    )
    print(
        f"Retrieval-Treffer: "
        f"{result['treffer']} von "
        f"{result['auswertbare_fragen']}"
    )
    print(
        f"Retrieval-Trefferquote: "
        f"{result['trefferquote_prozent']:.2f} %"
    )


# ==========================================================
# Chunk-Size-Test
# ==========================================================

def run_chunk_size_test() -> None:
    """
    Testet unterschiedliche Chunk-Größen.

    Konstant bleiben:

    - Dokumentloader
    - Overlap
    - top_k

    Verändert wird ausschließlich:

    - chunk_size
    """
    print()
    print("=" * 65)
    print("Chunk-Size-Test")
    print("=" * 65)

    try:
        document_text = load_document()
        test_questions = load_test_questions()
    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print()
        print(f"Fehler: {error}")
        return

    chunk_sizes = getattr(
        config,
        "CHUNK_SIZES",
        [100, 150, 250, 400, 600],
    )

    fixed_overlap = int(
        getattr(
            config,
            "OVERLAP",
            50,
        )
    )

    fixed_top_k = int(
        getattr(
            config,
            "TOP_K",
            5,
        )
    )

    loader_type = getattr(
        config,
        "DOCUMENT_LOADER",
        "text",
    )

    pdf_path = getattr(
        config,
        "PDF_PATH",
        "data/elektrotechnik_3.pdf",
    )

    print()
    print("Konstant gehaltene Werte:")
    print(f"Dokumentloader: {loader_type}")
    print(f"Overlap: {fixed_overlap}")
    print(f"Top-k: {fixed_top_k}")
    print()
    print(
        f"Getestete Chunk-Größen: {chunk_sizes}"
    )

    summary_results: list[dict[str, Any]] = []
    all_detail_results: list[dict[str, Any]] = []

    timestamp = create_timestamp()

    for chunk_size in chunk_sizes:
        chunk_size = int(chunk_size)

        if chunk_size <= 0:
            print()
            print(
                f"Chunk Size {chunk_size} wird "
                "übersprungen: Wert muss größer als 0 sein."
            )
            continue

        if fixed_overlap >= chunk_size:
            print()
            print(
                f"Chunk Size {chunk_size} wird "
                "übersprungen: Der konstante Overlap "
                f"{fixed_overlap} muss kleiner als die "
                "Chunk Size sein."
            )
            continue

        print()
        print(
            f"Teste Chunk Size {chunk_size} ..."
        )

        chunk_start = time.perf_counter()

        chunks = chunk_text(
            text=document_text,
            chunk_size=chunk_size,
            overlap=fixed_overlap,
        )

        chunk_runtime = (
            time.perf_counter() - chunk_start
        )

        statistics = calculate_chunk_statistics(
            chunks
        )

        try:
            retrieval_result = (
                calculate_retrieval_hit_rate(
                    chunks=chunks,
                    test_questions=test_questions,
                    top_k=fixed_top_k,
                )
            )
        except Exception as error:
            print(
                "Fehler bei Embedding oder Retrieval: "
                f"{error}"
            )

            retrieval_result = {
                "auswertbare_fragen": 0,
                "treffer": 0,
                "trefferquote_prozent": 0.0,
                "embedding_laufzeit_sekunden": 0.0,
                "retrieval_laufzeit_sekunden": 0.0,
                "details": [],
            }

        combined_result = {
            **statistics,
            **retrieval_result,
        }

        summary_result = {
            "testart": "chunk_size",
            "pdf_datei": pdf_path,
            "document_loader": loader_type,
            "chunk_size": chunk_size,
            "overlap_konstant": fixed_overlap,
            "top_k_konstant": fixed_top_k,
            "anzahl_chunks": (
                statistics["anzahl_chunks"]
            ),
            "woerter_gesamt_mit_overlap": (
                statistics["woerter_gesamt"]
            ),
            "chunk_woerter_mittelwert": round(
                statistics["woerter_mittelwert"],
                2,
            ),
            "chunk_woerter_minimum": (
                statistics["woerter_minimum"]
            ),
            "chunk_woerter_maximum": (
                statistics["woerter_maximum"]
            ),
            "speicher_bytes": (
                statistics["speicher_bytes"]
            ),
            "speicher_kib": round(
                statistics["speicher_kib"],
                2,
            ),
            "chunking_laufzeit_sekunden": round(
                chunk_runtime,
                6,
            ),
            "embedding_laufzeit_sekunden": round(
                retrieval_result[
                    "embedding_laufzeit_sekunden"
                ],
                4,
            ),
            "retrieval_laufzeit_sekunden": round(
                retrieval_result[
                    "retrieval_laufzeit_sekunden"
                ],
                4,
            ),
            "auswertbare_fragen": (
                retrieval_result[
                    "auswertbare_fragen"
                ]
            ),
            "retrieval_treffer": (
                retrieval_result["treffer"]
            ),
            "retrieval_trefferquote_prozent": round(
                retrieval_result[
                    "trefferquote_prozent"
                ],
                2,
            ),
        }

        summary_results.append(summary_result)

        for detail in retrieval_result["details"]:
            all_detail_results.append(
                {
                    "testart": "chunk_size",
                    "chunk_size": chunk_size,
                    "overlap_konstant": (
                        fixed_overlap
                    ),
                    "top_k_konstant": fixed_top_k,
                    **detail,
                }
            )

        print_result(
            parameter_name="Chunk Size",
            parameter_value=chunk_size,
            result=combined_result,
        )

    if not summary_results:
        print()
        print(
            "Es konnten keine gültigen "
            "Chunk-Size-Tests durchgeführt werden."
        )
        return

    summary_path = write_summary_csv(
        filename=(
            f"{timestamp}_chunk_size_results.csv"
        ),
        results=summary_results,
    )

    detail_path = write_detail_csv(
        filename=(
            f"{timestamp}_chunk_size_details.csv"
        ),
        detail_results=all_detail_results,
    )

    print()
    print("=" * 65)
    print("Chunk-Size-Test abgeschlossen")
    print("=" * 65)
    print(
        "Zusammenfassung gespeichert unter:\n"
        f"{summary_path.resolve()}"
    )

    if detail_path is not None:
        print()
        print(
            "Detailergebnisse gespeichert unter:\n"
            f"{detail_path.resolve()}"
        )

    if not any(
        question["expected_keywords"]
        for question in test_questions
    ):
        print()
        print(
            "Hinweis: Es wurde keine Trefferquote "
            "berechnet, weil in test_questions.json "
            "keine expected_keywords eingetragen sind."
        )


# ==========================================================
# Overlap-Test
# ==========================================================

def run_overlap_test() -> None:
    """
    Testet unterschiedliche Overlap-Werte.

    Konstant bleiben:

    - Dokumentloader
    - Chunk Size
    - top_k

    Verändert wird ausschließlich:

    - overlap
    """
    print()
    print("=" * 65)
    print("Overlap-Test")
    print("=" * 65)

    try:
        document_text = load_document()
        test_questions = load_test_questions()
    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print()
        print(f"Fehler: {error}")
        return

    overlaps = getattr(
        config,
        "OVERLAPS",
        [0, 25, 50, 75, 100],
    )

    fixed_chunk_size = int(
        getattr(
            config,
            "CHUNK_SIZE",
            250,
        )
    )

    fixed_top_k = int(
        getattr(
            config,
            "TOP_K",
            5,
        )
    )

    loader_type = getattr(
        config,
        "DOCUMENT_LOADER",
        "text",
    )

    pdf_path = getattr(
        config,
        "PDF_PATH",
        "data/elektrotechnik_3.pdf",
    )

    print()
    print("Konstant gehaltene Werte:")
    print(f"Dokumentloader: {loader_type}")
    print(f"Chunk Size: {fixed_chunk_size}")
    print(f"Top-k: {fixed_top_k}")
    print()
    print(
        f"Getestete Overlap-Werte: {overlaps}"
    )

    summary_results: list[dict[str, Any]] = []
    all_detail_results: list[dict[str, Any]] = []

    timestamp = create_timestamp()

    for overlap in overlaps:
        overlap = int(overlap)

        if overlap < 0:
            print()
            print(
                f"Overlap {overlap} wird übersprungen: "
                "Der Wert darf nicht negativ sein."
            )
            continue

        if overlap >= fixed_chunk_size:
            print()
            print(
                f"Overlap {overlap} wird übersprungen: "
                "Der Wert muss kleiner als die konstante "
                f"Chunk Size {fixed_chunk_size} sein."
            )
            continue

        print()
        print(
            f"Teste Overlap {overlap} ..."
        )

        chunk_start = time.perf_counter()

        chunks = chunk_text(
            text=document_text,
            chunk_size=fixed_chunk_size,
            overlap=overlap,
        )

        chunk_runtime = (
            time.perf_counter() - chunk_start
        )

        statistics = calculate_chunk_statistics(
            chunks
        )

        try:
            retrieval_result = (
                calculate_retrieval_hit_rate(
                    chunks=chunks,
                    test_questions=test_questions,
                    top_k=fixed_top_k,
                )
            )
        except Exception as error:
            print(
                "Fehler bei Embedding oder Retrieval: "
                f"{error}"
            )

            retrieval_result = {
                "auswertbare_fragen": 0,
                "treffer": 0,
                "trefferquote_prozent": 0.0,
                "embedding_laufzeit_sekunden": 0.0,
                "retrieval_laufzeit_sekunden": 0.0,
                "details": [],
            }

        combined_result = {
            **statistics,
            **retrieval_result,
        }

        summary_result = {
            "testart": "overlap",
            "pdf_datei": pdf_path,
            "document_loader": loader_type,
            "chunk_size_konstant": (
                fixed_chunk_size
            ),
            "overlap": overlap,
            "top_k_konstant": fixed_top_k,
            "anzahl_chunks": (
                statistics["anzahl_chunks"]
            ),
            "woerter_gesamt_mit_overlap": (
                statistics["woerter_gesamt"]
            ),
            "chunk_woerter_mittelwert": round(
                statistics["woerter_mittelwert"],
                2,
            ),
            "chunk_woerter_minimum": (
                statistics["woerter_minimum"]
            ),
            "chunk_woerter_maximum": (
                statistics["woerter_maximum"]
            ),
            "speicher_bytes": (
                statistics["speicher_bytes"]
            ),
            "speicher_kib": round(
                statistics["speicher_kib"],
                2,
            ),
            "chunking_laufzeit_sekunden": round(
                chunk_runtime,
                6,
            ),
            "embedding_laufzeit_sekunden": round(
                retrieval_result[
                    "embedding_laufzeit_sekunden"
                ],
                4,
            ),
            "retrieval_laufzeit_sekunden": round(
                retrieval_result[
                    "retrieval_laufzeit_sekunden"
                ],
                4,
            ),
            "auswertbare_fragen": (
                retrieval_result[
                    "auswertbare_fragen"
                ]
            ),
            "retrieval_treffer": (
                retrieval_result["treffer"]
            ),
            "retrieval_trefferquote_prozent": round(
                retrieval_result[
                    "trefferquote_prozent"
                ],
                2,
            ),
        }

        summary_results.append(summary_result)

        for detail in retrieval_result["details"]:
            all_detail_results.append(
                {
                    "testart": "overlap",
                    "chunk_size_konstant": (
                        fixed_chunk_size
                    ),
                    "overlap": overlap,
                    "top_k_konstant": fixed_top_k,
                    **detail,
                }
            )

        print_result(
            parameter_name="Overlap",
            parameter_value=overlap,
            result=combined_result,
        )

    if not summary_results:
        print()
        print(
            "Es konnten keine gültigen "
            "Overlap-Tests durchgeführt werden."
        )
        return

    summary_path = write_summary_csv(
        filename=(
            f"{timestamp}_overlap_results.csv"
        ),
        results=summary_results,
    )

    detail_path = write_detail_csv(
        filename=(
            f"{timestamp}_overlap_details.csv"
        ),
        detail_results=all_detail_results,
    )

    print()
    print("=" * 65)
    print("Overlap-Test abgeschlossen")
    print("=" * 65)
    print(
        "Zusammenfassung gespeichert unter:\n"
        f"{summary_path.resolve()}"
    )

    if detail_path is not None:
        print()
        print(
            "Detailergebnisse gespeichert unter:\n"
            f"{detail_path.resolve()}"
        )

    if not any(
        question["expected_keywords"]
        for question in test_questions
    ):
        print()
        print(
            "Hinweis: Es wurde keine Trefferquote "
            "berechnet, weil in test_questions.json "
            "keine expected_keywords eingetragen sind."
        )


# ==========================================================
# Direkter Start zum Testen
# ==========================================================

if __name__ == "__main__":
    print()
    print("Welche Untersuchung soll gestartet werden?")
    print("1 - Chunk Size testen")
    print("2 - Overlap testen")

    choice = input("\nAuswahl: ").strip()

    if choice == "1":
        run_chunk_size_test()

    elif choice == "2":
        run_overlap_test()

    else:
        print("Ungültige Auswahl.")