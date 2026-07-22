"""
Evaluation des Top-k-Parameters.

Diese Datei untersucht den Einfluss verschiedener Top-k-Werte.

Konstant bleiben:

- Dokumentloader
- Chunk Size
- Overlap
- Embedding-Modell

Verändert wird ausschließlich:

- top_k

Ausgewertet werden:

- Retrieval-Trefferquote
- Anzahl der abgerufenen Chunks
- Größe des abgerufenen Kontexts
- geschätzter Tokenverbrauch des Kontexts
- durchschnittlicher Tokenverbrauch pro Frage
- Retrieval-Laufzeit

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
# Pfade
# ==========================================================

TEST_QUESTIONS_PATH = (
    EVALUATION_DIR / "test_questions.json"
)


# ==========================================================
# Testfragen laden
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

    Eine Frage ist nur automatisch auswertbar, wenn
    expected_keywords vorhanden sind.
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

    if not valid_questions:
        raise ValueError(
            "In test_questions.json wurden keine "
            "gültigen Testfragen gefunden."
        )

    return valid_questions


# ==========================================================
# Dokument laden und Chunks erzeugen
# ==========================================================

def prepare_retrieval_data() -> tuple[
    list[str],
    EmbeddingModel,
    LocalVectorStore,
]:
    """
    Bereitet das Dokument einmal für alle Top-k-Tests vor.

    Ablauf:

    1. Dokument mit dem Standardloader laden
    2. Dokument mit konstanter Chunk Size und konstantem
       Overlap aufteilen
    3. Embeddings einmalig erzeugen
    4. Chunks und Embeddings in den Vector Store einfügen

    Dadurch werden bei den verschiedenen Top-k-Werten
    dieselben Chunks und Embeddings verwendet.
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

    chunk_size = int(
        getattr(
            config,
            "CHUNK_SIZE",
            250,
        )
    )

    overlap = int(
        getattr(
            config,
            "OVERLAP",
            50,
        )
    )

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            "Die PDF-Datei wurde nicht gefunden:\n"
            f"{pdf_file.resolve()}"
        )

    if chunk_size <= 0:
        raise ValueError(
            "CHUNK_SIZE muss größer als 0 sein."
        )

    if overlap < 0:
        raise ValueError(
            "OVERLAP darf nicht negativ sein."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "OVERLAP muss kleiner als CHUNK_SIZE sein."
        )

    print()
    print(f"PDF-Datei: {pdf_path}")
    print(f"Dokumentloader: {loader_type}")
    print(f"Konstante Chunk Size: {chunk_size}")
    print(f"Konstanter Overlap: {overlap}")

    print()
    print("Dokument wird geladen ...")

    loading_start = time.perf_counter()

    document_text = load_pdf_by_type(
        pdf_path=pdf_path,
        loader_type=loader_type,
    )

    loading_runtime = (
        time.perf_counter() - loading_start
    )

    if not document_text:
        raise ValueError(
            "Der Dokumentloader hat keinen Text geliefert."
        )

    print(
        f"Dokument geladen in "
        f"{loading_runtime:.4f} Sekunden."
    )

    print("Dokument wird in Chunks aufgeteilt ...")

    chunking_start = time.perf_counter()

    chunks = chunk_text(
        text=document_text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunking_runtime = (
        time.perf_counter() - chunking_start
    )

    if not chunks:
        raise ValueError(
            "Es wurden keine Chunks erzeugt."
        )

    print(
        f"{len(chunks)} Chunks erzeugt in "
        f"{chunking_runtime:.4f} Sekunden."
    )

    print("Chunk-Embeddings werden erstellt ...")

    embedding_model = EmbeddingModel()

    embedding_start = time.perf_counter()

    chunk_embeddings = embedding_model.embed_texts(
        chunks
    )

    embedding_runtime = (
        time.perf_counter() - embedding_start
    )

    if not chunk_embeddings:
        raise ValueError(
            "Es wurden keine Embeddings erzeugt."
        )

    print(
        f"Embeddings erstellt in "
        f"{embedding_runtime:.4f} Sekunden."
    )

    vector_store = LocalVectorStore()

    vector_store.add(
        chunks=chunks,
        embeddings=chunk_embeddings,
    )

    return chunks, embedding_model, vector_store


# ==========================================================
# Trefferquote
# ==========================================================

def chunk_contains_keyword(
    chunk: str,
    expected_keywords: list[str],
) -> bool:
    """
    Prüft, ob mindestens eines der erwarteten
    Schlüsselwörter in einem Chunk enthalten ist.
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


def count_matching_retrieved_chunks(
    retrieved_chunks: list[str],
    expected_keywords: list[str],
) -> int:
    """
    Zählt, wie viele der abgerufenen Chunks mindestens
    eines der erwarteten Schlüsselwörter enthalten.

    Diese Kennzahl ergänzt die einfache Trefferquote.
    """
    if not expected_keywords:
        return 0

    return sum(
        1
        for chunk in retrieved_chunks
        if chunk_contains_keyword(
            chunk=chunk,
            expected_keywords=expected_keywords,
        )
    )


# ==========================================================
# Tokenzählung
# ==========================================================

def estimate_tokens_from_characters(
    text: str,
) -> int:
    """
    Schätzt die Anzahl der Tokens anhand der Textlänge.

    Als einfache Näherung wird angenommen:

        ungefähr 4 Zeichen = 1 Token

    Diese Methode wird verwendet, wenn tiktoken nicht
    installiert ist.
    """
    if not text:
        return 0

    return max(
        1,
        round(len(text) / 4),
    )


def count_tokens(
    text: str,
    model_name: str | None = None,
) -> tuple[int, str]:
    """
    Zählt Tokens möglichst mit tiktoken.

    Falls tiktoken nicht installiert ist oder für das Modell
    keine passende Kodierung gefunden wird, wird eine
    Schätzung anhand der Zeichenanzahl verwendet.

    Rückgabe:

    - Tokenanzahl
    - verwendete Messmethode
    """
    if not text:
        return 0, "kein Text"

    try:
        import tiktoken
    except ImportError:
        return (
            estimate_tokens_from_characters(text),
            "Schätzung: Zeichen / 4",
        )

    try:
        if model_name:
            encoding = tiktoken.encoding_for_model(
                model_name
            )
        else:
            encoding = tiktoken.get_encoding(
                "cl100k_base"
            )
    except KeyError:
        encoding = tiktoken.get_encoding(
            "cl100k_base"
        )

    return (
        len(encoding.encode(text)),
        "tiktoken",
    )


# ==========================================================
# Kontextstatistiken
# ==========================================================

def calculate_context_statistics(
    retrieved_chunks: list[str],
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Berechnet Größe und Tokenverbrauch des abgerufenen
    Kontexts.
    """
    context = "\n\n".join(
        retrieved_chunks
    )

    token_count, token_method = count_tokens(
        text=context,
        model_name=model_name,
    )

    word_count = sum(
        len(chunk.split())
        for chunk in retrieved_chunks
    )

    character_count = len(context)

    storage_bytes = len(
        context.encode("utf-8")
    )

    return {
        "kontext_woerter": word_count,
        "kontext_zeichen": character_count,
        "kontext_speicher_bytes": storage_bytes,
        "kontext_speicher_kib": (
            storage_bytes / 1024
        ),
        "kontext_tokens": token_count,
        "token_messmethode": token_method,
    }


# ==========================================================
# CSV-Ausgabe
# ==========================================================

def create_timestamp() -> str:
    """
    Erstellt einen Zeitstempel für Dateinamen.
    """
    return datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


def get_results_directory() -> Path:
    """
    Ermittelt den Ergebnisordner aus config.py.
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


def write_csv(
    filename: str,
    rows: list[dict[str, Any]],
) -> Path:
    """
    Speichert eine Liste von Dictionaries als
    semikolongetrennte CSV-Datei.
    """
    if not rows:
        raise ValueError(
            "Es liegen keine Ergebnisse zum Speichern vor."
        )

    output_path = (
        get_results_directory() / filename
    )

    fieldnames = list(
        rows[0].keys()
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
        writer.writerows(rows)

    return output_path


# ==========================================================
# Ausgabe in der Konsole
# ==========================================================

def print_top_k_result(
    result: dict[str, Any],
) -> None:
    """
    Gibt das zusammengefasste Ergebnis eines Top-k-Werts
    in der Konsole aus.
    """
    print()
    print("-" * 65)
    print(f"Top-k: {result['top_k']}")
    print(
        f"Retrieval-Treffer: "
        f"{result['retrieval_treffer']} von "
        f"{result['auswertbare_fragen']}"
    )
    print(
        f"Retrieval-Trefferquote: "
        f"{result['retrieval_trefferquote_prozent']:.2f} %"
    )
    print(
        f"Abgerufene Chunks insgesamt: "
        f"{result['abgerufene_chunks_gesamt']}"
    )
    print(
        f"Kontext-Tokens insgesamt: "
        f"{result['kontext_tokens_gesamt']}"
    )
    print(
        f"Kontext-Tokens pro Frage: "
        f"{result['kontext_tokens_mittel_pro_frage']:.2f}"
    )
    print(
        f"Retrieval-Laufzeit insgesamt: "
        f"{result['retrieval_laufzeit_sekunden']:.4f} s"
    )


# ==========================================================
# Hauptfunktion
# ==========================================================

def run_topk_test() -> None:
    """
    Testet verschiedene Top-k-Werte.

    Für jeden Top-k-Wert werden dieselben Testfragen,
    dieselben Chunks und dieselben Chunk-Embeddings
    verwendet.

    Nur die Anzahl der abgerufenen Chunks verändert sich.
    """
    print()
    print("=" * 65)
    print("Top-k-Test")
    print("=" * 65)

    try:
        test_questions = load_test_questions()

        (
            chunks,
            embedding_model,
            vector_store,
        ) = prepare_retrieval_data()

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print()
        print(f"Fehler: {error}")
        return

    top_k_values = getattr(
        config,
        "TOP_K_VALUES",
        [1, 3, 5, 7, 10],
    )

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

    chunk_size = int(
        getattr(
            config,
            "CHUNK_SIZE",
            250,
        )
    )

    overlap = int(
        getattr(
            config,
            "OVERLAP",
            50,
        )
    )

    model_name = getattr(
        embedding_model,
        "model_name",
        None,
    )

    questions_with_keywords = [
        question
        for question in test_questions
        if question["expected_keywords"]
    ]

    print()
    print("Konstant gehaltene Werte:")
    print(f"Dokumentloader: {loader_type}")
    print(f"Chunk Size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print(f"Anzahl Chunks: {len(chunks)}")
    print(f"Embedding-Modell: {model_name}")
    print()
    print(
        f"Getestete Top-k-Werte: {top_k_values}"
    )

    summary_results: list[dict[str, Any]] = []
    detail_results: list[dict[str, Any]] = []

    timestamp = create_timestamp()

    for top_k_value in top_k_values:
        try:
            top_k = int(top_k_value)
        except (TypeError, ValueError):
            print()
            print(
                f"Top-k-Wert '{top_k_value}' wird "
                "übersprungen: kein gültiger Integer."
            )
            continue

        if top_k <= 0:
            print()
            print(
                f"Top-k {top_k} wird übersprungen: "
                "Der Wert muss größer als 0 sein."
            )
            continue

        effective_top_k = min(
            top_k,
            len(chunks),
        )

        if effective_top_k != top_k:
            print()
            print(
                f"Top-k {top_k} ist größer als die "
                f"Anzahl der Chunks ({len(chunks)}). "
                f"Effektiv werden {effective_top_k} "
                "Chunks abgerufen."
            )

        print()
        print(
            f"Teste Top-k {top_k} ..."
        )

        hits = 0
        evaluated_questions = 0
        retrieved_chunks_total = 0
        relevant_retrieved_chunks_total = 0
        context_words_total = 0
        context_characters_total = 0
        context_storage_bytes_total = 0
        context_tokens_total = 0
        token_methods: set[str] = set()

        retrieval_start = time.perf_counter()

        for question_data in test_questions:
            question_id = question_data["id"]
            question = question_data["question"]
            expected_keywords = question_data[
                "expected_keywords"
            ]

            question_start = time.perf_counter()

            try:
                retrieved_chunks = (
                    retrieve_relevant_chunks(
                        query=question,
                        embedding_model=embedding_model,
                        vector_store=vector_store,
                        top_k=top_k,
                    )
                )
            except Exception as error:
                print()
                print(
                    f"Retrieval-Fehler bei Frage "
                    f"{question_id}: {error}"
                )
                continue

            question_runtime = (
                time.perf_counter() - question_start
            )

            context_statistics = (
                calculate_context_statistics(
                    retrieved_chunks=(
                        retrieved_chunks
                    ),
                    model_name=None,
                )
            )

            hit = retrieval_is_hit(
                retrieved_chunks=retrieved_chunks,
                expected_keywords=expected_keywords,
            )

            matching_chunks = (
                count_matching_retrieved_chunks(
                    retrieved_chunks=(
                        retrieved_chunks
                    ),
                    expected_keywords=(
                        expected_keywords
                    ),
                )
            )

            if expected_keywords:
                evaluated_questions += 1

                if hit:
                    hits += 1

            retrieved_chunks_total += len(
                retrieved_chunks
            )

            relevant_retrieved_chunks_total += (
                matching_chunks
            )

            context_words_total += (
                context_statistics[
                    "kontext_woerter"
                ]
            )

            context_characters_total += (
                context_statistics[
                    "kontext_zeichen"
                ]
            )

            context_storage_bytes_total += (
                context_statistics[
                    "kontext_speicher_bytes"
                ]
            )

            context_tokens_total += (
                context_statistics[
                    "kontext_tokens"
                ]
            )

            token_methods.add(
                context_statistics[
                    "token_messmethode"
                ]
            )

            detail_results.append(
                {
                    "testart": "top_k",
                    "pdf_datei": pdf_path,
                    "document_loader": loader_type,
                    "chunk_size_konstant": chunk_size,
                    "overlap_konstant": overlap,
                    "top_k": top_k,
                    "effektives_top_k": (
                        len(retrieved_chunks)
                    ),
                    "frage_id": question_id,
                    "frage": question,
                    "expected_keywords": (
                        ", ".join(expected_keywords)
                    ),
                    "retrieval_treffer": (
                        int(hit)
                        if expected_keywords
                        else ""
                    ),
                    "passende_chunks_im_retrieval": (
                        matching_chunks
                    ),
                    "abgerufene_chunks": (
                        len(retrieved_chunks)
                    ),
                    "kontext_woerter": (
                        context_statistics[
                            "kontext_woerter"
                        ]
                    ),
                    "kontext_zeichen": (
                        context_statistics[
                            "kontext_zeichen"
                        ]
                    ),
                    "kontext_speicher_bytes": (
                        context_statistics[
                            "kontext_speicher_bytes"
                        ]
                    ),
                    "kontext_speicher_kib": round(
                        context_statistics[
                            "kontext_speicher_kib"
                        ],
                        2,
                    ),
                    "kontext_tokens": (
                        context_statistics[
                            "kontext_tokens"
                        ]
                    ),
                    "token_messmethode": (
                        context_statistics[
                            "token_messmethode"
                        ]
                    ),
                    "retrieval_laufzeit_sekunden": round(
                        question_runtime,
                        6,
                    ),
                }
            )

        total_retrieval_runtime = (
            time.perf_counter() - retrieval_start
        )

        hit_rate = (
            hits / evaluated_questions * 100
            if evaluated_questions > 0
            else 0.0
        )

        question_count = len(
            test_questions
        )

        summary_result = {
            "testart": "top_k",
            "pdf_datei": pdf_path,
            "document_loader": loader_type,
            "chunk_size_konstant": chunk_size,
            "overlap_konstant": overlap,
            "anzahl_dokument_chunks": len(chunks),
            "top_k": top_k,
            "effektives_top_k_maximal": (
                effective_top_k
            ),
            "fragen_gesamt": question_count,
            "auswertbare_fragen": (
                evaluated_questions
            ),
            "retrieval_treffer": hits,
            "retrieval_trefferquote_prozent": round(
                hit_rate,
                2,
            ),
            "abgerufene_chunks_gesamt": (
                retrieved_chunks_total
            ),
            "passende_chunks_gesamt": (
                relevant_retrieved_chunks_total
            ),
            "kontext_woerter_gesamt": (
                context_words_total
            ),
            "kontext_zeichen_gesamt": (
                context_characters_total
            ),
            "kontext_speicher_bytes_gesamt": (
                context_storage_bytes_total
            ),
            "kontext_speicher_kib_gesamt": round(
                context_storage_bytes_total / 1024,
                2,
            ),
            "kontext_tokens_gesamt": (
                context_tokens_total
            ),
            "kontext_tokens_mittel_pro_frage": round(
                (
                    context_tokens_total
                    / question_count
                )
                if question_count > 0
                else 0.0,
                2,
            ),
            "retrieval_laufzeit_sekunden": round(
                total_retrieval_runtime,
                4,
            ),
            "retrieval_laufzeit_mittel_pro_frage": round(
                (
                    total_retrieval_runtime
                    / question_count
                )
                if question_count > 0
                else 0.0,
                6,
            ),
            "token_messmethode": (
                ", ".join(sorted(token_methods))
            ),
        }

        summary_results.append(
            summary_result
        )

        print_top_k_result(
            summary_result
        )

    if not summary_results:
        print()
        print(
            "Es konnten keine gültigen "
            "Top-k-Tests durchgeführt werden."
        )
        return

    summary_path = write_csv(
        filename=(
            f"{timestamp}_top_k_results.csv"
        ),
        rows=summary_results,
    )

    detail_path = write_csv(
        filename=(
            f"{timestamp}_top_k_details.csv"
        ),
        rows=detail_results,
    )

    print()
    print("=" * 65)
    print("Top-k-Test abgeschlossen")
    print("=" * 65)

    print(
        "Zusammenfassung gespeichert unter:\n"
        f"{summary_path.resolve()}"
    )

    print()
    print(
        "Detailergebnisse gespeichert unter:\n"
        f"{detail_path.resolve()}"
    )

    if not questions_with_keywords:
        print()
        print(
            "Hinweis: Die Retrieval-Trefferquote konnte "
            "nicht berechnet werden, weil keine Frage "
            "expected_keywords enthält."
        )

    print()
    print(
        "Hinweis zum Tokenverbrauch: Gemessen wird der "
        "Tokenumfang der abgerufenen Chunks. Es wird dabei "
        "noch keine Antwort mit dem LLM generiert."
    )


# ==========================================================
# Direkter Start
# ==========================================================

if __name__ == "__main__":
    run_topk_test()
