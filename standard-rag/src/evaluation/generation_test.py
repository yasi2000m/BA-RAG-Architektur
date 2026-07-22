"""
Evaluation des Temperature-Parameters.

Dieses Modul untersucht den Einfluss verschiedener
Temperature-Werte auf die generierten Antworten.

Konstant bleiben:

- Dokumentloader
- Chunk Size
- Overlap
- Top-k
- Embedding-Modell
- LLM-Modell
- Testfragen
- abgerufene Chunks
- Prompt

Verändert wird ausschließlich:

- temperature

Ausgewertet werden:

- generierte Antwort
- Antwortlaufzeit
- Input-Tokens
- Output-Tokens
- Gesamt-Tokens
- Antwortlänge
- Antwortgenauigkeit nach dem 2/1/0-Punktesystem

Die Antwortgenauigkeit wird nicht automatisch bewertet.
In der erzeugten CSV-Datei werden dafür leere Spalten
bereitgestellt, die anschließend manuell ausgefüllt werden.

Die Datei gehört in:

evaluation/generation_test.py
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
from generation import AnswerGenerator
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
    Lädt die Fragen und Referenzantworten aus der Datei
    test_questions.json.

    Erwartetes Format:

    [
        {
            "id": 1,
            "question": "Was ist das Ohmsche Gesetz?",
            "expected_keywords": [
                "spannung",
                "strom",
                "widerstand"
            ],
            "expected_answer": (
                "Das Ohmsche Gesetz beschreibt den "
                "Zusammenhang zwischen Spannung, "
                "Stromstärke und Widerstand."
            )
        }
    ]

    Für den Temperature-Test ist expected_answer hilfreich,
    weil die erzeugte Antwort später damit verglichen und
    manuell mit 2, 1 oder 0 Punkten bewertet werden kann.

    expected_keywords sind für diesen Test nicht zwingend
    erforderlich, werden aber weiterhin eingelesen.
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

        expected_answer = str(
            question_data.get(
                "expected_answer",
                "",
            )
        ).strip()

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
                "expected_answer": expected_answer,
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
# Dokument, Chunks und Vector Store vorbereiten
# ==========================================================

def prepare_generation_data() -> tuple[
    list[str],
    EmbeddingModel,
    LocalVectorStore,
    AnswerGenerator,
]:
    """
    Bereitet die RAG-Pipeline einmal für alle
    Temperature-Werte vor.

    Ablauf:

    1. PDF mit dem Standard-Dokumentloader laden
    2. Dokument mit konstanter Chunk Size und konstantem
       Overlap aufteilen
    3. Chunk-Embeddings einmalig erzeugen
    4. Chunks und Embeddings in den Vector Store einfügen
    5. AnswerGenerator erzeugen

    Dadurch werden bei allen Temperature-Werten dieselben
    Chunks, Embeddings und Modelle verwendet.
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

    answer_generator = AnswerGenerator()

    return (
        chunks,
        embedding_model,
        vector_store,
        answer_generator,
    )


# ==========================================================
# Retrieval einmalig vorbereiten
# ==========================================================

def prepare_question_contexts(
    test_questions: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    vector_store: LocalVectorStore,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Führt das Retrieval für jede Testfrage genau einmal aus.

    Die abgerufenen Chunks werden danach für alle
    Temperature-Werte wiederverwendet.

    Das ist wichtig, damit beim Temperature-Test wirklich
    nur die Temperature verändert wird.
    """
    prepared_questions: list[dict[str, Any]] = []

    print()
    print(
        "Relevante Chunks werden einmalig "
        "für alle Testfragen abgerufen ..."
    )

    for question_data in test_questions:
        question_id = question_data["id"]
        question = question_data["question"]

        retrieval_start = time.perf_counter()

        relevant_chunks = retrieve_relevant_chunks(
            query=question,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=top_k,
        )

        retrieval_runtime = (
            time.perf_counter() - retrieval_start
        )

        prepared_questions.append(
            {
                **question_data,
                "relevant_chunks": relevant_chunks,
                "retrieval_runtime_seconds": (
                    retrieval_runtime
                ),
            }
        )

        print(
            f"Frage {question_id}: "
            f"{len(relevant_chunks)} Chunks abgerufen."
        )

    return prepared_questions


# ==========================================================
# Usage-Daten auslesen
# ==========================================================

def get_usage_values(
    response: Any,
) -> tuple[int, int, int]:
    """
    Liest Input-, Output- und Gesamt-Tokens aus der
    API-Antwort aus.

    Falls keine Usage-Daten vorhanden sind, werden
    Nullwerte zurückgegeben.
    """
    usage = getattr(
        response,
        "usage",
        None,
    )

    if usage is None:
        return 0, 0, 0

    input_tokens = int(
        getattr(
            usage,
            "prompt_tokens",
            0,
        )
        or 0
    )

    output_tokens = int(
        getattr(
            usage,
            "completion_tokens",
            0,
        )
        or 0
    )

    total_tokens = int(
        getattr(
            usage,
            "total_tokens",
            0,
        )
        or (
            input_tokens
            + output_tokens
        )
    )

    return (
        input_tokens,
        output_tokens,
        total_tokens,
    )


# ==========================================================
# Antwort erzeugen
# ==========================================================

def generate_answer_with_temperature(
    answer_generator: AnswerGenerator,
    question: str,
    relevant_chunks: list[str],
    temperature: float,
) -> dict[str, Any]:
    """
    Erzeugt eine Antwort mit dem angegebenen
    Temperature-Wert.

    Der Prompt wird mit der vorhandenen build_prompt()-Methode
    des AnswerGenerator erstellt.

    Die vorhandene generate_answer()-Methode wird hier nicht
    verwendet, weil dort temperature fest auf 0.2 gesetzt ist.
    """
    prompt = answer_generator.build_prompt(
        query=question,
        relevant_chunks=relevant_chunks,
    )

    generation_start = time.perf_counter()

    response = (
        answer_generator.client.chat.completions.create(
            model=answer_generator.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
        )
    )

    generation_runtime = (
        time.perf_counter() - generation_start
    )

    answer = (
        response.choices[0].message.content
        or ""
    )

    (
        input_tokens,
        output_tokens,
        total_tokens,
    ) = get_usage_values(response)

    return {
        "prompt": prompt,
        "answer": answer,
        "generation_runtime_seconds": (
            generation_runtime
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


# ==========================================================
# Optionale Kostenberechnung
# ==========================================================

def calculate_generation_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> tuple[float, float, float]:
    """
    Berechnet die geschätzten Kosten anhand der tatsächlich
    von der API gemeldeten Tokenzahlen.

    Preise werden pro eine Million Tokens angegeben.

    Solange die Preise in config.py auf 0.0 stehen, bleiben
    auch die berechneten Kosten bei 0.0.
    """
    input_cost = (
        input_tokens
        / 1_000_000
        * input_price_per_million
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * output_price_per_million
    )

    total_cost = (
        input_cost
        + output_cost
    )

    return (
        input_cost,
        output_cost,
        total_cost,
    )


# ==========================================================
# Text für CSV vorbereiten
# ==========================================================

def normalize_csv_text(
    text: str,
) -> str:
    """
    Ersetzt Zeilenumbrüche durch Leerzeichen.

    Dadurch bleiben längere Antworten in Excel innerhalb
    einer einzelnen Tabellenzeile besser lesbar.
    """
    return " ".join(
        text.split()
    )


# ==========================================================
# Ergebnisordner und CSV
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
    Speichert die Ergebnisse als semikolongetrennte
    CSV-Datei.
    """
    if not rows:
        raise ValueError(
            "Es liegen keine Ergebnisse zum Speichern vor."
        )

    output_path = (
        get_results_directory()
        / filename
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
# Konsolenausgabe
# ==========================================================

def print_generation_result(
    temperature: float,
    repetition: int,
    question_id: Any,
    generation_result: dict[str, Any],
) -> None:
    """
    Gibt ein einzelnes Ergebnis kurz in der Konsole aus.
    """
    print()
    print("-" * 65)
    print(f"Temperature: {temperature}")
    print(f"Wiederholung: {repetition}")
    print(f"Frage-ID: {question_id}")
    print(
        f"Laufzeit: "
        f"{generation_result['generation_runtime_seconds']:.4f} s"
    )
    print(
        f"Input-Tokens: "
        f"{generation_result['input_tokens']}"
    )
    print(
        f"Output-Tokens: "
        f"{generation_result['output_tokens']}"
    )
    print(
        f"Gesamt-Tokens: "
        f"{generation_result['total_tokens']}"
    )


# ==========================================================
# Hauptfunktion
# ==========================================================

def run_temperature_test() -> None:
    """
    Testet alle in config.py eingetragenen
    Temperature-Werte automatisch nacheinander.

    Für jeden Temperature-Wert werden alle Testfragen
    beantwortet.

    Optional kann jede Kombination mehrfach ausgeführt
    werden. Dadurch lässt sich untersuchen, wie stark sich
    Antworten bei derselben Temperature unterscheiden.
    """
    print()
    print("=" * 65)
    print("Temperature-Test")
    print("=" * 65)

    try:
        test_questions = load_test_questions()

        (
            chunks,
            embedding_model,
            vector_store,
            answer_generator,
        ) = prepare_generation_data()

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print()
        print(f"Fehler: {error}")
        return

    temperature_values = getattr(
        config,
        "TEMPERATURE_VALUES",
        [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
    )

    repetitions = int(
        getattr(
            config,
            "TEMPERATURE_TEST_REPETITIONS",
            1,
        )
    )

    fixed_top_k = int(
        getattr(
            config,
            "TOP_K",
            5,
        )
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

    input_price_per_million = float(
        getattr(
            config,
            "LLM_INPUT_PRICE_PER_MILLION",
            0.0,
        )
    )

    output_price_per_million = float(
        getattr(
            config,
            "LLM_OUTPUT_PRICE_PER_MILLION",
            0.0,
        )
    )

    currency = getattr(
        config,
        "CURRENCY",
        "USD",
    )

    if repetitions <= 0:
        print()
        print(
            "TEMPERATURE_TEST_REPETITIONS muss "
            "größer als 0 sein."
        )
        return

    if fixed_top_k <= 0:
        print()
        print(
            "TOP_K muss größer als 0 sein."
        )
        return

    try:
        prepared_questions = prepare_question_contexts(
            test_questions=test_questions,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=fixed_top_k,
        )
    except Exception as error:
        print()
        print(
            "Fehler beim Abrufen der relevanten Chunks: "
            f"{error}"
        )
        return

    print()
    print("Konstant gehaltene Werte:")
    print(f"Dokumentloader: {loader_type}")
    print(f"Chunk Size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print(f"Top-k: {fixed_top_k}")
    print(
        f"Embedding-Modell: "
        f"{embedding_model.model_name}"
    )
    print(
        f"LLM-Modell: "
        f"{answer_generator.model_name}"
    )
    print(f"Anzahl Dokument-Chunks: {len(chunks)}")
    print(f"Anzahl Testfragen: {len(test_questions)}")
    print(
        f"Wiederholungen pro Temperature: "
        f"{repetitions}"
    )
    print()
    print(
        "Getestete Temperature-Werte: "
        f"{temperature_values}"
    )

    detail_results: list[dict[str, Any]] = []
    timestamp = create_timestamp()

    for temperature_value in temperature_values:
        try:
            temperature = float(
                temperature_value
            )
        except (TypeError, ValueError):
            print()
            print(
                f"Temperature-Wert "
                f"'{temperature_value}' wird "
                "übersprungen: ungültige Zahl."
            )
            continue

        if temperature < 0:
            print()
            print(
                f"Temperature {temperature} wird "
                "übersprungen: Der Wert darf "
                "nicht negativ sein."
            )
            continue

        print()
        print("=" * 65)
        print(
            f"Teste Temperature {temperature}"
        )
        print("=" * 65)

        for repetition in range(
            1,
            repetitions + 1,
        ):
            for question_data in prepared_questions:
                question_id = question_data["id"]
                question = question_data["question"]
                expected_answer = question_data[
                    "expected_answer"
                ]
                expected_keywords = question_data[
                    "expected_keywords"
                ]
                relevant_chunks = question_data[
                    "relevant_chunks"
                ]

                try:
                    generation_result = (
                        generate_answer_with_temperature(
                            answer_generator=(
                                answer_generator
                            ),
                            question=question,
                            relevant_chunks=(
                                relevant_chunks
                            ),
                            temperature=temperature,
                        )
                    )
                except Exception as error:
                    print()
                    print(
                        f"Fehler bei Temperature "
                        f"{temperature}, Wiederholung "
                        f"{repetition}, Frage "
                        f"{question_id}: {error}"
                    )
                    continue

                (
                    input_cost,
                    output_cost,
                    total_cost,
                ) = calculate_generation_cost(
                    input_tokens=(
                        generation_result[
                            "input_tokens"
                        ]
                    ),
                    output_tokens=(
                        generation_result[
                            "output_tokens"
                        ]
                    ),
                    input_price_per_million=(
                        input_price_per_million
                    ),
                    output_price_per_million=(
                        output_price_per_million
                    ),
                )

                answer = generation_result[
                    "answer"
                ]

                context_text = "\n\n".join(
                    relevant_chunks
                )

                detail_results.append(
                    {
                        "testart": "temperature",
                        "pdf_datei": pdf_path,
                        "document_loader": (
                            loader_type
                        ),
                        "chunk_size_konstant": (
                            chunk_size
                        ),
                        "overlap_konstant": overlap,
                        "top_k_konstant": (
                            fixed_top_k
                        ),
                        "embedding_modell": (
                            embedding_model.model_name
                        ),
                        "llm_modell": (
                            answer_generator.model_name
                        ),
                        "temperature": temperature,
                        "wiederholung": repetition,
                        "frage_id": question_id,
                        "frage": question,
                        "expected_keywords": (
                            ", ".join(
                                expected_keywords
                            )
                        ),
                        "referenzantwort": (
                            normalize_csv_text(
                                expected_answer
                            )
                        ),
                        "generierte_antwort": (
                            normalize_csv_text(
                                answer
                            )
                        ),
                        "antwortgenauigkeit_punkte_0_1_2": "",
                        "bewertungsbegruendung": "",
                        "abgerufene_chunks": (
                            len(relevant_chunks)
                        ),
                        "kontext_zeichen": (
                            len(context_text)
                        ),
                        "antwort_zeichen": (
                            len(answer)
                        ),
                        "antwort_woerter": (
                            len(answer.split())
                        ),
                        "retrieval_laufzeit_sekunden": round(
                            question_data[
                                "retrieval_runtime_seconds"
                            ],
                            6,
                        ),
                        "generierungs_laufzeit_sekunden": round(
                            generation_result[
                                "generation_runtime_seconds"
                            ],
                            4,
                        ),
                        "input_tokens": (
                            generation_result[
                                "input_tokens"
                            ]
                        ),
                        "output_tokens": (
                            generation_result[
                                "output_tokens"
                            ]
                        ),
                        "gesamt_tokens": (
                            generation_result[
                                "total_tokens"
                            ]
                        ),
                        "input_kosten": round(
                            input_cost,
                            8,
                        ),
                        "output_kosten": round(
                            output_cost,
                            8,
                        ),
                        "gesamt_kosten": round(
                            total_cost,
                            8,
                        ),
                        "waehrung": currency,
                    }
                )

                print_generation_result(
                    temperature=temperature,
                    repetition=repetition,
                    question_id=question_id,
                    generation_result=(
                        generation_result
                    ),
                )

    if not detail_results:
        print()
        print(
            "Es konnten keine Temperature-Tests "
            "durchgeführt werden."
        )
        return

    detail_path = write_csv(
        filename=(
            f"{timestamp}_temperature_results.csv"
        ),
        rows=detail_results,
    )

    print()
    print("=" * 65)
    print("Temperature-Test abgeschlossen")
    print("=" * 65)
    print(
        "Ergebnisse gespeichert unter:\n"
        f"{detail_path.resolve()}"
    )

    print()
    print(
        "Trage anschließend in der CSV-Datei in der Spalte "
        "'antwortgenauigkeit_punkte_0_1_2' deine Bewertung "
        "ein:"
    )
    print("2 Punkte = vollständig und korrekt")
    print("1 Punkt  = teilweise korrekt")
    print("0 Punkte = falsch oder nicht beantwortet")

    missing_reference_answers = sum(
        1
        for question in test_questions
        if not question["expected_answer"]
    )

    if missing_reference_answers > 0:
        print()
        print(
            f"Hinweis: Bei {missing_reference_answers} "
            "Testfrage(n) fehlt eine Referenzantwort. "
            "Ergänze dafür expected_answer in "
            "test_questions.json."
        )

    if (
        input_price_per_million == 0.0
        and output_price_per_million == 0.0
    ):
        print()
        print(
            "Hinweis: Die Kosten stehen bei 0.0, weil "
            "in config.py noch keine LLM-Preise "
            "eingetragen wurden."
        )


# ==========================================================
# Direkter Start
# ==========================================================

if __name__ == "__main__":
    run_temperature_test()
