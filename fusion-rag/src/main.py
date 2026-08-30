import time
from pathlib import Path

from questions import QUESTIONS
from chunking import chunk_text
from document_loader import load_pdf
from embeddings import EmbeddingModel
from generation import AnswerGenerator
from retrieval import retrieve_relevant_chunks
from vector_store import LocalVectorStore


def get_directory_size(path: Path) -> int:
    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    )


def main() -> None:
    pdf_path = (
        "/Users/yasi/Documents/New project/BA-RAG-Architektur/"
        "fusion-rag/data/ErneuerbareEnergien.pdf"
    )

    saved_text_path = Path(
        "/Users/yasi/Documents/New project/BA-RAG-Architektur/"
        "fusion-rag/data/geladener_text2.txt"
    )

    # Bereits gespeicherten Dokumenttext verwenden.
    if saved_text_path.exists():
        print("Gespeicherter Dokumenttext wird verwendet.")
        document_text = saved_text_path.read_text(encoding="utf-8")

    # Falls noch kein Text gespeichert wurde, PDF-Loader einmalig ausfuehren.
    else:
        print("Kein gespeicherter Text gefunden. PDF-Loader wird ausgefuehrt.")

        document_text = load_pdf(pdf_path)

        if document_text:
            saved_text_path.write_text(
                document_text,
                encoding="utf-8",
            )
            print(f"Dokumenttext gespeichert unter: {saved_text_path}")

    if not document_text:
        print(f"Keine Inhalte im PDF gefunden. Pruefe die Datei: {pdf_path}")
        return


    # Gesamtlaufzeit startet nach dem Document Loader.
    total_start_time = time.perf_counter()


    chunks = chunk_text(
        document_text,
        chunk_size=250,
        overlap=25,
    )

    embedding_model = EmbeddingModel()
    chunk_embeddings = embedding_model.embed_texts(chunks)

    vector_db_path = Path(
        "vector_db_ErneuebareEnergien_Fusion"
    )

    vector_store = LocalVectorStore(
        str(vector_db_path)
    )
    vector_store.add(chunks, chunk_embeddings)
    vector_store.save()


    # Speicherbedarf der Vektordatenbank.
    storage_bytes = get_directory_size(vector_db_path)
    storage_mb = storage_bytes / (1024 * 1024)


    generator = AnswerGenerator()

    total_generation_tokens = 0
    total_query_generation_tokens = 0
    total_answer_time = 0.0


    for number, query in enumerate(QUESTIONS, start=1):
        print(f"\n{'=' * 70}")
        print(f"Frage {number}: {query}")


        # Antwortzeit startet vor dem Retrieval.
        answer_start_time = time.perf_counter()


        relevant_chunks, query_generation_tokens = retrieve_relevant_chunks(
            query=query,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=5,
            num_query_variants=5,
            fusion_search_k=10,
            rrf_k=60,
        )


        answer, used_tokens = generator.generate_answer(
            query,
            relevant_chunks,
        )


        # Antwortzeit = Retrieval + Antwortgenerierung.
        total_answer_time += (
            time.perf_counter() - answer_start_time
        )

        total_generation_tokens += used_tokens
        total_query_generation_tokens += query_generation_tokens


        print(f"\nAntwort {number}:")
        print(answer)


    # Gesamtlaufzeit endet nach der letzten Antwort.
    total_runtime = (
        time.perf_counter() - total_start_time
    )


    # Gesamter Tokenverbrauch ohne Document Loader.
    total_token_usage = (
        embedding_model.total_tokens
        + total_query_generation_tokens
        + total_generation_tokens
    )


    print(
        f"\nGesamte Antwortzeit: "
        f"{total_answer_time:.2f} Sekunden"
    )

    print(
        f"Gesamtlaufzeit: "
        f"{total_runtime:.2f} Sekunden"
    )

    print(
        f"Embedding-Tokens: "
        f"{embedding_model.total_tokens}"
    )

    print(
        f"Query-Generierungs-Tokens: "
        f"{total_query_generation_tokens}"
    )

    print(
        f"Generierungs-Tokens: "
        f"{total_generation_tokens}"
    )

    print(
        f"Gesamter Tokenverbrauch: "
        f"{total_token_usage}"
    )

    print(
        f"Speicherbedarf der RAG-Datenstruktur: "
        f"{storage_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()