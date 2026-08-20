from pathlib import Path
from questions import QUESTIONS
from chunking import chunk_text
from document_loader import load_pdf
from embeddings import EmbeddingModel
from generation import AnswerGenerator
from retrieval import retrieve_relevant_chunks
from vector_store import LocalVectorStore
import time


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    pdf_path = project_dir / "data" / "Elektrotechnik 3.pdf"
    saved_text_path = project_dir / "data" / "geladener_text.txt"

    # Bereits gespeicherten Dokumenttext verwenden.
    if saved_text_path.exists():
        print("Gespeicherter Dokumenttext wird verwendet.")
        document_text = saved_text_path.read_text(encoding="utf-8")

    # Falls noch kein Text gespeichert wurde, PDF-Loader einmalig ausfuehren.
    else:
        print("Kein gespeicherter Text gefunden. PDF-Loader wird ausgefuehrt.")

        document_text = load_pdf(str(pdf_path))

        if document_text:
            saved_text_path.write_text(
                document_text,
                encoding="utf-8",
            )
            print(f"Dokumenttext gespeichert unter: {saved_text_path}")

    if not document_text:
        print(f"Keine Inhalte im PDF gefunden. Pruefe die Datei: {pdf_path}")
        return

    chunks = chunk_text(
        document_text,
        chunk_size=250,
        overlap=25,
    )

    embedding_model = EmbeddingModel()
    chunk_embeddings = embedding_model.embed_texts(chunks)

    vector_store = LocalVectorStore("vector_db_elektrotechnik_3")
    vector_store.add(chunks, chunk_embeddings)
    vector_store.save()

    # query = input("Bitte gib deine Frage ein: ")

    # relevant_chunks = retrieve_relevant_chunks(
    #     query=query,
    #     embedding_model=embedding_model,
    #     vector_store=vector_store,
    #     top_k=5,
    # )

    # generator = AnswerGenerator()
    # answer = generator.generate_answer(
    #     query,
    #     relevant_chunks,
    # )

    # print("\nAntwort:")
    # print(answer)

    generator = AnswerGenerator()
    total_tokens = 0
    total_retrieval_time = 0.0

    for number, query in enumerate(QUESTIONS, start=1):
        print(f"\n{'=' * 70}")
        print(f"Frage {number}: {query}")


        start_time = time.perf_counter()

        relevant_chunks = retrieve_relevant_chunks(
            query=query,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=5,
            num_query_variants=5,
            fusion_search_k=10,
            rrf_k=60,
        )

        retrieval_time = time.perf_counter() - start_time
        total_retrieval_time += retrieval_time
  

        answer, used_tokens = generator.generate_answer(
            query,
            relevant_chunks,
        )

        total_tokens += used_tokens

        print(f"\nAntwort {number}:")
        print(answer)



    print(
        f"\nGesamte Retrieval-Laufzeit: "
        f"{total_retrieval_time:.2f} Sekunden"
    )

    print(f"\nGesamter Tokenverbrauch bei Fusion RAG Top-k = 5: {total_tokens}")

if __name__ == "__main__":
    main()
