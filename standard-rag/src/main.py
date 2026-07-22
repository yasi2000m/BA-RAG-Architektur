from chunking import chunk_text
from document_loader import load_pdf
from embeddings import EmbeddingModel
from generation import AnswerGenerator
from retrieval import retrieve_relevant_chunks
from vector_store import LocalVectorStore


def main() -> None:
    """
    Fuehrt die komplette Standard-RAG-Pipeline aus:

    1. Ein PDF-Dokument laden inklusive Text, Tabellen und Bildern
    2. Chunking
    3. Embedding der Chunks
    4. Speicherung in lokaler Vektordatenbank
    5. Nutzeranfrage entgegennehmen
    6. Anfrage einbetten
    7. Relevanteste Chunks abrufen
    8. Prompt erstellen
    9. Antwort mit LLM generieren
    10. Antwort ausgeben
    """
    pdf_path = input("Pfad zur PDF-Datei [data/elektrotechnik_3.pdf]: ").strip()
    pdf_path = pdf_path or "data/elektrotechnik_3.pdf"

    document_text = load_pdf(pdf_path)

    if not document_text:
        print(f"Keine Inhalte im PDF gefunden. Pruefe die Datei: {pdf_path}")
        return

    chunks = chunk_text(document_text, chunk_size=250, overlap=50)

    embedding_model = EmbeddingModel()
    chunk_embeddings = embedding_model.embed_texts(chunks)

    vector_store = LocalVectorStore("vector_db_elektrotechnik_3")
    vector_store.add(chunks, chunk_embeddings)
    vector_store.save()

    query = input("Bitte gib deine Frage ein: ")

    relevant_chunks = retrieve_relevant_chunks(
        query=query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=5,
    )

    generator = AnswerGenerator()
    answer = generator.generate_answer(query, relevant_chunks)

    print("\nAntwort:")
    print(answer)


if __name__ == "__main__":
    main()
