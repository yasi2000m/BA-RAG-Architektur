from document_loader import load_pdf
from generation import AnswerGenerator
from graph_store import KnowledgeGraphExtractor, KnowledgeGraphStore
from retrieval import retrieve_graph_context
from text_segmentation import segment_text


def main() -> None:
    """
    Fuehrt die komplette Graph-RAG-Pipeline aus:

    1. Ein PDF-Dokument laden inklusive Text, Tabellen und Bildern
    2. Textabschnitte nur fuer die Graph-Extraktion bilden
    3. Entitaeten und Beziehungen extrahieren
    4. Knowledge Graph lokal speichern
    5. Nutzeranfrage entgegennehmen
    6. Spezifisches Kontextwissen aus dem Graph abrufen
    7. Prompt mit Graph-Kontext erstellen
    8. Antwort mit LLM generieren
    9. Antwort ausgeben
    """
    pdf_path = input("Pfad zur PDF-Datei [data/elektrotechnik_3.pdf]: ").strip()
    pdf_path = pdf_path or "data/elektrotechnik_3.pdf"

    document_text = load_pdf(pdf_path)

    if not document_text:
        print(f"Keine Inhalte im PDF gefunden. Pruefe die Datei: {pdf_path}")
        return

    text_segments = segment_text(document_text, segment_size=250, overlap=50)

    graph_store = KnowledgeGraphStore("graph_db_elektrotechnik_3")
    graph_extractor = KnowledgeGraphExtractor()
    graph_store.build_from_segments(text_segments, graph_extractor)
    graph_store.save()

    query = input("Bitte gib deine Frage ein: ")

    rag_context = retrieve_graph_context(
        query=query,
        graph_store=graph_store,
    )

    generator = AnswerGenerator()
    answer = generator.generate_answer(query, rag_context.graph_context)

    print("\nAntwort:")
    print(answer)

    print("\nVerwendeter Graph-Kontext:")
    print(rag_context.graph_context or "Kein Graph-Kontext gefunden.")


if __name__ == "__main__":
    main()
