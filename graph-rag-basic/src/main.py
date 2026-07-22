from chunking import chunk_text
from document_loader import load_pdf_with_visuals
from generation import AnswerGenerator
from graph_store import KnowledgeGraphExtractor, KnowledgeGraphStore
from retrieval import retrieve_graph_context


DEFAULT_PDF_PATH = "data/elektrotechnik_3.pdf"
MAX_PAGES = 20
GRAPH_DB_PATH = "graph_db_elektrotechnik_3_first_20_full_visuals"


def main() -> None:
    """
    Fuehrt die komplette Basic-Graph-RAG-Pipeline aus:

    1. PDF-Dokument laden
    2. Text in Chunks aufteilen
    3. Entitaeten und Beziehungen extrahieren
    4. Knowledge Graph lokal speichern
    5. Nutzeranfrage entgegennehmen
    6. Relevante Entitaeten finden
    7. Graph ueber Beziehungen durchlaufen
    8. Graph-Kontext an das LLM geben
    9. Antwort ausgeben
    """
    document_text = load_pdf_with_visuals(DEFAULT_PDF_PATH, max_pages=MAX_PAGES)

    if not document_text:
        print(f"Keine Inhalte im PDF gefunden. Pruefe die Datei: {DEFAULT_PDF_PATH}")
        return

    chunks = chunk_text(document_text, chunk_size=250, overlap=50)

    graph_store = KnowledgeGraphStore(GRAPH_DB_PATH)
    graph_extractor = KnowledgeGraphExtractor()
    graph_store.build_from_chunks(chunks, graph_extractor)
    graph_store.save()

    query = input("Bitte gib deine Frage ein: ")

    rag_context = retrieve_graph_context(
        query=query,
        graph_store=graph_store,
        max_depth=2,
    )

    generator = AnswerGenerator()
    answer = generator.generate_answer(query, rag_context.graph_context)

    print("\nAntwort:")
    print(answer)

    print("\nVerwendeter Graph-Kontext:")
    print(rag_context.graph_context or "Kein Graph-Kontext gefunden.")


if __name__ == "__main__":
    main()
