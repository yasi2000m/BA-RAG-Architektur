
from pathlib import Path

from questions import QUESTIONS
from chunking import chunk_text
from document_loader import load_pdf_with_visuals
from generation import AnswerGenerator
from graph_store import KnowledgeGraphExtractor, KnowledgeGraphStore
from retrieval import retrieve_graph_context


def main() -> None:
    pdf_path = (
        "/Users/yasi/Documents/New project/BA-RAG-Architektur/graph-rag-basic/data/Elektrotechnik 3.pdf"
    )

    saved_text_path = Path(
        "/Users/yasi/Documents/New project/BA-RAG-Architektur/graph-rag-basic/data/geladener_text.txt"
    )

    # Bereits gespeicherten Dokumenttext verwenden.
    if saved_text_path.exists():
        print("Gespeicherter Dokumenttext wird verwendet.")
        document_text = saved_text_path.read_text(encoding="utf-8")

    # Falls noch kein Text gespeichert wurde, PDF-Loader einmalig ausfuehren.
    else:
        print("Kein gespeicherter Text gefunden. PDF-Loader wird ausgefuehrt.")

        document_text = load_pdf_with_visuals(pdf_path)

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

    graph_store = KnowledgeGraphStore("graph_db_elektrotechnik_3")

    # Bereits gespeicherten Knowledge Graph verwenden.
    if graph_store.graph_path.exists():
        print("Gespeicherter Knowledge Graph wird verwendet.")
        graph_store.load()

    # Falls noch kein Knowledge Graph gespeichert wurde, einmalig erstellen.
    else:
        print("Kein gespeicherter Knowledge Graph gefunden.")
        print("Knowledge Graph wird erstellt.")

        graph_extractor = KnowledgeGraphExtractor()

        graph_store.build_from_chunks(
            chunks,
            graph_extractor,
        )

        graph_store.save()

        print(
            f"Knowledge Graph gespeichert unter: "
            f"{graph_store.graph_path}"
        )

    generator = AnswerGenerator()
    total_tokens = 0

    max_entities = 150
    max_relationships = 150

    for number, query in enumerate(QUESTIONS, start=1):
        print(f"\n{'=' * 70}")
        print(f"Frage {number}: {query}")

        rag_context = graph_store.query_subgraph(
            query=query,
            max_depth=2,
            max_entities= max_entities,
            max_relationships= max_relationships,
        )

        answer, used_tokens = generator.generate_answer(
            query,
            rag_context.context,
        )

        total_tokens += used_tokens

        print(f"\nAntwort {number}:")
        print(answer)

        actual_entities = len(rag_context.entity_names)
        actual_relationships = len(rag_context.relationships)

        print(
            f"\nAus max. Entitäten: {max_entities} und "
            f"max. Beziehungen: {max_relationships} ergibt sich "
            f"Anzahl tatsächliche Entitäten: {actual_entities} und "
            f"Anzahl tatsächliche Beziehungen: {actual_relationships}"
)

    print(
        f"\nGesamter Tokenverbrauch bei max_depth = 2: "
        f"{total_tokens}"
    )


if __name__ == "__main__":
    main()
