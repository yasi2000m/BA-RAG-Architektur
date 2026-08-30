
import time
from pathlib import Path

from questions import QUESTIONS
from chunking import chunk_text
from document_loader import load_pdf_with_visuals
from generation import AnswerGenerator
from graph_store import KnowledgeGraphExtractor, KnowledgeGraphStore
# from retrieval import retrieve_graph_context


def main(query: str) -> str:

    project_dir = Path(__file__).resolve().parents[1]
    root_dir = Path(__file__).resolve().parents[2]

    pdf_path = project_dir / "data" / "ErneuerbareEnergien.pdf"
    saved_text_path = project_dir / "data" / "geladener_text2.txt"
    graph_store = KnowledgeGraphStore(str(root_dir / "graph_db_ErneuebareEnergien"))
    
    '''
    pdf_path = (
        "/Users/yasi/Documents/New project/BA-RAG-Architektur/"
        "graph-rag-basic/data/ErneuerbareEnergien.pdf"
    )

    saved_text_path = Path(
        "/Users/yasi/Documents/New project/BA-RAG-Architektur/"
        "graph-rag-basic/data/geladener_text2.txt"
    )
    '''

    # Graph existiert -> direkt laden
    if graph_store.graph_path.exists():
        print("Gespeicherter Knowledge Graph wird verwendet.")
        graph_store.load()

    # Graph fehlt -> einmalig erzeugen
    else:
        print("graph nicht gefunden. wird generiert...")
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
    
    
        # Gesamtlaufzeit startet nach dem Document Loader.
        total_start_time = time.perf_counter()
    
    
        chunks = chunk_text(
            document_text,
            chunk_size=250,
            overlap=25,
        )
    
        graph_store = KnowledgeGraphStore("graph_db_ErneuebareEnergien")
    
        graph_creation_tokens = 0

        print("Knowledge Graph wird erstellt.")

        graph_extractor = KnowledgeGraphExtractor()

        graph_store.build_from_chunks(
            chunks,
            graph_extractor,
        )

        graph_store.save()

        # Falls der Extractor Tokenverbrauch speichert.
        graph_creation_tokens = graph_extractor.total_tokens

        print(
            f"Knowledge Graph gespeichert unter: "
            f"{graph_store.graph_path}"
        )


    # Speicherbedarf des Knowledge Graph.
    storage_bytes = graph_store.graph_path.stat().st_size
    storage_mb = storage_bytes / (1024 * 1024)


    generator = AnswerGenerator()

    total_generation_tokens = 0
    total_answer_time = 0.0

    max_entities = 150
    max_relationships = 150

    max_depth = 2
    max_entities = 150
    max_relationships = 150
    
    rag_context = graph_store.query_subgraph(
        query=query,
        max_depth=max_depth,
        max_entities=max_entities,
        max_relationships=max_relationships,
    )

    answer, used_tokens = generator.generate_answer(
        query,
        rag_context.context,
    )

    actual_entities = len(rag_context.entity_names)
    actual_relationships = len(rag_context.relationships)

    return {
        "answer": answer,
        "used_tokens": used_tokens,
        "max_depth": max_depth,
        "entities": actual_entities,
        "relationships": actual_relationships,
    }

    '''
    for number, query in enumerate(QUESTIONS, start=1):
        print(f"\n{'=' * 70}")
        print(f"Frage {number}: {query}")


        # Antwortzeit startet vor dem Retrieval.
        answer_start_time = time.perf_counter()


        rag_context = graph_store.query_subgraph(
            query=query,
            max_depth=2,
            max_entities=max_entities,
            max_relationships=max_relationships,
        )

        answer, used_tokens = generator.generate_answer(
            query,
            rag_context.context,
        )


        # Antwortzeit = Retrieval + Antwortgenerierung.
        total_answer_time += (
            time.perf_counter() - answer_start_time
        )

        total_generation_tokens += used_tokens


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


    # Gesamtlaufzeit endet nach der letzten Antwort.
    total_runtime = (
        time.perf_counter() - total_start_time
    )


    total_token_usage = (
        graph_creation_tokens
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
        f"Graph-Erstellungs-Tokens: "
        f"{graph_creation_tokens}"
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
'''

if __name__ == "__main__":
    main()
