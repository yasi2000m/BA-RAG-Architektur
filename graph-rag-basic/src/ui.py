import streamlit as st

from chunking import chunk_text
from document_loader import load_pdf_with_visuals
from generation import AnswerGenerator
from graph_store import GraphRelationship
from graph_store import KnowledgeGraphExtractor, KnowledgeGraphStore
from retrieval import retrieve_graph_context


DEFAULT_PDF_PATH = "data/Elektrotechnik 3.pdf"
MAX_PAGES = 20
GRAPH_DB_PATH = "graph_db_elektrotechnik_3_first_20_full_visuals"


def load_document() -> str:
    """Laedt das Elektrotechnik-Skript inklusive Bilder, Tabellen und Diagramme."""
    return load_pdf_with_visuals(DEFAULT_PDF_PATH, max_pages=MAX_PAGES)


def build_knowledge_graph() -> tuple[KnowledgeGraphStore, int]:
    """Laedt das Elektrotechnik-Skript und erstellt einen lokalen Knowledge Graph."""
    document_text = load_document()

    if not document_text:
        raise ValueError("Im Elektrotechnik-Skript wurden keine auswertbaren Inhalte gefunden.")

    chunks = chunk_text(document_text, chunk_size=250, overlap=50)

    graph_store = KnowledgeGraphStore(GRAPH_DB_PATH)
    graph_extractor = KnowledgeGraphExtractor()
    graph_store.build_from_chunks(chunks, graph_extractor)
    graph_store.save()

    return graph_store, len(chunks)


def load_or_build_knowledge_graph() -> tuple[KnowledgeGraphStore, int, bool]:
    """Laedt vorhandenen Knowledge Graph oder baut ihn einmalig auf."""
    graph_store = KnowledgeGraphStore(GRAPH_DB_PATH)

    try:
        graph_store.load()
        chunk_count = len(graph_store.chunk_entities)
        return graph_store, chunk_count, False
    except FileNotFoundError:
        graph_store, chunk_count = build_knowledge_graph()
        return graph_store, chunk_count, True


def entity_rows(graph_store: KnowledgeGraphStore) -> list[dict[str, str]]:
    return [
        {
            "Name": entity.name,
            "Typ": entity.entity_type,
            "Beschreibung": entity.description,
        }
        for entity in graph_store.entities.values()
    ]


def relationship_rows(graph_store: KnowledgeGraphStore) -> list[dict[str, str]]:
    return [
        {
            "Quelle": relationship.source,
            "Beziehung": relationship.relation,
            "Ziel": relationship.target,
            "Beschreibung": relationship.description,
        }
        for relationship in graph_store.relationships
    ]


def build_subgraph_dot(
    entity_names: list[str],
    relationships: list[GraphRelationship],
) -> str:
    """Erstellt eine einfache Graphviz-Darstellung des genutzten Teilgraphen."""
    graph_entities = set(entity_names)
    for relationship in relationships:
        graph_entities.add(relationship.source)
        graph_entities.add(relationship.target)

    lines = [
        "digraph used_graph {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.2"];',
        '  node [shape=box, style="rounded,filled", fillcolor="#eef6ff", color="#2563eb", fontname="Arial"];',
        '  edge [color="#334155", fontname="Arial", fontsize=10];',
    ]

    for entity_name in sorted(graph_entities):
        lines.append(f'  "{_dot_escape(entity_name)}";')

    for relationship in relationships:
        lines.append(
            f'  "{_dot_escape(relationship.source)}" -> "{_dot_escape(relationship.target)}" '
            f'[label="{_dot_escape(relationship.relation)}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


st.set_page_config(page_title="Basic Graph-RAG", page_icon="PDF", layout="wide")

st.title("Basic Graph-RAG fuer Elektrotechnik")
st.write("Die ersten 20 Seiten des Elektrotechnik-Skripts werden inklusive Bilder, Tabellen und Diagramme als lokaler Knowledge Graph verarbeitet.")

if "graph_store" not in st.session_state:
    with st.spinner("Knowledge Graph wird geladen ..."):
        try:
            graph_store, chunk_count, was_built = load_or_build_knowledge_graph()
            st.session_state.graph_store = graph_store
            st.session_state.chunk_count = chunk_count
            st.session_state.was_built = was_built
        except Exception as error:
            st.session_state.graph_store = None
            st.error(f"Fehler beim Laden des Knowledge Graph: {error}")

if st.session_state.graph_store is not None:
    graph_store = st.session_state.graph_store
    st.success(
        "Knowledge Graph bereit. "
        "PDFs: 1 | "
        f"Seiten: {MAX_PAGES} | "
        f"Chunks: {st.session_state.chunk_count} | "
        f"Entitaeten: {len(graph_store.entities)} | "
        f"Beziehungen: {len(graph_store.relationships)}"
    )
    st.caption(f"PDF: {DEFAULT_PDF_PATH} | Seiten: 1-{MAX_PAGES} | Graphdatenbank: {GRAPH_DB_PATH}")

    with st.expander("Entitaeten anzeigen"):
        st.dataframe(entity_rows(graph_store), use_container_width=True)

    with st.expander("Beziehungen anzeigen"):
        st.dataframe(relationship_rows(graph_store), use_container_width=True)

st.divider()

question = st.text_input("Deine Frage")

if st.button("Antwort generieren", disabled=not question):
    if st.session_state.graph_store is None:
        st.warning("Der Knowledge Graph konnte nicht geladen werden.")
    else:
        with st.spinner("Graph wird durchsucht und Antwort wird generiert ..."):
            rag_context = retrieve_graph_context(
                query=question,
                graph_store=st.session_state.graph_store,
                max_depth=2,
            )

            generator = AnswerGenerator()
            answer = generator.generate_answer(question, rag_context.graph_context)

        st.subheader("Antwort")
        st.write(answer)

        st.subheader("Verwendeter Graph")
        if rag_context.entity_names or rag_context.relationships:
            st.graphviz_chart(
                build_subgraph_dot(rag_context.entity_names, rag_context.relationships),
                use_container_width=True,
            )
        else:
            st.info("Zu dieser Frage wurde kein passender Teilgraph gefunden.")

        with st.expander("Verwendeter Graph-Kontext"):
            st.write(rag_context.graph_context or "Kein Graph-Kontext gefunden.")
