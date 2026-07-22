import streamlit as st

from document_loader import load_pdf_text_with_targeted_visuals
from generation import AnswerGenerator
from graph_store import KnowledgeGraphExtractor, KnowledgeGraphStore
from retrieval import retrieve_graph_context
from text_segmentation import segment_text


DEFAULT_PDF_PATH = "data/elektrotechnik_3.pdf"
GRAPH_DB_PATH = "graph_db_elektrotechnik_3"
MAX_GRAPH_EDGES = 80
ENTITY_TYPE_COLORS = {
    "Konzept": "#29b6f6",
    "Person": "#ff2d6f",
    "Organisation": "#2dd36f",
    "Ort": "#ffb020",
    "Bauteil": "#7c4dff",
    "Regel": "#00bfa5",
    "Messwert": "#f06292",
    "Sonstiges": "#8e7cc3",
}


def build_knowledge_base(
    pdf_path: str,
) -> tuple[KnowledgeGraphStore, int]:
    """Laedt das PDF und erstellt einen lokalen Knowledge Graph."""
    document_text = load_pdf_text_with_targeted_visuals(
        pdf_path,
        visual_keywords=["Sicherheitsregeln"],
    )

    if not document_text:
        raise ValueError("Im PDF wurden keine auswertbaren Inhalte gefunden.")

    text_segments = segment_text(document_text, segment_size=250, overlap=50)

    graph_store = KnowledgeGraphStore(GRAPH_DB_PATH)
    graph_extractor = KnowledgeGraphExtractor()
    graph_store.build_from_segments(text_segments, graph_extractor)
    graph_store.save()

    return graph_store, len(text_segments)


def load_or_build_knowledge_base() -> tuple[KnowledgeGraphStore, int, bool]:
    """Laedt vorhandenen Knowledge Graph oder baut ihn einmalig auf."""
    graph_store = KnowledgeGraphStore(GRAPH_DB_PATH)

    try:
        graph_store.load()
        segment_count = len({segment_id for entity in graph_store.entities.values() for segment_id in entity.segment_ids})
        return graph_store, segment_count, False
    except FileNotFoundError:
        graph_store, segment_count = build_knowledge_base(DEFAULT_PDF_PATH)
        return graph_store, segment_count, True


def build_graphviz_dot(graph_store: KnowledgeGraphStore, max_edges: int = MAX_GRAPH_EDGES) -> str:
    """Erstellt eine farbige Netzwerkdarstellung des Knowledge Graph."""
    relationships = _relationships_for_visual(graph_store, max_edges=max_edges)
    graph_entities = {
        relationship.source
        for relationship in relationships
    } | {
        relationship.target
        for relationship in relationships
    }

    if not relationships and graph_store.entities:
        graph_entities = set(list(graph_store.entities)[:max_edges])

    lines = [
        "digraph knowledge_graph {",
        '  graph [layout="sfdp", overlap="scale", sep="+24", K="1.15", splines="curved", bgcolor="transparent", outputorder="edgesfirst", pad="0.35"];',
        '  node [shape=circle, style="filled", color="white", penwidth=2, fontname="Arial", fontcolor="white", fixedsize=true];',
        '  edge [color="#2f3340", fontname="Arial", fontsize=9, fontcolor="#111827", arrowsize=0.65, penwidth=1.2];',
    ]

    degrees = _entity_degrees(relationships)
    for entity_name in sorted(graph_entities):
        safe_name = _dot_escape(entity_name)
        entity = graph_store.entities.get(entity_name)
        entity_type = entity.entity_type if entity is not None else "Sonstiges"
        color = ENTITY_TYPE_COLORS.get(entity_type, ENTITY_TYPE_COLORS["Sonstiges"])
        degree = degrees.get(entity_name, 1)
        size = min(2.2, 0.78 + degree * 0.13)
        label_name = _short_label(entity_name)
        label = label_name if entity is None else f"{label_name}\\n{entity_type}"
        lines.append(
            f'  "{safe_name}" [label="{_dot_escape(label)}", '
            f'fillcolor="{color}", width={size:.2f}, height={size:.2f}, fontsize=12];'
        )

    for relationship in relationships:
        source = _dot_escape(relationship.source)
        target = _dot_escape(relationship.target)
        relation = _dot_escape(_short_relation(relationship.relation))
        lines.append(f'  "{source}" -> "{target}" [label="{relation}"];')

    lines.append("}")
    return "\n".join(lines)


def build_subgraph_dot(
    graph_store: KnowledgeGraphStore,
    entity_names: list[str],
    relationships: list,
) -> str:
    """Erstellt die Visualisierung des tatsaechlich genutzten Teilgraphen."""
    graph_entities = set(entity_names)
    for relationship in relationships:
        graph_entities.add(relationship.source)
        graph_entities.add(relationship.target)

    lines = [
        "digraph used_subgraph {",
        '  graph [layout="dot", rankdir="LR", splines="curved", bgcolor="transparent", pad="0.25"];',
        '  node [shape=circle, style="filled", color="white", penwidth=2.4, fontname="Arial", fontcolor="white", fixedsize=true];',
        '  edge [color="#111827", fontname="Arial", fontsize=10, fontcolor="#111827", arrowsize=0.75, penwidth=1.5];',
    ]

    degrees = _entity_degrees(relationships)
    for entity_name in sorted(graph_entities):
        entity = graph_store.entities.get(entity_name)
        entity_type = entity.entity_type if entity is not None else "Sonstiges"
        color = ENTITY_TYPE_COLORS.get(entity_type, ENTITY_TYPE_COLORS["Sonstiges"])
        degree = degrees.get(entity_name, 1)
        size = min(2.0, 0.88 + degree * 0.16)
        label_name = _short_label(entity_name)
        label = label_name if entity is None else f"{label_name}\\n{entity_type}"
        lines.append(
            f'  "{_dot_escape(entity_name)}" [label="{_dot_escape(label)}", '
            f'fillcolor="{color}", width={size:.2f}, height={size:.2f}, fontsize=12];'
        )

    for relationship in relationships:
        lines.append(
            f'  "{_dot_escape(relationship.source)}" -> "{_dot_escape(relationship.target)}" '
            f'[label="{_dot_escape(_short_relation(relationship.relation))}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def _relationships_for_visual(graph_store: KnowledgeGraphStore, max_edges: int) -> list:
    """Waehlt stark vernetzte Beziehungen fuer eine lesbare Visualisierung."""
    degrees = _entity_degrees(graph_store.relationships)
    return sorted(
        graph_store.relationships,
        key=lambda relationship: (
            degrees.get(relationship.source, 0) + degrees.get(relationship.target, 0),
            relationship.relation,
        ),
        reverse=True,
    )[:max_edges]


def _entity_degrees(relationships: list) -> dict[str, int]:
    degrees: dict[str, int] = {}
    for relationship in relationships:
        degrees[relationship.source] = degrees.get(relationship.source, 0) + 1
        degrees[relationship.target] = degrees.get(relationship.target, 0) + 1
    return degrees


def _short_label(value: str, max_length: int = 22) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 1]}."


def _short_relation(value: str, max_length: int = 24) -> str:
    relation = value.replace(" ", "_").upper()
    if len(relation) <= max_length:
        return relation
    return f"{relation[: max_length - 1]}."


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def entity_rows(graph_store: KnowledgeGraphStore) -> list[dict[str, str]]:
    """Bereitet Entitaeten fuer die Tabellenansicht auf."""
    return [
        {
            "Name": entity.name,
            "Typ": entity.entity_type,
            "Beschreibung": entity.description,
        }
        for entity in graph_store.entities.values()
    ]


def relationship_rows(graph_store: KnowledgeGraphStore) -> list[dict[str, str]]:
    """Bereitet Beziehungen fuer die Tabellenansicht auf."""
    return [
        {
            "Quelle": relationship.source,
            "Beziehung": relationship.relation,
            "Ziel": relationship.target,
            "Beschreibung": relationship.description,
        }
        for relationship in graph_store.relationships
    ]


st.set_page_config(page_title="Graph-RAG", page_icon="PDF", layout="wide")

st.markdown(
    """
    <style>
    .stGraphVizChart {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background:
            radial-gradient(circle at 20% 10%, rgba(41, 182, 246, 0.10), transparent 26%),
            radial-gradient(circle at 80% 20%, rgba(124, 77, 255, 0.10), transparent 28%),
            linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        padding: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Graph-RAG fuer Elektrotechnik")
st.write("Die Elektrotechnik-PDF wird als lokaler Knowledge Graph verarbeitet.")

if "graph_store" not in st.session_state or st.session_state.get("graph_db_path") != GRAPH_DB_PATH:
    with st.spinner("Wissensbasis wird geladen ..."):
        try:
            graph_store, segment_count, was_built = load_or_build_knowledge_base()
            st.session_state.graph_store = graph_store
            st.session_state.segment_count = segment_count
            st.session_state.entity_count = len(graph_store.entities)
            st.session_state.relationship_count = len(graph_store.relationships)
            st.session_state.was_built = was_built
            st.session_state.graph_db_path = GRAPH_DB_PATH
        except Exception as error:
            st.session_state.graph_store = None
            st.error(f"Fehler beim Laden der Wissensbasis: {error}")

if st.session_state.graph_store is not None:
    st.success(
        "Wissensbasis bereit. "
        f"Entitaeten: {st.session_state.entity_count} | "
        f"Beziehungen: {st.session_state.relationship_count}"
    )
    st.caption(f"PDF: {DEFAULT_PDF_PATH} | Graphdatenbank: {GRAPH_DB_PATH}")

    with st.expander("Knowledge Graph anzeigen", expanded=False):
        visible_edges = st.slider(
            "Anzahl sichtbarer Beziehungen",
            min_value=10,
            max_value=max(10, st.session_state.relationship_count),
            value=min(MAX_GRAPH_EDGES, max(10, st.session_state.relationship_count)),
            step=10,
        )
        st.graphviz_chart(
            build_graphviz_dot(st.session_state.graph_store, max_edges=visible_edges),
            use_container_width=True,
        )
        if st.session_state.relationship_count > visible_edges:
            st.caption(
                f"Visualisierung zeigt {visible_edges} stark vernetzte Beziehungen. "
                "Die Tabellen darunter enthalten den vollstaendigen Graph."
            )

        st.caption(
            "Farben: Konzept blau, Person rot, Organisation gruen, Ort gelb, "
            "Bauteil violett, Regel tuerkis, Messwert rosa."
        )

        st.markdown("**Entitaeten**")
        st.dataframe(entity_rows(st.session_state.graph_store), use_container_width=True)

        st.markdown("**Beziehungen**")
        st.dataframe(relationship_rows(st.session_state.graph_store), use_container_width=True)

st.divider()

question = st.text_input("Deine Frage")

if st.button("Antwort generieren", disabled=not question):
    if st.session_state.graph_store is None:
        st.warning("Die Wissensbasis konnte nicht geladen werden.")
    else:
        with st.spinner("Relevante Graph-Beziehungen werden gesucht ..."):
            rag_context = retrieve_graph_context(
                query=question,
                graph_store=st.session_state.graph_store,
            )

            generator = AnswerGenerator()
            answer = generator.generate_answer(question, rag_context.graph_context)

        st.subheader("Antwort")
        st.write(answer)

        with st.expander("Verwendeten Graph-Kontext anzeigen"):
            st.text(rag_context.graph_context or "Kein Graph-Kontext gefunden.")

        with st.expander("Benutzten Teilgraph anzeigen", expanded=True):
            if rag_context.entity_names or rag_context.relationships:
                st.graphviz_chart(
                    build_subgraph_dot(
                        st.session_state.graph_store,
                        rag_context.entity_names,
                        rag_context.relationships,
                    ),
                    use_container_width=True,
                )

                st.markdown("**Benutzte Entitaeten**")
                used_entities = {
                    name: st.session_state.graph_store.entities[name]
                    for name in rag_context.entity_names
                    if name in st.session_state.graph_store.entities
                }
                st.dataframe(
                    [
                        {
                            "Name": entity.name,
                            "Typ": entity.entity_type,
                            "Beschreibung": entity.description,
                        }
                        for entity in used_entities.values()
                    ],
                    use_container_width=True,
                )

                st.markdown("**Benutzte Beziehungen**")
                st.dataframe(
                    [
                        {
                            "Quelle": relationship.source,
                            "Beziehung": relationship.relation,
                            "Ziel": relationship.target,
                            "Beschreibung": relationship.description,
                        }
                        for relationship in rag_context.relationships
                    ],
                    use_container_width=True,
                )
            else:
                st.info("Fuer diese Frage wurde kein passender Teilgraph gefunden.")
