from dataclasses import dataclass

from graph_store import GraphRelationship
from graph_store import KnowledgeGraphStore


@dataclass
class GraphRAGContext:
    """Das Ergebnis der Graph-RAG-Retrieval-Phase."""

    graph_context: str
    entity_names: list[str]
    relationships: list[GraphRelationship]


def retrieve_graph_context(
    query: str,
    graph_store: KnowledgeGraphStore,
) -> GraphRAGContext:
    """
    Ruft Kontext fuer Graph-RAG ab.

    Die Nutzerfrage wird direkt gegen den Knowledge Graph ausgewertet.
    Zurueckgegeben werden nur Entitaeten und Beziehungen als spezifisches
    Kontextwissen, keine Textauszuege aus dem Dokument.
    """
    graph_result = graph_store.query_subgraph(query=query)

    return GraphRAGContext(
        graph_context=graph_result.context,
        entity_names=graph_result.entity_names,
        relationships=graph_result.relationships,
    )
