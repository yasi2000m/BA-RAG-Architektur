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
    max_depth: int = 2,
) -> GraphRAGContext:
    """
    Ruft Kontext fuer Graph-RAG ab.

    Die Anfrage wird nicht als Vektor verglichen. Stattdessen werden passende
    Entitaeten gesucht und der Graph wird ueber deren Beziehungen durchlaufen.
    """
    graph_result = graph_store.query_subgraph(
        query=query,
        max_depth=max_depth,
    )

    return GraphRAGContext(
        graph_context=graph_result.context,
        entity_names=graph_result.entity_names,
        relationships=graph_result.relationships,
    )

