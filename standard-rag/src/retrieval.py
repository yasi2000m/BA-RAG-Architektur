from embeddings import EmbeddingModel
from vector_store import LocalVectorStore


def retrieve_relevant_chunks(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store: LocalVectorStore,
    top_k: int = 3,
) -> list[str]:
    """
    Vektorisiert die Nutzeranfrage und ruft die relevantesten Chunks ab.
    """
    query_embedding = embedding_model.embed_query(query)
    return vector_store.similarity_search(query_embedding, top_k=top_k)
