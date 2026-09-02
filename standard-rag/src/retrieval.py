# Dieses Modul ist dein Retrieval-Modul. Es bekommt eine Nutzerfrage, wandelt sie in ein Embedding um,
# sucht damit im Vector Store nach den passendsten Chunks und zählt anschließend, wie viele Tokens diese gefundenen Chunks insgesamt enthalten.

import tiktoken

from embeddings import EmbeddingModel
from vector_store import LocalVectorStore


def retrieve_relevant_chunks(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store: LocalVectorStore,
    top_k: int = 5,
) -> list[str]:
    """
    Vektorisiert die Nutzeranfrage und ruft die relevantesten Chunks ab.
    """
    query_embedding = embedding_model.embed_query(query)

    relevant_chunks = vector_store.similarity_search(
        query_embedding,
        top_k=top_k,
    )

    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = sum(
        len(encoding.encode(chunk))
        for chunk in relevant_chunks
    )

    print(f"Top K: {top_k}")
    print(f"Anzahl Kontext-Tokens: {token_count}")

    return relevant_chunks


if __name__ == "__main__":
    from embeddings import EmbeddingModel
    from vector_store import LocalVectorStore

    query = "Was ist der Unterschied zwischen Strom und Spannung?"

    embedding_model = EmbeddingModel()

    vector_store = LocalVectorStore("vector_db_elektrotechnik_3")
    vector_store.load()

    retrieve_relevant_chunks(
        query=query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=5,
    )