import json
import os
import re

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from embeddings import EmbeddingModel
from vector_store import LocalVectorStore


# Definiert den Datentyp eines einzelnen Retrieval-Treffers: 
# Chunk-Index, Chunk-Text und Similarity-Score.
RankedChunk = tuple[int, str, float]

# Verarbeitet die vom LLM erzeugten Query-Varianten. 
# Die Varianten werden bevorzugt aus einer JSON-Liste gelesen. 
# Falls kein gültiges JSON vorliegt, wird versucht, nummerierte oder zeilenweise ausgegebene Varianten aus dem Text zu extrahieren. 
# Doppelte Varianten und die ursprüngliche Nutzerfrage werden entfernt
def _parse_query_variants(
    content: str,
    original_query: str,
    num_variants: int,
) -> list[str]:
    variants: list[str] = []

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            variants = [str(item).strip() for item in parsed]
    except json.JSONDecodeError:
        lines = content.splitlines()
        variants = [
            re.sub(r"^\s*[-*]?\s*\d*[\).\:]?\s*", "", line).strip(" \"'")
            for line in lines
        ]

    deduplicated: list[str] = []
    seen = {original_query.casefold()}

    for variant in variants:
        if not variant:
            continue

        key = variant.casefold()
        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(variant)

        if len(deduplicated) >= num_variants:
            break

    return deduplicated


# Erzeugt mithilfe eines LLM mehrere semantisch unterschiedliche Varianten der ursprünglichen Nutzerfrage. 
# Zusätzlich wird der Tokenverbrauch der Query-Generierung zurückgegeb
def generate_query_variants(
    query: str,
    num_variants: int = 4,
    model_name: str | None = None,
) -> tuple[list[str], int]:

    if num_variants <= 0:
        return []

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = model_name or os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")

    prompt = f"""Erzeuge {num_variants} semantisch unterschiedliche Suchanfragen fuer Retrieval.
Die Varianten sollen Synonyme, korrigierte Schreibweisen und unterschiedliche fachliche Blickwinkel abdecken.
Behalte die Bedeutung der urspruenglichen Nutzerfrage bei.
Gib ausschliesslich eine JSON-Liste mit Strings zurueck.

Nutzerfrage:
{query}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content or "[]"
    query_generation_tokens = (
        response.usage.total_tokens
        if response.usage
        else 0
    )

    variants = _parse_query_variants(
        content,
        original_query=query,
        num_variants=num_variants,
    )

    return variants, query_generation_tokens


# Erstellt die vollständige Query-Liste für Fusion RAG. 
# Kombiniert die ursprüngliche Nutzerfrage mit den vom LLM erzeugten Suchvarianten und entfernt doppelte Einträge.
def build_fusion_queries(
    query: str,
    num_query_variants: int,
) -> tuple[list[str], int]:
    """
    Kombiniert Originalfrage und LLM-Varianten fuer paralleles Retrieval.
    """
    try:
        variants, query_generation_tokens = generate_query_variants(
            query,
            num_variants=num_query_variants,
        )
    except Exception as error:
        print(f"Query-Varianten konnten nicht erzeugt werden: {error}")
        variants = []
        query_generation_tokens = 0

    queries: list[str] = []
    seen: set[str] = set()

    for fusion_query in [query, *variants]:
        key = fusion_query.casefold()
        if key in seen:
            continue

        seen.add(key)
        queries.append(fusion_query)

    return queries, query_generation_tokens

# Führt mehrere Retrieval-Rankings mithilfe von Reciprocal Rank Fusion (RRF) zu einem Gesamtranking zusammen. 
# Chunks, die in mehreren Ergebnislisten weit oben vorkommen, erhalten einen höheren Gesamtwert.
def reciprocal_rank_fusion(
    ranked_result_lists: list[list[RankedChunk]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[str]:
    """
    Fuehrt mehrere Ergebnislisten mit Reciprocal Rank Fusion zusammen.
    """
    fused_scores: dict[int, float] = {}
    chunks_by_index: dict[int, str] = {}
    best_similarity_by_index: dict[int, float] = {}

    for ranked_results in ranked_result_lists:
        for rank, (chunk_index, chunk, similarity) in enumerate(
            ranked_results,
            start=1,
        ):
            fused_scores[chunk_index] = (
                fused_scores.get(chunk_index, 0.0)
                + 1.0 / (rrf_k + rank)
            )
            chunks_by_index[chunk_index] = chunk
            best_similarity_by_index[chunk_index] = max(
                best_similarity_by_index.get(chunk_index, float("-inf")),
                similarity,
            )

    ranked_indices = sorted(
        fused_scores,
        key=lambda index: (
            fused_scores[index],
            best_similarity_by_index[index],
        ),
        reverse=True,
    )

    return [
        chunks_by_index[index]
        for index in ranked_indices[:top_k]
    ]


# Führt den vollständigen Fusion-RAG-Retrieval-Prozess aus. 
# Die Nutzerfrage wird erweitert, jede Query wird separat vektorbasiert durchsucht und die einzelnen Rankings werden anschließend mithilfe von Reciprocal Rank Fusion zusammengeführt. 
# Gibt die final ausgewählten Chunks und den Tokenverbrauch der Query-Generierung zurück.
def retrieve_relevant_chunks(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store: LocalVectorStore,
    top_k: int = 5,
    num_query_variants: int = 4,
    fusion_search_k: int = 10,
    rrf_k: int = 60,
) -> list[str]:
    """
    Fusion RAG:
    Erzeugt Query-Varianten, sucht pro Variante und fusioniert die Rankings per RRF.
    """
    fusion_queries, query_generation_tokens = build_fusion_queries(
        query,
        num_query_variants=num_query_variants,
    )

    query_embeddings = embedding_model.embed_texts(fusion_queries)

    ranked_result_lists: list[list[RankedChunk]] = []

    for fusion_query, query_embedding in zip(fusion_queries, query_embeddings):
        ranked_results = vector_store.similarity_search_with_scores(
            query_embedding,
            top_k=fusion_search_k,
        )
        ranked_result_lists.append(ranked_results)
        # print(f"Fusion-Suche fuer Query: {fusion_query}")

    unique_chunk_indices = {
        chunk_index
        for ranked_results in ranked_result_lists
        for chunk_index, _, _ in ranked_results
    }

    num_unique_chunks = len(unique_chunk_indices)

    theoretical_hits = len(fusion_queries) * fusion_search_k

    print(f"Theoretische Treffer: {theoretical_hits}")
    print(f"Unterschiedliche Chunks vor RRF: {num_unique_chunks}") 

    relevant_chunks = reciprocal_rank_fusion(
        ranked_result_lists,
        top_k=top_k,
        rrf_k=rrf_k,
    )

    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = sum(
        len(encoding.encode(chunk))
        for chunk in relevant_chunks
    )

    # print(f"Fusion Queries: {len(fusion_queries)}")
    # print(f"Fusion Search K pro Query: {fusion_search_k}")
    # print(f"Final Top K nach RRF: {top_k}")
    # print(f"Anzahl Kontext-Tokens: {token_count}")

    return relevant_chunks, query_generation_tokens


# if __name__ == "__main__":
#     from embeddings import EmbeddingModel
#     from vector_store import LocalVectorStore

#     query = "Was ist der Unterschied zwischen Strom und Spannung?"

#     embedding_model = EmbeddingModel()

#     vector_store = LocalVectorStore("vector_db_elektrotechnik_3")
#     vector_store.load()

#     retrieve_relevant_chunks(
#         query=query,
#         embedding_model=embedding_model,
#         vector_store=vector_store,
#         top_k=5,
#     )
