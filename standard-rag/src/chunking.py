def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """
    Teilt einen einzelnen Text in einfache, ueberlappende Chunks auf.

    Die Aufteilung erfolgt wortbasiert. Das ist bewusst einfach gehalten,
    damit der Schritt gut nachvollziehbar bleibt.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size muss groesser als 0 sein.")

    if overlap < 0:
        raise ValueError("overlap darf nicht negativ sein.")

    if overlap >= chunk_size:
        raise ValueError("overlap muss kleiner als chunk_size sein.")

    words = text.split()
    chunks: list[str] = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    print(f"Anzahl Chunks: {len(chunks)}")
    print(f"Speicherbedarf: {sum(len(chunk.encode('utf-8')) for chunk in chunks) / 1024:.2f} KB")

    return chunks



if __name__ == "__main__":
    from embeddings import EmbeddingModel
    from vector_store import LocalVectorStore

    with open("standard-rag/data/geladener_text.txt", "r", encoding="utf-8") as file:
        text = file.read()

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=50,
    )
    
    embedding_model = EmbeddingModel()
    chunk_embeddings = embedding_model.embed_texts(chunks)

    vector_store = LocalVectorStore("vector_db_elektrotechnik_3")
    vector_store.add(chunks, chunk_embeddings)
    vector_store.save()

    print("Vektordatenbank wurde gespeichert.")
