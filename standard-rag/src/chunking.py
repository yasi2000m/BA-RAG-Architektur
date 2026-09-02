

# Diese Funktion zerlegt einen längeren Text in mehrere kleinere Chunks. Die Aufteilung erfolgt wortbasiert.
# Teilt einen Text wortbasiert in überlappende Textabschnitte auf.
# chunk_size bestimmt die maximale Anzahl der Wörter pro Chunk.
# overlap bestimmt, wie viele Wörter zwischen zwei aufeinanderfolgenden Chunks wiederholt werden. 
# Zusätzlich werden ungewöhnlich lange Zeichenfolgen in kleinere Abschnitte zerlegt.
# Gibt eine Liste der erzeugten Text-Chunks zurück.

# Parameter:
# text: der gesamte Text, der aufgeteilt werden soll
# chunk_size: maximale Anzahl an Wörtern pro Chunk
# overlap: Anzahl der Wörter, die sich zwei aufeinanderfolgende Chunks überlappen
# -> list[str]: die Funktion gibt eine Liste von Text-Chunks zurück

def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 50,
) -> list[str]:

    ## Prüft, ob eine gültige Chunk-Größe angegeben wurde.

    if chunk_size <= 0:
        raise ValueError("chunk_size muss groesser als 0 sein.")

    if overlap < 0:
        raise ValueError("overlap darf nicht negativ sein.")

    if overlap >= chunk_size:
        raise ValueError("overlap muss kleiner als chunk_size sein.")


# Zerlegt den Text in Wörter und teilt ungewöhnlich lange
# Zeichenfolgen zusätzlich in Abschnitte von maximal 200 Zeichen.

    words = []

    for word in text.split():
        if len(word) > 200:
            words.extend(
                word[i:i + 200]
                for i in range(0, len(word), 200)
            )
        else:
            words.append(word)

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
