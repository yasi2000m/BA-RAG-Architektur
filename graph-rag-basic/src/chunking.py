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

    return chunks

