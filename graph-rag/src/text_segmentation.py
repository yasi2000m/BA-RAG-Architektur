def segment_text(
    text: str,
    segment_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """
    Teilt einen Text in ueberlappende Textabschnitte fuer die Graph-Extraktion.

    Diese Textabschnitte sind kein Retrieval-Kontext. Sie dienen nur dazu,
    lange Dokumente schrittweise in Entitaeten und Beziehungen umzuwandeln.
    """
    if segment_size <= 0:
        raise ValueError("segment_size muss groesser als 0 sein.")

    if overlap < 0:
        raise ValueError("overlap darf nicht negativ sein.")

    if overlap >= segment_size:
        raise ValueError("overlap muss kleiner als segment_size sein.")

    words = text.split()
    segments: list[str] = []

    start = 0
    while start < len(words):
        end = start + segment_size
        segment = " ".join(words[start:end])

        if segment:
            segments.append(segment)

        start += segment_size - overlap

    return segments
