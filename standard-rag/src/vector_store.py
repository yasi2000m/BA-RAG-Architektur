import json
from pathlib import Path

import numpy as np


class LocalVectorStore:
    """
    Eine einfache lokale Vektordatenbank.

    Chunks werden als JSON gespeichert, Embeddings als NumPy-Datei.
    Die Aehnlichkeitssuche nutzt Cosine Similarity.
    """

    # Bereitet die lokale Vektordatenbank vor und legt fest, wo Chunks und Embeddings gespeichert werden.
    
    def __init__(self, storage_dir: str = "vector_db") -> None:
        self.storage_dir = Path(storage_dir)
        self.chunks_path = self.storage_dir / "chunks.json"
        self.embeddings_path = self.storage_dir / "embeddings.npy"
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None


    # Übernimmt Chunks und ihre Embeddings und hält sie gemeinsam im Speicher.

    def add(self, chunks: list[str], embeddings: list[list[float]]) -> None:
        """
        Speichert Chunks und passende Embeddings im Speicherobjekt.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Die Anzahl der Chunks und Embeddings muss gleich sein.")

        self.chunks = chunks
        self.embeddings = np.array(embeddings, dtype=np.float32)


    # Speichert Chunks und Embeddings dauerhaft auf der Festplatte.

    def save(self) -> None:
        """
        Persistiert die Vektordatenbank lokal auf der Festplatte.
        """
        if self.embeddings is None:
            raise ValueError("Es wurden noch keine Embeddings gespeichert.")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(self.embeddings_path, self.embeddings)


    # Lädt eine bereits gespeicherte Vektordatenbank wieder.    

    def load(self) -> None:
        """
        Laedt eine bereits gespeicherte lokale Vektordatenbank.
        """
        if not self.chunks_path.exists() or not self.embeddings_path.exists():
            raise FileNotFoundError("Keine gespeicherte Vektordatenbank gefunden.")

        self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(self.embeddings_path)


     # Vergleicht das Embedding einer Nutzerfrage mit allen Chunk-Embeddings, sortiert sie nach semantischer Ähnlichkeit und gibt die besten Chunks zurück.   

    def similarity_search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        """
        Gibt die top_k relevantesten Chunks anhand semantischer Aehnlichkeit zurueck.
        """
        if self.embeddings is None or not self.chunks:
            raise ValueError("Die Vektordatenbank ist leer.")

        query_vector = np.array(query_embedding, dtype=np.float32)

        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vector)

        similarities = (self.embeddings @ query_vector) / (embedding_norms * query_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.chunks[index] for index in top_indices]
