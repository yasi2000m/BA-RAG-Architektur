import json
from pathlib import Path

import numpy as np


class LocalVectorStore:
    """
    Eine einfache lokale Vektordatenbank.

    Chunks werden als JSON gespeichert, Embeddings als NumPy-Datei.
    Die Aehnlichkeitssuche nutzt Cosine Similarity.
    """

    def __init__(self, storage_dir: str = "vector_db") -> None:
        self.storage_dir = Path(storage_dir)
        self.chunks_path = self.storage_dir / "chunks.json"
        self.embeddings_path = self.storage_dir / "embeddings.npy"
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: list[str], embeddings: list[list[float]]) -> None:
        """
        Speichert Chunks und passende Embeddings im Speicherobjekt.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Die Anzahl der Chunks und Embeddings muss gleich sein.")

        self.chunks = chunks
        self.embeddings = np.array(embeddings, dtype=np.float32)

    def save(self) -> None:
        """
        Persistiert die Vektordatenbank lokal auf der Festplatte.
        """
        if self.embeddings is None:
            raise ValueError("Es wurden noch keine Embeddings gespeichert.")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(self.embeddings_path, self.embeddings)

    def load(self) -> None:
        """
        Laedt eine bereits gespeicherte lokale Vektordatenbank.
        """
        if not self.chunks_path.exists() or not self.embeddings_path.exists():
            raise FileNotFoundError("Keine gespeicherte Vektordatenbank gefunden.")

        self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(self.embeddings_path)

    def similarity_search_with_scores(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[tuple[int, str, float]]:
        """
        Gibt die top_k relevantesten Chunks mit Index und Score zurueck.
        """
        if self.embeddings is None or not self.chunks:
            raise ValueError("Die Vektordatenbank ist leer.")

        query_vector = np.array(query_embedding, dtype=np.float32)

        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vector)

        similarities = (
            (self.embeddings @ query_vector)
            / (embedding_norms * query_norm)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (
                int(index),
                self.chunks[int(index)],
                float(similarities[int(index)]),
            )
            for index in top_indices
        ]

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[str]:
        """
        Gibt die top_k relevantesten Chunks anhand semantischer Aehnlichkeit zurueck.
        """
        return [
            chunk
            for _, chunk, _ in self.similarity_search_with_scores(
                query_embedding,
                top_k=top_k,
            )
        ]
