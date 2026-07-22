import os

from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingModel:
    """
    Kapselt das Embedding-Modell.

    Dasselbe Modell wird fuer Dokument-Chunks und Nutzeranfragen verwendet.
    """

    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv(
            "OPENAI_EMBEDDING_MODEL", 
            "text-embedding-3-small")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Erstellt Embeddings fuer mehrere Texte.
        """
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """
        Erstellt ein Embedding fuer eine einzelne Nutzeranfrage.
        """
        return self.embed_texts([query])[0]
