# Das Embedding-Modul wandelt sowohl die Dokument-Chunks als auch Nutzeranfragen mithilfe desselben Embedding-Modells in numerische Vektoren um. 
# Dadurch können ihre semantischen Ähnlichkeiten später mathematisch bestimmt und relevante Dokumentabschnitte gefunden werden.

import os

from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingModel:

# Lädt API-Key und Modell und initialisiert den Tokenzähler.
    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv(
            "OPENAI_EMBEDDING_MODEL", 
            "text-embedding-3-small")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Tokenverbrauch der Embeddings speichern
        self.total_tokens = 0

# Wandelt mehrere Dokument-Chunks in Embedding-Vektoren um.
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Erstellt Embeddings fuer mehrere Texte in kleinen Gruppen.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        batch_size = 1

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )

            embeddings.extend(
                item.embedding for item in response.data
            )
            # Tokenverbrauch addieren
            if response.usage:
                self.total_tokens += response.usage.total_tokens

        return embeddings
# Wandelt eine einzelne Nutzerfrage in einen Embedding-Vektor um.
    def embed_query(self, query: str) -> list[float]:
        """
        Erstellt ein Embedding fuer eine einzelne Nutzeranfrage.
        """
        return self.embed_texts([query])[0]
