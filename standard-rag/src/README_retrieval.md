# README zu `retrieval.py`

Dieses Modul fuehrt den Retrieval-Schritt aus. Retrieval bedeutet: Zur Nutzerfrage werden passende Chunks aus der Vektordatenbank gesucht.

## Imports

```python
from embeddings import EmbeddingModel
```

Die Klasse `EmbeddingModel` wird importiert, damit die Nutzerfrage in einen Vektor umgewandelt werden kann.

```python
from vector_store import LocalVectorStore
```

Die lokale Vektordatenbank wird importiert, damit darin nach aehnlichen Chunks gesucht werden kann.

## Funktion `retrieve_relevant_chunks`

```python
def retrieve_relevant_chunks(
```

Hier beginnt die Retrieval-Funktion.

```python
query: str,
```

`query` ist die Nutzerfrage als Text.

```python
embedding_model: EmbeddingModel,
```

`embedding_model` ist das Embedding-Modell, das die Frage vektorisiert.

```python
vector_store: LocalVectorStore,
```

`vector_store` ist die lokale Vektordatenbank mit Chunks und Embeddings.

```python
top_k: int = 3,
```

`top_k` legt fest, wie viele relevante Chunks zurueckgegeben werden. In der UI wird aktuell `5` verwendet.

```python
) -> list[str]:
```

Die Funktion gibt eine Liste von Text-Chunks zurueck.

```python
"""
Vektorisiert die Nutzeranfrage und ruft die relevantesten Chunks ab.
"""
```

Der Docstring fasst die Aufgabe zusammen.

```python
query_embedding = embedding_model.embed_query(query)
```

Die Nutzerfrage wird in einen Embedding-Vektor umgewandelt.

```python
return vector_store.similarity_search(query_embedding, top_k=top_k)
```

Die Vektordatenbank sucht die aehnlichsten Chunk-Vektoren und gibt die passenden Texte zurueck.
