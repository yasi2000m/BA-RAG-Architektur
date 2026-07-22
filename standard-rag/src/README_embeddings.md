# README zu `embeddings.py`

Dieses Modul wandelt Texte in Vektoren um. Diese Vektoren nennt man Embeddings.

Im RAG-System werden sowohl PDF-Chunks als auch Nutzerfragen mit demselben Embedding-Modell vektorisiert. Nur so koennen sie semantisch miteinander verglichen werden.

## Imports

```python
import os
```

`os` liest Umgebungsvariablen aus.

```python
from dotenv import load_dotenv
```

`load_dotenv()` laedt die `.env`-Datei.

```python
from openai import OpenAI
```

`OpenAI` ist der API-Client fuer Embeddings.

## Klasse `EmbeddingModel`

```python
class EmbeddingModel:
```

Die Klasse kapselt alles, was mit Embeddings zu tun hat.

```python
"""
Kapselt das Embedding-Modell.
...
"""
```

Der Docstring beschreibt, dass dasselbe Modell fuer Chunks und Anfragen verwendet wird.

## Konstruktor

```python
def __init__(self, model_name: str | None = None) -> None:
```

Der Konstruktor wird ausgefuehrt, wenn `EmbeddingModel()` erstellt wird.

`model_name` ist optional. Wenn kein Name uebergeben wird, wird der Wert aus `.env` verwendet.

```python
load_dotenv()
```

Die `.env`-Datei wird geladen.

```python
self.model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
```

Das Embedding-Modell wird bestimmt.

Die Reihenfolge ist:

1. uebergebener `model_name`
2. Wert aus `OPENAI_EMBEDDING_MODEL`
3. Standard `text-embedding-3-small`

```python
self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

Der OpenAI-Client wird mit dem API-Key erstellt.

## Methode `embed_texts`

```python
def embed_texts(self, texts: list[str]) -> list[list[float]]:
```

Diese Methode erstellt Embeddings fuer mehrere Texte.

```python
if not texts:
```

Es wird geprueft, ob die Eingabeliste leer ist.

```python
return []
```

Bei leerer Eingabe wird eine leere Liste zurueckgegeben. Es wird kein API-Aufruf gemacht.

```python
response = self.client.embeddings.create(
```

Der Embedding-API-Aufruf beginnt.

```python
model=self.model_name,
```

Das vorher festgelegte Embedding-Modell wird verwendet.

```python
input=texts,
```

Die Liste der Texte wird als Eingabe uebergeben.

```python
return [item.embedding for item in response.data]
```

Aus der API-Antwort werden alle Embedding-Vektoren extrahiert.

## Methode `embed_query`

```python
def embed_query(self, query: str) -> list[float]:
```

Diese Methode erstellt ein Embedding fuer eine einzelne Nutzerfrage.

```python
return self.embed_texts([query])[0]
```

Die einzelne Frage wird als Liste mit einem Element an `embed_texts` gegeben. Danach wird das erste und einzige Embedding zurueckgegeben.
