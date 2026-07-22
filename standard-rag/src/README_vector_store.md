# README zu `vector_store.py`

Dieses Modul implementiert eine einfache lokale Vektordatenbank.

Die Vektordatenbank speichert:

- Text-Chunks
- Embedding-Vektoren der Chunks

Danach kann sie mit Cosine Similarity die relevantesten Chunks zu einer Nutzerfrage finden.

## Imports

```python
import json
```

`json` wird verwendet, um die Text-Chunks lesbar in `chunks.json` zu speichern.

```python
from pathlib import Path
```

`Path` wird verwendet, um Datei- und Ordnerpfade sauber zu verwalten.

```python
import numpy as np
```

`numpy` wird fuer numerische Vektoroperationen verwendet.

## Klasse `LocalVectorStore`

```python
class LocalVectorStore:
```

Diese Klasse ist die lokale Vektordatenbank.

```python
"""
Eine einfache lokale Vektordatenbank.
...
"""
```

Der Docstring erklaert, dass Chunks als JSON und Embeddings als NumPy-Datei gespeichert werden.

## Konstruktor

```python
def __init__(self, storage_dir: str = "vector_db") -> None:
```

Der Konstruktor bekommt den Speicherordner der Vektordatenbank.

Wenn kein Ordner angegeben wird, wird `vector_db` verwendet.

```python
self.storage_dir = Path(storage_dir)
```

Der Speicherordner wird als `Path`-Objekt gespeichert.

```python
self.chunks_path = self.storage_dir / "chunks.json"
```

Der Pfad fuer die Chunk-Datei wird erstellt.

```python
self.embeddings_path = self.storage_dir / "embeddings.npy"
```

Der Pfad fuer die Embedding-Datei wird erstellt.

```python
self.chunks: list[str] = []
```

Im Arbeitsspeicher wird eine leere Liste fuer Chunks vorbereitet.

```python
self.embeddings: np.ndarray | None = None
```

Die Embeddings werden spaeter als NumPy-Array gespeichert. Am Anfang gibt es noch keine Embeddings, deshalb steht hier `None`.

## Methode `add`

```python
def add(self, chunks: list[str], embeddings: list[list[float]]) -> None:
```

Diese Methode nimmt Chunks und Embeddings entgegen und speichert sie im Objekt.

```python
if len(chunks) != len(embeddings):
```

Es wird geprueft, ob jeder Chunk genau ein Embedding hat.

```python
raise ValueError("Die Anzahl der Chunks und Embeddings muss gleich sein.")
```

Wenn die Anzahl nicht passt, wird ein Fehler ausgelöst.

```python
self.chunks = chunks
```

Die Text-Chunks werden gespeichert.

```python
self.embeddings = np.array(embeddings, dtype=np.float32)
```

Die Embeddings werden in ein NumPy-Array umgewandelt.

`float32` spart Speicher und ist fuer Embeddings ueblich.

## Methode `save`

```python
def save(self) -> None:
```

Diese Methode schreibt die Vektordatenbank auf die Festplatte.

```python
if self.embeddings is None:
```

Es wird geprueft, ob ueberhaupt Embeddings vorhanden sind.

```python
raise ValueError("Es wurden noch keine Embeddings gespeichert.")
```

Ohne Embeddings kann nichts gespeichert werden.

```python
self.storage_dir.mkdir(parents=True, exist_ok=True)
```

Der Speicherordner wird erstellt.

`parents=True` erstellt auch fehlende Oberordner.

`exist_ok=True` verhindert einen Fehler, wenn der Ordner schon existiert.

```python
self.chunks_path.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
```

Die Chunks werden als JSON gespeichert.

`ensure_ascii=False` sorgt dafuer, dass Umlaute lesbar bleiben.

`indent=2` macht die JSON-Datei uebersichtlich.

```python
np.save(self.embeddings_path, self.embeddings)
```

Die Embeddings werden als `.npy`-Datei gespeichert.

## Methode `load`

```python
def load(self) -> None:
```

Diese Methode laedt eine gespeicherte Vektordatenbank.

```python
if not self.chunks_path.exists() or not self.embeddings_path.exists():
```

Es wird geprueft, ob beide Dateien vorhanden sind.

```python
raise FileNotFoundError("Keine gespeicherte Vektordatenbank gefunden.")
```

Wenn eine Datei fehlt, wird ein Fehler gemeldet.

```python
self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
```

Die gespeicherten Chunks werden aus der JSON-Datei geladen.

```python
self.embeddings = np.load(self.embeddings_path)
```

Die gespeicherten Embeddings werden aus der NumPy-Datei geladen.

## Methode `similarity_search`

```python
def similarity_search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
```

Diese Methode sucht die relevantesten Chunks zu einer Nutzerfrage.

`query_embedding` ist der Vektor der Nutzerfrage.

`top_k` ist die Anzahl der gewuenschten Treffer.

```python
if self.embeddings is None or not self.chunks:
```

Es wird geprueft, ob die Datenbank leer ist.

```python
raise ValueError("Die Vektordatenbank ist leer.")
```

Wenn keine Daten vorhanden sind, wird ein Fehler ausgelöst.

```python
query_vector = np.array(query_embedding, dtype=np.float32)
```

Das Query-Embedding wird in ein NumPy-Array umgewandelt.

```python
embedding_norms = np.linalg.norm(self.embeddings, axis=1)
```

Fuer jedes gespeicherte Embedding wird die Vektorlaenge berechnet.

```python
query_norm = np.linalg.norm(query_vector)
```

Die Vektorlaenge der Nutzerfrage wird berechnet.

```python
similarities = (self.embeddings @ query_vector) / (embedding_norms * query_norm)
```

Hier wird Cosine Similarity berechnet.

`self.embeddings @ query_vector` ist das Skalarprodukt.

`embedding_norms * query_norm` ist das Produkt der Vektorlaengen.

Je groesser der Wert, desto aehnlicher sind Chunk und Frage.

```python
top_indices = np.argsort(similarities)[::-1][:top_k]
```

Die Chunks werden nach Aehnlichkeit sortiert.

`[::-1]` dreht die Reihenfolge, sodass die besten Treffer zuerst kommen.

`[:top_k]` nimmt nur die gewuenschte Anzahl.

```python
return [self.chunks[index] for index in top_indices]
```

Die passenden Text-Chunks werden zurueckgegeben.
