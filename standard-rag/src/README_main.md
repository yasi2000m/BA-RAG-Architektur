# README zu `main.py`

`main.py` ist die Konsolen-Version des RAG-Systems.

Die Datei fuehrt die gesamte Pipeline nacheinander aus:

```text
PDF laden -> Chunking -> Embeddings -> Vektordatenbank -> Frage -> Retrieval -> Prompt -> Antwort
```

## Imports

```python
from chunking import chunk_text
```

Importiert die Funktion, die Text in Chunks aufteilt.

```python
from document_loader import load_pdf
```

Importiert den Standard-PDF-Loader. Dieser laedt schnellen direkt auslesbaren PDF-Text.

```python
from embeddings import EmbeddingModel
```

Importiert die Klasse fuer Embeddings.

```python
from generation import AnswerGenerator
```

Importiert die Klasse fuer Prompt-Erstellung und LLM-Antwort.

```python
from retrieval import retrieve_relevant_chunks
```

Importiert die Retrieval-Funktion.

```python
from vector_store import LocalVectorStore
```

Importiert die lokale Vektordatenbank.

## Funktion `main`

```python
def main() -> None:
```

Definiert die Hauptfunktion. Sie gibt keinen Wert zurueck.

```python
"""
Fuehrt die komplette Standard-RAG-Pipeline aus:
...
"""
```

Der Docstring listet die zehn Pipeline-Schritte auf.

```python
pdf_path = input("Pfad zur PDF-Datei [data/elektrotechnik_3.pdf]: ").strip()
```

Das Programm fragt im Terminal nach einem PDF-Pfad.

`strip()` entfernt Leerzeichen am Anfang und Ende.

```python
pdf_path = pdf_path or "data/elektrotechnik_3.pdf"
```

Wenn der Nutzer nichts eingibt, wird automatisch die integrierte PDF verwendet.

```python
document_text = load_pdf(pdf_path)
```

Das PDF wird geladen und als Text zurueckgegeben.

```python
if not document_text:
```

Es wird geprueft, ob Text gefunden wurde.

```python
print(f"Keine Inhalte im PDF gefunden. Pruefe die Datei: {pdf_path}")
```

Wenn kein Text gefunden wurde, erscheint eine Fehlermeldung.

```python
return
```

Die Pipeline wird beendet.

```python
chunks = chunk_text(document_text, chunk_size=250, overlap=50)
```

Der PDF-Text wird in kleinere Chunks zerlegt.

`250` Woerter pro Chunk helfen, kurze Tabelleninhalte besser auffindbar zu machen.

`50` Woerter Overlap erhalten Kontext zwischen Chunks.

```python
embedding_model = EmbeddingModel()
```

Das Embedding-Modell wird erstellt.

```python
chunk_embeddings = embedding_model.embed_texts(chunks)
```

Alle Chunks werden in Vektoren umgewandelt.

```python
vector_store = LocalVectorStore("vector_db_elektrotechnik_3")
```

Eine lokale Vektordatenbank wird erstellt.

```python
vector_store.add(chunks, chunk_embeddings)
```

Chunks und Embeddings werden in die Vektordatenbank gelegt.

```python
vector_store.save()
```

Die Vektordatenbank wird auf der Festplatte gespeichert.

```python
query = input("Bitte gib deine Frage ein: ")
```

Die Nutzerfrage wird im Terminal abgefragt.

```python
relevant_chunks = retrieve_relevant_chunks(
```

Der Retrieval-Schritt beginnt.

```python
query=query,
```

Die Nutzerfrage wird uebergeben.

```python
embedding_model=embedding_model,
```

Dasselbe Embedding-Modell wird verwendet.

```python
vector_store=vector_store,
```

Die Vektordatenbank wird durchsucht.

```python
top_k=5,
```

Die fuenf relevantesten Chunks werden abgerufen.

```python
generator = AnswerGenerator()
```

Der Antwortgenerator wird erstellt.

```python
answer = generator.generate_answer(query, relevant_chunks)
```

Aus Frage und relevanten Chunks wird eine Antwort generiert.

```python
print("\nAntwort:")
```

Eine Ueberschrift wird ausgegeben.

```python
print(answer)
```

Die Antwort wird ausgegeben.

```python
if __name__ == "__main__":
```

Dieser Block prueft, ob die Datei direkt gestartet wurde.

```python
main()
```

Wenn ja, wird die Pipeline gestartet.
