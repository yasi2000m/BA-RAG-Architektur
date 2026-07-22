# README zu `ui.py`

`ui.py` ist die Streamlit-Benutzeroberflaeche fuer das RAG-System.

Die UI nutzt automatisch die integrierte PDF:

```text
data/elektrotechnik_3.pdf
```

Sie baut oder laedt eine lokale Vektordatenbank und erlaubt danach direkte Fragen im Browser.

## Imports

```python
import streamlit as st
```

Importiert Streamlit fuer die Web-Oberflaeche.

```python
from chunking import chunk_text
```

Importiert das Chunking.

```python
from document_loader import load_pdf_text_with_targeted_visuals
```

Importiert den PDF-Loader, der normalen Text schnell laedt und bestimmte Seiten gezielt visuell analysiert.

```python
from embeddings import EmbeddingModel
```

Importiert das Embedding-Modell.

```python
from generation import AnswerGenerator
```

Importiert die Antwortgenerierung.

```python
from retrieval import retrieve_relevant_chunks
```

Importiert den Retrieval-Schritt.

```python
from vector_store import LocalVectorStore
```

Importiert die lokale Vektordatenbank.

## Konstanten

```python
DEFAULT_PDF_PATH = "data/elektrotechnik_3.pdf"
```

Legt die fest integrierte PDF-Datei fest.

```python
VECTOR_DB_PATH = "vector_db_elektrotechnik_3_targeted_visuals"
```

Legt den Speicherordner fuer die Vektordatenbank fest.

## Funktion `build_vector_store`

```python
def build_vector_store(pdf_path: str) -> tuple[LocalVectorStore, EmbeddingModel, int]:
```

Diese Funktion baut die Wissensbasis neu auf.

Sie gibt drei Dinge zurueck:

- Vektordatenbank
- Embedding-Modell
- Anzahl der Chunks

```python
document_text = load_pdf_text_with_targeted_visuals(
```

Die PDF wird geladen.

```python
pdf_path,
```

Der Pfad zur PDF wird uebergeben.

```python
visual_keywords=["Sicherheitsregeln"],
```

Seiten mit dem Stichwort `Sicherheitsregeln` werden zusaetzlich visuell analysiert. Das hilft bei Tabellen, die nicht sauber als normaler PDF-Text extrahiert werden.

```python
if not document_text:
```

Es wird geprueft, ob Inhalt gefunden wurde.

```python
raise ValueError("Im PDF wurden keine auswertbaren Inhalte gefunden.")
```

Wenn nichts gefunden wurde, wird ein Fehler ausgelöst.

```python
chunks = chunk_text(document_text, chunk_size=250, overlap=50)
```

Der Text wird in kleinere Chunks zerlegt.

```python
embedding_model = EmbeddingModel()
```

Das Embedding-Modell wird erstellt.

```python
chunk_embeddings = embedding_model.embed_texts(chunks)
```

Alle Chunks werden vektorisiert.

```python
vector_store = LocalVectorStore(VECTOR_DB_PATH)
```

Die lokale Vektordatenbank wird erstellt.

```python
vector_store.add(chunks, chunk_embeddings)
```

Chunks und Embeddings werden in die Datenbank gelegt.

```python
vector_store.save()
```

Die Datenbank wird gespeichert.

```python
return vector_store, embedding_model, len(chunks)
```

Die fertige Datenbank, das Embedding-Modell und die Chunk-Anzahl werden zurueckgegeben.

## Funktion `load_or_build_vector_store`

```python
def load_or_build_vector_store() -> tuple[LocalVectorStore, EmbeddingModel, int, bool]:
```

Diese Funktion laedt eine vorhandene Wissensbasis oder baut sie neu auf.

Der vierte Rueckgabewert ist ein Boolean:

- `False`: Datenbank wurde geladen
- `True`: Datenbank wurde neu gebaut

```python
embedding_model = EmbeddingModel()
```

Das Embedding-Modell wird erstellt.

```python
vector_store = LocalVectorStore(VECTOR_DB_PATH)
```

Die Vektordatenbank wird mit dem festen Speicherpfad vorbereitet.

```python
try:
```

Es wird versucht, eine vorhandene Datenbank zu laden.

```python
vector_store.load()
```

Die gespeicherte Datenbank wird geladen.

```python
return vector_store, embedding_model, len(vector_store.chunks), False
```

Bei Erfolg wird die geladene Datenbank zurueckgegeben.

```python
except FileNotFoundError:
```

Wenn keine gespeicherte Datenbank existiert, wird dieser Block ausgefuehrt.

```python
vector_store, embedding_model, chunk_count = build_vector_store(DEFAULT_PDF_PATH)
```

Die Wissensbasis wird aus der Standard-PDF neu aufgebaut.

```python
return vector_store, embedding_model, chunk_count, True
```

Die neu erstellte Datenbank wird zurueckgegeben.

## Streamlit-Seite

```python
st.set_page_config(page_title="Standard-RAG", page_icon="PDF", layout="centered")
```

Legt Titel, Symbol und Layout der Streamlit-Seite fest.

```python
st.title("Standard-RAG fuer Elektrotechnik")
```

Zeigt den Seitentitel.

```python
st.write("Die Elektrotechnik-PDF ist integriert. Relevante Tabellenseiten werden gezielt visuell gelesen.")
```

Zeigt eine kurze Beschreibung der UI.

## Session-State laden

```python
if "vector_store" not in st.session_state or st.session_state.get("vector_db_path") != VECTOR_DB_PATH:
```

Prueft, ob die Wissensbasis bereits in der aktuellen UI-Sitzung geladen ist.

Außerdem wird geprueft, ob der gespeicherte Datenbankpfad noch zum aktuellen Code passt.

```python
with st.spinner("Wissensbasis wird geladen ..."):
```

Zeigt einen Ladehinweis in der UI.

```python
try:
```

Fehler beim Laden werden abgefangen.

```python
vector_store, embedding_model, chunk_count, was_built = load_or_build_vector_store()
```

Die Wissensbasis wird geladen oder neu aufgebaut.

```python
st.session_state.vector_store = vector_store
```

Die Vektordatenbank wird in der Streamlit-Sitzung gespeichert.

```python
st.session_state.embedding_model = embedding_model
```

Das Embedding-Modell wird gespeichert.

```python
st.session_state.chunk_count = chunk_count
```

Die Anzahl der Chunks wird gespeichert.

```python
st.session_state.was_built = was_built
```

Es wird gespeichert, ob die Datenbank neu gebaut wurde.

```python
st.session_state.vector_db_path = VECTOR_DB_PATH
```

Der verwendete Datenbankpfad wird gespeichert.

```python
except Exception as error:
```

Falls ein Fehler passiert, wird er abgefangen.

```python
st.session_state.vector_store = None
st.session_state.embedding_model = None
```

Bei Fehlern werden Datenbank und Embedding-Modell zurueckgesetzt.

```python
st.error(f"Fehler beim Laden der Wissensbasis: {error}")
```

Der Fehler wird in der UI angezeigt.

## Statusanzeige

```python
if st.session_state.vector_store is not None:
```

Prueft, ob die Wissensbasis verfuegbar ist.

```python
st.success(f"Wissensbasis bereit. Chunks: {st.session_state.chunk_count}")
```

Zeigt eine Erfolgsmeldung mit Chunk-Anzahl.

```python
st.caption(f"PDF: {DEFAULT_PDF_PATH} | Vektordatenbank: {VECTOR_DB_PATH}")
```

Zeigt, welche PDF und welche Datenbank verwendet werden.

```python
st.divider()
```

Fuegt eine Trennlinie ein.

## Frage und Antwort

```python
question = st.text_input("Deine Frage")
```

Erstellt ein Eingabefeld fuer die Nutzerfrage.

```python
if st.button("Antwort generieren", disabled=not question):
```

Erstellt den Antwort-Button. Er ist deaktiviert, solange keine Frage eingegeben wurde.

```python
if st.session_state.vector_store is None or st.session_state.embedding_model is None:
```

Prueft, ob die Wissensbasis geladen ist.

```python
st.warning("Die Wissensbasis konnte nicht geladen werden.")
```

Falls nicht, erscheint eine Warnung.

```python
with st.spinner("Relevante Chunks werden gesucht und die Antwort wird generiert ..."):
```

Zeigt einen Ladehinweis waehrend Retrieval und Generation.

```python
relevant_chunks = retrieve_relevant_chunks(
```

Der Retrieval-Schritt beginnt.

```python
query=question,
```

Die Nutzerfrage wird uebergeben.

```python
embedding_model=st.session_state.embedding_model,
```

Das gespeicherte Embedding-Modell wird verwendet.

```python
vector_store=st.session_state.vector_store,
```

Die gespeicherte Vektordatenbank wird verwendet.

```python
top_k=5,
```

Die fuenf relevantesten Chunks werden abgerufen.

```python
generator = AnswerGenerator()
```

Der Antwortgenerator wird erstellt.

```python
answer = generator.generate_answer(question, relevant_chunks)
```

Die Antwort wird mit dem LLM generiert.

```python
st.subheader("Antwort")
```

Zeigt eine Ueberschrift.

```python
st.write(answer)
```

Zeigt die Antwort in der UI.

```python
with st.expander("Verwendete Chunks anzeigen"):
```

Erstellt einen ausklappbaren Bereich fuer Transparenz.

```python
for index, chunk in enumerate(relevant_chunks, start=1):
```

Alle verwendeten Chunks werden durchlaufen.

```python
st.markdown(f"**Chunk {index}**")
```

Zeigt die Chunk-Nummer fett an.

```python
st.write(chunk)
```

Zeigt den Inhalt des Chunks.
