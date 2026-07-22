# Standard-RAG-System

Dieses Projekt implementiert ein einfaches Standard-RAG-System in Python. Die fachliche Kernpipeline folgt der klassischen Architektur:

**Retrieval -> Augmentation -> Generation**

Die Pipeline besteht aus folgenden Schritten:

1. Ein PDF-Dokument inklusive Text, Tabellen, Bildern, Diagrammen und Formeln laden
2. Die extrahierten Inhalte in kleinere Textabschnitte aufteilen
3. Chunks mit einem Embedding-Modell in Vektoren umwandeln
4. Chunks und Embeddings lokal speichern
5. Nutzeranfrage entgegennehmen
6. Nutzeranfrage mit demselben Embedding-Modell vektorisieren
7. Relevanteste Chunks ueber semantische Aehnlichkeit abrufen
8. Prompt aus Nutzeranfrage und relevanten Chunks erstellen
9. Antwort mit einem Large Language Model generieren
10. Antwort ausgeben

Die Kernpipeline ist bewusst einfach gehalten. Sie nutzt keine zusaetzlichen RAG-Verfahren wie Re-Ranking, Query Expansion, Agenten, Memory, Websuche oder hybride Suche. Fuer Experimente und Auswertung gibt es aber ein separates Evaluationsmodul unter `src/evaluation/`.

## Projektstruktur

```text
standard-rag/
├── data/
│   ├── Elektrotechnik 3.pdf
│   └── documents/
├── docs/
│   └── standard_rag_class_diagram.md
├── graph_db_elektrotechnik_3/
│   └── knowledge_graph.json
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── main.py
│   ├── ui.py
│   ├── README_*.md
│   └── evaluation/
│       ├── config.py
│       ├── evaluation.py
│       ├── loader_test.py
│       ├── chunk_test.py
│       ├── retrieval_test.py
│       ├── generation_test.py
│       ├── metrics.py
│       ├── test_questions.json
│       └── results/
├── vector_db*/
│   ├── chunks.json
│   └── embeddings.npy
├── .env.example
├── requirements.txt
└── README.md
```

## Voraussetzungen

- Python 3.10 oder neuer
- OpenAI API-Key

## Installation

Wechsle in den Projektordner:

```bash
cd standard-rag
```

Erstelle und aktiviere optional eine virtuelle Umgebung:

```bash
python -m venv .venv
source .venv/bin/activate
```

Installiere die Abhaengigkeiten:

```bash
pip install -r requirements.txt
```

## Konfiguration

Kopiere die Beispieldatei:

```bash
cp .env.example .env
```

Trage deinen API-Key in `.env` ein:

```env
OPENAI_API_KEY=dein_api_key_hier
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4.1-mini
OPENAI_VISION_MODEL=gpt-4.1-mini
```

## PDF-Dokument

Die enthaltene Beispiel-PDF liegt aktuell unter:

```text
data/Elektrotechnik 3.pdf
```

Du kannst beim Start des Programms den Pfad zu einer eigenen PDF-Datei eingeben. Fuer die enthaltene Datei gibst du diesen Pfad ein:

```text
data/Elektrotechnik 3.pdf
```

## Start ueber die Konsole

Fuehre das System aus:

```bash
python src/main.py
```

Danach gibst du zuerst den Pfad zur PDF-Datei und danach deine Frage ein. Das System:

- laedt das PDF-Dokument,
- beschreibt sichtbare Tabellen, Bilder, Diagramme und Formeln als Text,
- erstellt Chunks,
- erzeugt Embeddings,
- speichert Chunks und Embeddings lokal,
- ruft passende Chunks zur Frage ab,
- erstellt daraus einen Prompt,
- generiert eine Antwort mit dem LLM.

## Start mit Web-UI

Die einfache Benutzeroberflaeche startest du mit:

```bash
streamlit run src/ui.py
```

Die UI arbeitet mit der lokalen Wissensbasis `vector_db_elektrotechnik_3_full_visuals`. Wenn diese Vektordatenbank bereits vorhanden ist, wird sie direkt geladen. Falls sie noch nicht vorhanden ist, baut die UI sie beim ersten Start neu auf. Dabei muss der in `src/ui.py` gesetzte PDF-Pfad zur vorhandenen PDF-Datei passen. Beim Neuaufbau wird jede PDF-Seite als Text extrahiert und zusaetzlich visuell gelesen, damit Bilder, Tabellen, Formeln und Diagramme beruecksichtigt werden.

Danach kannst du:

- direkt Fragen stellen,
- Antworten generieren,
- die verwendeten Chunks anzeigen.

## Lokale Wissensbasen

Im Projekt liegen mehrere vorbereitete lokale Vector Stores. Jede Variante besteht aus:

- `chunks.json`: die gespeicherten Textabschnitte
- `embeddings.npy`: die dazugehoerigen Embedding-Vektoren

Beispiele fuer vorhandene Varianten:

- `vector_db_elektrotechnik_3_full_visuals`: Wissensbasis mit vollstaendiger visueller Analyse
- `vector_db_elektrotechnik_3_targeted_visuals`: Wissensbasis mit gezielter visueller Analyse bestimmter Seiten
- `vector_db_elektrotechnik_3_text`: Wissensbasis nur aus direkt auslesbarem PDF-Text
- `vector_db_elektrotechnik_3_text_chunks_250`: Textvariante mit Chunk-Groesse 250
- `vector_db_elektrotechnik_3_graph_rag`: vorbereitete Daten fuer den Vergleich mit einer Graph-RAG-Variante

Der Ordner `graph_db_elektrotechnik_3/` enthaelt einen Knowledge-Graph-Export. Er gehoert nicht zur Standard-RAG-Kernpipeline, kann aber fuer Architekturvergleiche oder Graph-RAG-Auswertungen relevant sein.

## Evaluation

Das Verzeichnis `src/evaluation/` enthaelt ein separates Evaluationsmodul. Es dient dazu, verschiedene Parameter und Varianten des RAG-Systems vergleichbar zu testen.

Die zentrale Startdatei ist:

```bash
python src/evaluation/evaluation.py
```

Das Menue bietet folgende Tests:

1. Dokumentloader testen
2. Chunk Size testen
3. Overlap testen
4. Top-k testen
5. Temperature testen

Die Standard- und Testparameter stehen in:

```text
src/evaluation/config.py
```

Wichtige Parameter sind:

- `DOCUMENT_LOADER`: verwendeter Loader, z. B. `text`, `targeted_visual` oder `full_visual`
- `CHUNK_SIZE`: Anzahl der Woerter pro Chunk
- `OVERLAP`: Ueberlappung zwischen Chunks
- `TOP_K`: Anzahl der abgerufenen Chunks
- `TEMPERATURE`: Temperatur fuer die Antwortgenerierung

Die Ergebnisdateien werden als CSV unter `src/evaluation/results/` gespeichert.

## Beschreibung der Module

### `document_loader.py`

Laedt genau eine PDF-Datei und gibt Text plus beschriebene Tabellen-, Bild-, Diagramm- und Formelinhalte als String zurueck. Es gibt drei Loader-Varianten:

- `load_pdf_text()`: nur direkt auslesbarer PDF-Text
- `load_pdf_text_with_targeted_visuals()`: Text plus visuelle Analyse ausgewaehlter Seiten
- `load_pdf_with_visuals()`: Text plus visuelle Analyse aller Seiten

### `chunking.py`

Teilt die extrahierten PDF-Inhalte wortbasiert in einfache Chunks auf. Die Chunk-Groesse und der Overlap koennen angepasst werden.

### `embeddings.py`

Erstellt Embeddings fuer Dokument-Chunks und Nutzeranfragen. Beide verwenden dasselbe Embedding-Modell.

### `vector_store.py`

Speichert Chunks und Embeddings lokal. Die semantische Suche erfolgt ueber Cosine Similarity.

### `retrieval.py`

Vektorisiert die Nutzeranfrage und ruft die relevantesten Chunks aus der lokalen Vektordatenbank ab.

### `generation.py`

Erstellt den Prompt aus Nutzerfrage und relevanten Chunks und uebergibt ihn an ein Large Language Model.

### `main.py`

Fuehrt die gesamte Standard-RAG-Pipeline in der vorgesehenen Reihenfolge als Konsolenanwendung aus.

### `ui.py`

Stellt eine einfache Streamlit-Oberflaeche bereit, damit das RAG-System im Browser genutzt werden kann.
