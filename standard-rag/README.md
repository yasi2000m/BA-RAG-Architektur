# Standard-RAG-System

Dieses Projekt implementiert ein einfaches Standard-RAG-System in Python. Es folgt exakt der klassischen Architektur:

**Retrieval -> Augmentation -> Generation**

Die Pipeline besteht aus folgenden Schritten:

1. Ein PDF-Dokument inklusive Text, Tabellen und Bildern laden
2. Die extrahierten Inhalte in kleinere Textabschnitte aufteilen
3. Chunks mit einem Embedding-Modell in Vektoren umwandeln
4. Chunks und Embeddings lokal speichern
5. Nutzeranfrage entgegennehmen
6. Nutzeranfrage mit demselben Embedding-Modell vektorisieren
7. Relevanteste Chunks ueber semantische Aehnlichkeit abrufen
8. Prompt aus Nutzeranfrage und relevanten Chunks erstellen
9. Antwort mit einem Large Language Model generieren
10. Antwort ausgeben

Es werden bewusst keine zusaetzlichen Verfahren wie Re-Ranking, Query Expansion, Agenten, Memory, Evaluation, Websuche oder hybride Suche eingesetzt.

## Projektstruktur

```text
standard-rag/
├── data/
│   └── elektrotechnik_3.pdf
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── generation.py
│   └── main.py
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

## PDF-Dokument hinzufuegen

Das System arbeitet mit genau einer PDF-Datei als Wissensbasis:

```text
data/elektrotechnik_3.pdf
```

Du kannst beim Start des Programms den Pfad zu deinem eigenen PDF eingeben. Wenn du nichts eingibst, wird die Beispiel-PDF verwendet.

## Start

Fuehre das System aus:

```bash
python src/main.py
```

Danach gibst du zuerst optional den Pfad zu deiner PDF-Datei ein und danach deine Frage. Das System:

- laedt das PDF-Dokument,
- beschreibt sichtbare Tabellen, Bilder und Diagramme als Text,
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

Die UI nutzt automatisch:

```text
data/elektrotechnik_3.pdf
```

Wenn `vector_db_elektrotechnik_3_full_visuals` bereits vorhanden ist, wird die Wissensbasis direkt geladen. Falls sie noch nicht vorhanden ist, baut die UI sie beim ersten Start automatisch auf. Dabei wird jede PDF-Seite als Text extrahiert und zusaetzlich visuell gelesen, damit Bilder, Tabellen, Formeln und Diagramme beruecksichtigt werden.

Danach kannst du:

- direkt Fragen stellen,
- Antworten generieren,
- die verwendeten Chunks anzeigen.

## Beschreibung der Module

### `document_loader.py`

Laedt genau eine PDF-Datei und gibt Text plus beschriebene Tabellen-, Bild- und Diagramminhalte als String zurueck.

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

Fuehrt die gesamte Pipeline in der vorgesehenen Reihenfolge aus.

### `ui.py`

Stellt eine einfache Streamlit-Oberflaeche bereit, damit das RAG-System im Browser genutzt werden kann.
