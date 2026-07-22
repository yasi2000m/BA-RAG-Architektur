# Graph-RAG-System

Dieses Projekt implementiert ein einfaches Graph-RAG-System in Python. Es folgt der Architektur aus deiner Abbildung:

**Retrieval -> Augmentation -> Generation**

Dabei werden keine Textauszuege als Kontext an das Sprachmodell uebergeben. Das Dokument wird in einen lokalen Knowledge Graph ueberfuehrt. Die Nutzerfrage ruft anschliessend spezifisches Kontextwissen aus diesem Graphen ab.

## Pipeline

1. Ein PDF-Dokument inklusive Text, Tabellen und Bildern laden
2. Lange PDF-Inhalte nur intern in Textabschnitte fuer die Graph-Extraktion aufteilen
3. Aus den Textabschnitten Entitaeten und Beziehungen extrahieren
4. Entitaeten, Beziehungen und Quellenzuordnungen lokal als Knowledge Graph speichern
5. Nutzeranfrage entgegennehmen
6. Spezifisches Kontextwissen aus dem Graph abrufen
7. Prompt aus Nutzerfrage und Knowledge-Graph-Kontext erstellen
8. Antwort mit einem Large Language Model generieren
9. Antwort ausgeben

Die Umsetzung orientiert sich an der Graph-RAG-Idee aus IBM Think, bei der RAG durch Knowledge Graphs erweitert wird, um Entitaeten und Beziehungen explizit als Kontext nutzbar zu machen:

https://www.ibm.com/think/tutorials/knowledge-graph-rag

Es werden bewusst keine externen Graphdatenbanken wie Neo4j oder Memgraph verwendet. Der Graph wird lokal als JSON gespeichert, damit die Architektur fuer die Bachelorarbeit einfach nachvollziehbar und direkt mit Standard-RAG vergleichbar bleibt.

## Projektstruktur

```text
graph-rag/
├── data/
│   └── elektrotechnik_3.pdf
├── src/
│   ├── document_loader.py
│   ├── text_segmentation.py
│   ├── graph_store.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── ui.py
│   └── main.py
├── .env.example
├── requirements.txt
└── README.md
```

## Voraussetzungen

- Python 3.10 oder neuer
- OpenAI API-Key

## Installation

```bash
cd graph-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Konfiguration

```env
OPENAI_API_KEY=dein_api_key_hier
OPENAI_LLM_MODEL=gpt-4.1-mini
OPENAI_VISION_MODEL=gpt-4.1-mini
```

## Start

```bash
python src/main.py
```

Das System:

- laedt das PDF-Dokument,
- beschreibt sichtbare Tabellen, Bilder und Diagramme als Text,
- extrahiert Entitaeten und Beziehungen,
- speichert den Knowledge Graph lokal,
- ruft passende Graph-Beziehungen zur Frage ab,
- erstellt daraus einen Graph-RAG-Prompt,
- generiert eine Antwort mit dem LLM.

## Start mit Web-UI

```bash
streamlit run src/ui.py
```

Wenn `graph_db_elektrotechnik_3` bereits vorhanden ist, wird die Wissensbasis direkt geladen. Falls sie noch nicht vorhanden ist, baut die UI sie beim ersten Start automatisch auf.

Danach kannst du:

- direkt Fragen stellen,
- Antworten generieren,
- den verwendeten Graph-Kontext anzeigen.

## Beschreibung der Module

### `document_loader.py`

Laedt eine PDF-Datei und gibt Text plus beschriebene Tabellen-, Bild- und Diagramminhalte als String zurueck.

### `text_segmentation.py`

Teilt lange Dokumenttexte nur intern in Textabschnitte auf, damit daraus Entitaeten und Beziehungen extrahiert werden koennen. Diese Textabschnitte werden nicht als Kontext an das LLM uebergeben.

### `graph_store.py`

Extrahiert Entitaeten und Beziehungen und speichert sie als lokalen Knowledge Graph. Der Graph liefert beim Retrieval spezifisches Kontextwissen ueber Konzepte und deren Beziehungen.

### `retrieval.py`

Ruft zur Nutzerfrage passende Entitaeten und Beziehungen aus dem Knowledge Graph ab.

### `generation.py`

Erstellt den Prompt aus Nutzerfrage und Knowledge-Graph-Kontext und uebergibt ihn an ein Large Language Model.

### `main.py`

Fuehrt die gesamte Graph-RAG-Pipeline in der vorgesehenen Reihenfolge aus.

### `ui.py`

Stellt eine einfache Streamlit-Oberflaeche bereit, damit das Graph-RAG-System im Browser genutzt werden kann.
