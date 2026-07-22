# Basic Graph-RAG

Dieses Projekt ist ein bewusst einfach gehaltenes Graph-RAG-System auf Basis der Struktur deines Standard-RAG-Projekts.

Der Standard-RAG bleibt unveraendert. Dieses Projekt nutzt dieselbe Grundidee mit separaten Modulen fuer Laden, Chunking, Retrieval, Generation und UI. Der zentrale Unterschied liegt im Retrieval:

- Standard-RAG sucht relevante Text-Chunks ueber Embedding-Aehnlichkeit.
- Graph-RAG strukturiert die Inhalte zuerst als Knowledge Graph.
- Knoten sind Entitaeten, zum Beispiel Personen, Konzepte, Bauteile oder Regeln.
- Kanten beschreiben Beziehungen zwischen diesen Entitaeten.
- Bei einer Anfrage werden relevante Entitaeten identifiziert und der Graph wird ueber verbundene Beziehungen durchsucht.
- Der gefundene Graph-Kontext wird zusammen mit der Frage an das Sprachmodell uebergeben.

Damit folgt das Projekt der normalen Graph-RAG-Definition fuer eine Bachelorarbeit: vorhandene Daten werden in einem Wissensgraphen strukturiert, relevante Entitaeten werden zur Anfrage gesucht, zusammenhaengende Informationen werden ueber Graph-Verknuepfungen erschlossen und anschliessend fuer die Antwortgenerierung genutzt.

## Pipeline

1. PDF laden
2. Text in Chunks aufteilen
3. Entitaeten und Beziehungen aus jedem Chunk extrahieren
4. Lokalen Knowledge Graph als JSON speichern
5. Nutzerfrage entgegennehmen
6. Relevante Entitaeten in der Frage erkennen
7. Graph Traversal ueber 1 bis 2 Hops durchfuehren
8. Graph-Kontext an das Sprachmodell geben
9. Antwort generieren

## Projektstruktur

```text
graph-rag-basic/
  data/
    Elektrotechnik 3.pdf
  graph_db/
    knowledge_graph.json
  src/
    chunking.py
    document_loader.py
    generation.py
    graph_store.py
    main.py
    retrieval.py
    ui.py
  .env.example
  requirements.txt
```

## Installation

```bash
cd graph-rag-basic
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Lege danach eine `.env` Datei an:

```env
OPENAI_API_KEY=dein_api_key_hier
OPENAI_LLM_MODEL=gpt-4.1-mini
OPENAI_VISION_MODEL=gpt-4.1-mini
```

## Start ueber Konsole

```bash
python src/main.py
```

## Start mit Streamlit

```bash
streamlit run src/ui.py
```

## Abgrenzung zum Standard-RAG

Dieses Graph-RAG-Projekt verwendet keine lokale Vektordatenbank fuer die Auswahl der relevantesten Chunks. Die Antwort basiert auf dem Teilgraphen, der durch passende Entitaeten und deren Beziehungen gefunden wird. Dadurch ist der Retrieval-Prozess einfacher erklaerbar, weil sichtbar bleibt, welche Entitaeten und Kanten zur Antwort gefuehrt haben.
