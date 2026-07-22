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

## Aktueller Stand

Die aktuelle Konfiguration verarbeitet das PDF `data/Elektrotechnik 3.pdf`.
Standardmaessig werden die ersten 20 Seiten genutzt (`MAX_PAGES = 20`).
Dabei wird nicht nur der direkt auslesbare PDF-Text geladen, sondern jede Seite
zusaetzlich visuell analysiert, damit Bilder, Tabellen, Diagramme, Formeln und
Beschriftungen ebenfalls als Text in den Knowledge Graph einfliessen.

Der aktuell verwendete lokale Graph liegt in:

```text
graph_db_elektrotechnik_3_first_20_full_visuals/knowledge_graph.json
```

Die Streamlit-App laedt diesen gespeicherten Graph zuerst. Nur wenn die Datei
nicht vorhanden ist, wird der Graph aus dem PDF neu aufgebaut.

## Pipeline

1. PDF laden und auf die ersten 20 Seiten begrenzen
2. Direkt auslesbaren PDF-Text extrahieren
3. Seiten als Bilder analysieren und visuelle Inhalte als Text beschreiben
4. Text in ueberlappende Chunks aufteilen
5. Entitaeten und Beziehungen aus jedem Chunk extrahieren
6. Lokalen Knowledge Graph als JSON speichern
7. Nutzerfrage entgegennehmen
8. Relevante Entitaeten in der Frage erkennen
9. Graph Traversal ueber 1 bis 2 Hops durchfuehren
10. Graph-Kontext an das Sprachmodell geben
11. Antwort generieren

## Projektstruktur

```text
graph-rag-basic/
  data/
    Elektrotechnik 3.pdf
  graph_db/
    knowledge_graph.json
  graph_db_elektrotechnik_3_first_20_full_visuals/
    knowledge_graph.json
  graph_db_elektrotechnik_3_full_visuals/
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

Durch `.streamlit/config.toml` startet die App lokal unter:

```text
http://127.0.0.1:8503
```

## Abgrenzung zum Standard-RAG

Dieses Graph-RAG-Projekt verwendet keine lokale Vektordatenbank fuer die Auswahl der relevantesten Chunks. Die Antwort basiert auf dem Teilgraphen, der durch passende Entitaeten und deren Beziehungen gefunden wird. Dadurch ist der Retrieval-Prozess einfacher erklaerbar, weil sichtbar bleibt, welche Entitaeten und Kanten zur Antwort gefuehrt haben.
