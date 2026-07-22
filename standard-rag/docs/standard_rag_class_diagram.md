# Klassendiagramm: Standard-RAG-System

Dieses Diagramm zeigt die Struktur des Standard-RAG-Projekts. Einige Dateien enthalten keine Klassen, sondern Funktionen. Diese Module werden im Diagramm als Utility-Komponenten dargestellt, damit die Gesamtarchitektur sichtbar bleibt.

```mermaid
classDiagram
    direction LR

    class Main {
        <<entry point>>
        +main() None
    }

    class DocumentLoader {
        <<module>>
        +describe_page_image(client, model, image_bytes) str
        +load_pdf_text(pdf_path) str
        +load_pdf_text_with_targeted_visuals(pdf_path, visual_keywords) str
        +load_pdf_with_visuals(pdf_path) str
        +load_pdf(pdf_path) str
    }

    class Chunking {
        <<module>>
        +chunk_text(text, chunk_size, overlap) list~str~
    }

    class EmbeddingModel {
        -model_name str
        -client OpenAI
        +__init__(model_name) None
        +embed_texts(texts) list~list~float~~
        +embed_query(query) list~float~
    }

    class LocalVectorStore {
        -storage_dir Path
        -chunks_path Path
        -embeddings_path Path
        -chunks list~str~
        -embeddings ndarray
        +__init__(storage_dir) None
        +add(chunks, embeddings) None
        +save() None
        +load() None
        +similarity_search(query_embedding, top_k) list~str~
    }

    class Retrieval {
        <<module>>
        +retrieve_relevant_chunks(query, embedding_model, vector_store, top_k) list~str~
    }

    class AnswerGenerator {
        -model_name str
        -client OpenAI
        +__init__(model_name) None
        +build_prompt(query, relevant_chunks) str
        +generate_answer(query, relevant_chunks) str
    }

    class StreamlitUI {
        <<optional UI>>
        +build_vector_store(pdf_path) tuple
        +load_or_build_vector_store() tuple
    }

    Main ..> DocumentLoader : laedt PDF
    Main ..> Chunking : erstellt Chunks
    Main ..> EmbeddingModel : erstellt Embeddings
    Main ..> LocalVectorStore : speichert und durchsucht Vektoren
    Main ..> Retrieval : ruft relevante Chunks ab
    Main ..> AnswerGenerator : generiert Antwort

    StreamlitUI ..> DocumentLoader : laedt integrierte PDF
    StreamlitUI ..> Chunking : erstellt Chunks
    StreamlitUI ..> EmbeddingModel : erstellt Embeddings und Query-Vektor
    StreamlitUI ..> LocalVectorStore : laedt/speichert Wissensbasis
    StreamlitUI ..> Retrieval : sucht relevante Chunks
    StreamlitUI ..> AnswerGenerator : erzeugt Antwort

    Retrieval ..> EmbeddingModel : nutzt embed_query()
    Retrieval ..> LocalVectorStore : nutzt similarity_search()

    AnswerGenerator ..> LocalVectorStore : verwendet abgerufene Chunks indirekt
```

## Kurzbeschreibung der Struktur

- `main.py` ist der Einstiegspunkt der Konsolenanwendung.
- `document_loader.py` laedt die PDF-Datei und wandelt Inhalte in Text um.
- `chunking.py` teilt den Dokumenttext in kleinere Chunks.
- `embeddings.py` enthaelt die Klasse `EmbeddingModel`, die Chunks und Nutzerfragen in Vektoren umwandelt.
- `vector_store.py` enthaelt die Klasse `LocalVectorStore`, die Chunks und Embeddings lokal speichert und semantische Aehnlichkeitssuche ausfuehrt.
- `retrieval.py` verbindet Anfrage-Embedding und Vektordatenbank.
- `generation.py` enthaelt die Klasse `AnswerGenerator`, die aus Frage und Chunks einen Prompt baut und die Antwort generiert.
- `ui.py` ist die optionale Streamlit-Oberflaeche und gehoert nicht zur fachlichen Standard-RAG-Kernlogik.

## Pipeline-Sicht

```mermaid
flowchart LR
    A["PDF-Dokument"] --> B["DocumentLoader"]
    B --> C["Chunking"]
    C --> D["EmbeddingModel"]
    D --> E["LocalVectorStore"]
    F["Nutzerfrage"] --> G["EmbeddingModel"]
    G --> H["Retrieval"]
    E --> H
    H --> I["Relevante Chunks"]
    I --> J["AnswerGenerator"]
    F --> J
    J --> K["Antwort"]
```
