# Der Document Loader wandelt das ursprüngliche PDF-Dokument in einen möglichst vollständigen maschinenlesbaren Text um.
# Dieser Text wird anschließend an das nächste Modul, typischerweise den Chunker, übergeben.


import base64     # Wandelt ein Seitenbild in einen Base64-String um, damit es an das Vision-Modell geschickt werden kann.
import os         # Liest Umgebungsvariablen wie den OpenAI-API-Key aus.
import time       # Wird zur Messung der Laufzeit des Document Loaders verwendet.
import fitz       # PyMuPDF. Öffnet PDFs, extrahiert Text und rendert PDF-Seiten als Bilder.

from dotenv import load_dotenv
from openai import OpenAI


# Diese Funktion bekommt eine PDF-Seite als Bild und lässt deren sichtbaren Inhalt durch ein Vision-Modell in Text umwandeln.
# --> Parameter: 
# client: bereits erzeugter OpenAI-Client
# model: Name des verwendeten Vision-Modells
# image_bytes: Bilddaten der PDF-Seite
# -> str: Die Funktion gibt einen Text zurück

def describe_page_image(client: OpenAI, model: str, image_bytes: bytes) -> str:

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extrahiere die sichtbaren Informationen aus dieser PDF-Seite. "
                            "Beschreibe besonders Tabellen, Bilder, Diagramme, "
                            "Beschriftungen, Formeln, Formelzeichen, Rechenwege, "
                            "Zahlenwerte und technische Zusammenhaenge. "
                            "Lies alle Inhalte aus Tabellen und Bildern"
                            "Wenn eine Tabelle, Aufzaehlung oder nummerierte Liste sichtbar ist, "
                            "gib alle einzelnen Eintraege vollstaendig und in ihrer Reihenfolge wieder. "
                            "Fasse die Eintraege nicht nur zusammen. "
                            "Gib Tabellen nicht als Markdown-Tabelle aus. "
                            "Gib jede Tabellenzeile als eigenen nummerierten Punkt aus. "
                            "Uebernimm alle sichtbaren Eintraege vollstaendig. "
                            "Erzeuge keine Trennlinien aus Bindestrichen oder Sonderzeichen. "
                            "Gib Formeln in gut lesbarer Textform aus. "
                            "Erfinde keine Informationen."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ],
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content or ""


# Variante1:
# Öffnet eine PDF-Datei und extrahiert den direkt enthaltenen Text seitenweise. 
# Visuelle Inhalte wie Bilder oder Diagramme werden dabei nicht zusätzlich analysiert.

def load_pdf_text(pdf_path: str) -> str:

    document = fitz.open(pdf_path)

    pages = [
        f"Seite {page_number}\n\n{page.get_text('text').strip()}"
        for page_number, page in enumerate(document, start=1)
    ]

    return "\n\n".join(pages).strip()

# Variante 2: 
#   Extrahiert den Text jeder PDF-Seite und führt nur für Seiten, die festgelegte Schlüsselwörter enthalten, zusätzlich eine visuelle Analyse durch. 
#   Dadurch können Tabellen und Diagramme erfasst werden, ohne jede Seite durch das Vision-Modell zu senden.

def load_pdf_text_with_targeted_visuals(
    pdf_path: str,
    visual_keywords: list[str] | None = None,
) -> str:
    """
    Laedt den PDF-Text und analysiert visuell nur Seiten,
    die mindestens eines der angegebenen Schluesselwoerter enthalten.
    """

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

    keywords = [keyword.lower() for keyword in (visual_keywords or [])]

    document = fitz.open(pdf_path)
    pages: list[str] = []

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text("text").strip()
        visual_text = ""

        if any(keyword in page_text.lower() for keyword in keywords):
            page_image = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
            ).tobytes("png")

            visual_text = describe_page_image(
                client,
                vision_model,
                page_image,
            )

        page_content = (
            f"Seite {page_number}\n\n"
            f"Text aus PDF:\n{page_text}"
        )

        if visual_text:
            page_content += (
                "\n\n"
                "Zusatzinformationen aus Bildern, Tabellen und Diagrammen:\n"
                f"{visual_text}"
            )

        pages.append(page_content)

    return "\n\n".join(pages).strip()



# Variante 3: 
# Extrahiert den direkt verfügbaren Text jeder PDF-Seite und analysiert zusätzlich jede Seite vollständig mit einem Vision-Modell. 
# Beide Informationsquellen werden anschließend zu einem gemeinsamen Text zusammengeführt.

def load_pdf_with_visuals(pdf_path: str) -> str:

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

    document = fitz.open(pdf_path)
    pages: list[str] = []

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text("text").strip()

        page_image = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False,
        ).tobytes("png")

        visual_text = describe_page_image(
            client,
            vision_model,
            page_image,
        )

        pages.append(
            f"Seite {page_number}\n\n"
            f"Text aus PDF:\n{page_text}\n\n"
            f"Informationen aus Bildern, Tabellen und Diagrammen:\n{visual_text}"
        )

    return "\n\n".join(pages).strip()


# ==========================================================
# Variante 1:
# Nur direkt auslesbarer PDF-Text
# Ablauf: PDF --> PyMuPDF --> Text
# ==========================================================

# def load_pdf(pdf_path: str) -> str:
#     start_time = time.perf_counter()

#     text = load_pdf_text(pdf_path)

#     duration = time.perf_counter() - start_time

#     print(f"Laufzeit: {duration:.2f} Sekunden")
#     print(f"Textgroesse: {len(text.encode('utf-8')) / 1024:.2f} KB")

#     with open("standard-rag/data/geladener_text.txt", "w", encoding="utf-8") as file:
#         file.write(text)

#     return text


# ==========================================================
# Variante 2:
# PDF-Text + Vision nur fuer Seiten mit Schluesselwoertern
# Ablauf: PDF --> Text extrahieren --> schlüsselwort vorhanden? --> Nein: nur Text / Ja: Vision
# ==========================================================

# def load_pdf(pdf_path: str) -> str:
#     start_time = time.perf_counter()

#     text = load_pdf_text_with_targeted_visuals(
#         pdf_path,
#         visual_keywords=[
#             "abbildung",
#             "diagramm",
#             "schaltbild",
#             "schaltung",
#             "ersatzschaltbild",
#             "kennlinie",
#             "tabelle",
#             "formel",
#             "vektordiagramm",
#             "phasendiagramm",
#             "blockschaltbild",
#             "messschaltung",
#             "stromlaufplan",
#             "kurve",
#             "graph",
#         ],
#     )

#     duration = time.perf_counter() - start_time

#     print(f"Laufzeit: {duration:.2f} Sekunden")
#     print(f"Textgroesse: {len(text.encode('utf-8')) / 1024:.2f} KB")

#     return text


# # ==========================================================
# Variante 3:
# Vollstaendige visuelle Analyse aller Seiten
# ==========================================================

def load_pdf(pdf_path: str) -> str:
    start_time = time.perf_counter()

    text = load_pdf_with_visuals(pdf_path)

    duration = time.perf_counter() - start_time

    print(f"Laufzeit: {duration:.2f} Sekunden")
    print(f"Textgroesse: {len(text.encode('utf-8')) / 1024:.2f} KB")

    # with open("standard-rag/data/geladener_text.txt", "w", encoding="utf-8") as file:
    #     file.write(text)

    return text



if __name__ == "__main__":
    pdf_path = "/Users/yasi/Documents/New project/BA-RAG-Architektur/standard-rag/data/Elektrotechnik 3.pdf"
    load_pdf(pdf_path)
