import base64
import os

import fitz
from dotenv import load_dotenv
from openai import OpenAI


def describe_page_image(client: OpenAI, model: str, image_bytes: bytes) -> str:
    """Wandelt sichtbare Tabellen, Bilder und Diagramme einer PDF-Seite in Text um."""
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
                            "Beschreibe besonders Tabellen, Bilder, Diagramme, Beschriftungen, "
                            "Zahlenwerte und technische Zusammenhaenge. "
                            "Gib Tabellen wenn moeglich als Markdown-Tabelle aus. "
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


def load_pdf_text(pdf_path: str) -> str:
    """Laedt ein PDF und gibt den direkt auslesbaren Text zurueck."""
    document = fitz.open(pdf_path)
    pages = [
        f"Seite {page_number}\n\n{page.get_text('text').strip()}"
        for page_number, page in enumerate(document, start=1)
    ]
    return "\n\n".join(pages).strip()


def load_pdf_text_with_targeted_visuals(
    pdf_path: str,
    visual_keywords: list[str] | None = None,
) -> str:
    """
    Laedt PDF-Text schnell und beschreibt nur Seiten mit passenden Stichworten visuell.

    So koennen Tabellen/Bilder zu wichtigen Treffern beruecksichtigt werden,
    ohne jede PDF-Seite mit dem Vision-Modell zu analysieren.
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
            page_image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
            visual_text = describe_page_image(client, vision_model, page_image)

        page_content = f"Seite {page_number}\n\nText aus PDF:\n{page_text}"

        if visual_text:
            page_content += f"\n\nZusatzinformationen aus Tabelle/Bild/Diagramm:\n{visual_text}"

        pages.append(page_content)

    return "\n\n".join(pages).strip()


def load_pdf_with_visuals(pdf_path: str) -> str:
    """Laedt ein PDF und gibt Text plus beschriebene Bild- und Tabelleninhalte zurueck."""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

    document = fitz.open(pdf_path)
    pages: list[str] = []

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text("text").strip()
        page_image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
        visual_text = describe_page_image(client, vision_model, page_image)

        pages.append(
            f"Seite {page_number}\n\n"
            f"Text aus PDF:\n{page_text}\n\n"
            f"Informationen aus Bildern, Tabellen und Diagrammen:\n{visual_text}"
        )

    return "\n\n".join(pages).strip()


def load_pdf(pdf_path: str) -> str:
    """Standard-Loader: schnell, nur direkt auslesbarer PDF-Text."""
    return load_pdf_text(pdf_path)
