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
                            "Beschreibe besonders Tabellen, Bilder, Diagramme, "
                            "Beschriftungen, Formeln, Formelzeichen, Rechenwege, "
                            "Zahlenwerte und technische Zusammenhaenge. "
                            "Gib Tabellen wenn moeglich als Markdown-Tabelle aus. "
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


def load_pdf_with_visuals(pdf_path: str) -> str:
    """Laedt ein PDF und gibt Text plus beschriebene Bild- und Tabelleninhalte zurueck."""

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
# Auswahlfunktion fuer das Evaluationsmodul
# ==========================================================

def load_pdf_by_type(
    pdf_path: str,
    loader_type: str,
) -> str:
    """
    Waehlt den gewuenschten Dokumentloader fuer die Evaluation aus.

    Moegliche Werte fuer loader_type:
    - "text"
    - "targeted_visual"
    - "full_visual"

    main.py verwendet weiterhin unveraendert die Funktion load_pdf().
    """

    if loader_type == "text":
        return load_pdf_text(pdf_path)

    if loader_type == "targeted_visual":
        return load_pdf_text_with_targeted_visuals(
            pdf_path,
            visual_keywords=[
                "diagramm",
                "abbildung",
                "tabelle",
                "figure",
                "chart",
            ],
        )

    if loader_type == "full_visual":
        return load_pdf_with_visuals(pdf_path)

    raise ValueError(
        f"Unbekannter Loader: {loader_type}. "
        "Erlaubt sind: text, targeted_visual, full_visual."
    )


# ==========================================================
# Variante 1:
# Nur direkt auslesbarer PDF-Text
# ==========================================================

# def load_pdf(pdf_path: str) -> str:
#     """Standard-Loader: nur direkt auslesbarer PDF-Text."""
#     return load_pdf_text(pdf_path)


# ==========================================================
# Variante 2:
# PDF-Text + Vision nur fuer Seiten mit Schluesselwoertern
# ==========================================================

# def load_pdf(pdf_path: str) -> str:
#     """Standard-Loader: Text sowie Vision nur auf relevanten Seiten."""
#     return load_pdf_text_with_targeted_visuals(
#         pdf_path,
#         visual_keywords=[
#             "diagramm",
#             "abbildung",
#             "tabelle",
#             "figure",
#             "chart",
#         ],
#     )


# ==========================================================
# Variante 3 (Standard):
# Vollstaendige visuelle Analyse aller Seiten
# ==========================================================

def load_pdf(pdf_path: str) -> str:
    """Standard-Loader: Text sowie Bild-, Tabellen- und Diagramminformationen aller Seiten."""
    return load_pdf_with_visuals(pdf_path)