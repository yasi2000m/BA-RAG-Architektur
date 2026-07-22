# README zu `document_loader.py`

Dieses Modul ist fuer das Laden der PDF-Wissensquelle zustaendig. Es kann drei Varianten:

- nur direkt auslesbaren PDF-Text laden
- gezielt bestimmte Seiten zusaetzlich visuell analysieren
- alle Seiten visuell analysieren

Damit bleibt die Standard-RAG-Pipeline erhalten: Am Ende liefert der Loader immer Text, der danach gechunkt und eingebettet werden kann.

## Imports

```python
import base64
```

`base64` wird benutzt, um ein gerendertes PDF-Seitenbild in einen Base64-String umzuwandeln. Dieser String kann dann an das Vision-Modell uebergeben werden.

```python
import os
```

`os` liest Umgebungsvariablen aus, zum Beispiel `OPENAI_API_KEY` und `OPENAI_VISION_MODEL`.

```python
import fitz
```

`fitz` kommt aus `PyMuPDF`. Damit wird die PDF geoeffnet, Text wird extrahiert und Seiten koennen als Bild gerendert werden.

```python
from dotenv import load_dotenv
```

`load_dotenv()` laedt Werte aus der `.env`-Datei in die Umgebung.

```python
from openai import OpenAI
```

`OpenAI` ist der API-Client. Er wird fuer das Vision-Modell verwendet, wenn PDF-Seiten als Bild beschrieben werden sollen.

## Funktion `describe_page_image`

```python
def describe_page_image(client: OpenAI, model: str, image_bytes: bytes) -> str:
```

Diese Funktion beschreibt eine einzelne PDF-Seite, die vorher als Bild gerendert wurde.

`client` ist der OpenAI-Client.

`model` ist der Name des Vision-faehigen Modells.

`image_bytes` sind die Bilddaten der PDF-Seite.

```python
image_base64 = base64.b64encode(image_bytes).decode("utf-8")
```

Die Bilddaten werden Base64-kodiert. Danach werden sie als UTF-8-String gelesen, damit sie in eine JSON/API-Nachricht eingefuegt werden koennen.

```python
response = client.chat.completions.create(
```

Hier wird das Vision-Modell aufgerufen.

```python
model=model,
```

Das uebergebene Modell wird verwendet.

```python
messages=[
```

Die Nachricht an das Modell beginnt.

```python
{
    "role": "user",
```

Die Anfrage wird als User-Nachricht gesendet.

```python
"content": [
```

Der Inhalt besteht aus mehreren Teilen: einem Textauftrag und einem Bild.

```python
{
    "type": "text",
```

Der erste Inhaltsteil ist Text.

```python
"text": (
    "Extrahiere die sichtbaren Informationen aus dieser PDF-Seite. "
    ...
)
```

Dieser Prompt sagt dem Modell, was es aus der Seite lesen soll: Tabellen, Bilder, Diagramme, Beschriftungen, Zahlenwerte und technische Zusammenhaenge.

```python
"Gib Tabellen wenn moeglich als Markdown-Tabelle aus. "
```

Diese Anweisung macht Tabellen spaeter besser lesbar und besser auffindbar.

```python
"Erfinde keine Informationen."
```

Das Modell soll nur sichtbare Inhalte beschreiben und nichts hinzuerfinden.

```python
{
    "type": "image_url",
```

Der zweite Inhaltsteil ist ein Bild.

```python
"url": f"data:image/png;base64,{image_base64}",
```

Das Bild wird als Data-URL an das Modell uebergeben.

```python
temperature=0,
```

Die Ausgabe soll moeglichst deterministisch und sachlich sein.

```python
return response.choices[0].message.content or ""
```

Die Textantwort des Vision-Modells wird zurueckgegeben. Falls keine Antwort vorhanden ist, wird ein leerer String zurueckgegeben.

## Funktion `load_pdf_text`

```python
def load_pdf_text(pdf_path: str) -> str:
```

Diese Funktion laedt nur direkt auslesbaren PDF-Text. Sie ist schnell und verursacht keine Vision-Kosten.

```python
document = fitz.open(pdf_path)
```

Die PDF-Datei wird geoeffnet.

```python
pages = [
```

Eine Liste fuer alle Seitentexte wird aufgebaut.

```python
f"Seite {page_number}\n\n{page.get_text('text').strip()}"
```

Fuer jede Seite wird die Seitennummer und der extrahierte Text gespeichert.

`page.get_text('text')` liest den normalen PDF-Text.

`strip()` entfernt ueberfluessige Leerzeichen am Anfang und Ende.

```python
for page_number, page in enumerate(document, start=1)
```

Alle Seiten der PDF werden durchlaufen. Die Nummerierung beginnt bei 1, damit sie zur menschlichen Seitenzaehlung passt.

```python
return "\n\n".join(pages).strip()
```

Alle Seitentexte werden zu einem langen Text verbunden und zurueckgegeben.

## Funktion `load_pdf_text_with_targeted_visuals`

```python
def load_pdf_text_with_targeted_visuals(
    pdf_path: str,
    visual_keywords: list[str] | None = None,
) -> str:
```

Diese Funktion ist der praktische Mittelweg fuer die UI.

Sie extrahiert zuerst schnell den normalen PDF-Text. Nur wenn eine Seite bestimmte Stichwoerter enthaelt, wird diese Seite zusaetzlich als Bild analysiert.

```python
load_dotenv()
```

Die `.env`-Datei wird geladen.

```python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

Der OpenAI-Client wird mit dem API-Key erstellt.

```python
vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
```

Das Vision-Modell wird aus der `.env` gelesen. Falls nichts gesetzt ist, wird `gpt-4.1-mini` verwendet.

```python
keywords = [keyword.lower() for keyword in (visual_keywords or [])]
```

Alle Suchbegriffe werden in Kleinbuchstaben umgewandelt. Dadurch wird die Suche einfacher und robust gegen Gross-/Kleinschreibung.

```python
document = fitz.open(pdf_path)
```

Die PDF wird geoeffnet.

```python
pages: list[str] = []
```

Eine Liste fuer die fertigen Seitentexte wird erstellt.

```python
for page_number, page in enumerate(document, start=1):
```

Alle Seiten werden nacheinander verarbeitet.

```python
page_text = page.get_text("text").strip()
```

Der normale Text der Seite wird extrahiert.

```python
visual_text = ""
```

Zunaechst gibt es noch keinen visuellen Zusatztext.

```python
if any(keyword in page_text.lower() for keyword in keywords):
```

Diese Bedingung prueft, ob eines der Stichwoerter im Seitentext vorkommt.

In der UI wird zum Beispiel nach `Sicherheitsregeln` gesucht. Dadurch wird die relevante Tabellenseite gezielt visuell analysiert.

```python
page_image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
```

Die Seite wird als PNG-Bild gerendert.

`fitz.Matrix(2, 2)` erhoeht die Aufloesung.

`alpha=False` entfernt Transparenz, damit das Bild einfacher verarbeitet werden kann.

```python
visual_text = describe_page_image(client, vision_model, page_image)
```

Das gerenderte Seitenbild wird an das Vision-Modell gesendet und als Text beschrieben.

```python
page_content = f"Seite {page_number}\n\nText aus PDF:\n{page_text}"
```

Der normale Seitentext wird in einen strukturierten Textblock geschrieben.

```python
if visual_text:
```

Wenn es visuelle Zusatzinformationen gibt, werden sie angehaengt.

```python
page_content += f"\n\nZusatzinformationen aus Tabelle/Bild/Diagramm:\n{visual_text}"
```

Die Vision-Beschreibung wird dem Seitentext hinzugefuegt.

```python
pages.append(page_content)
```

Der fertige Seitenblock wird gespeichert.

```python
return "\n\n".join(pages).strip()
```

Alle Seitenbloecke werden zu einem langen Text verbunden.

## Funktion `load_pdf_with_visuals`

Diese Funktion analysiert jede Seite visuell. Sie ist vollstaendiger, aber langsamer und teurer.

Sie folgt derselben Logik wie der gezielte Modus, nur ohne Stichwortfilter. Jede Seite wird als Bild gerendert und beschrieben.

## Funktion `load_pdf`

```python
def load_pdf(pdf_path: str) -> str:
```

Diese Funktion ist ein einfacher Standard-Wrapper.

```python
return load_pdf_text(pdf_path)
```

Standardmaessig wird nur schneller PDF-Text geladen. Die UI verwendet bewusst den gezielten Modus, damit wichtige Tabellen dennoch beruecksichtigt werden koennen.
