# README zu `generation.py`

Dieses Modul ist fuer Augmentation und Generation zustaendig.

Augmentation bedeutet: Nutzerfrage und relevante Chunks werden zu einem Prompt kombiniert.

Generation bedeutet: Das LLM erzeugt aus diesem Prompt die Antwort.

## Imports

```python
import os
```

`os` liest Umgebungsvariablen.

```python
from dotenv import load_dotenv
```

`load_dotenv()` laedt die `.env`-Datei.

```python
from openai import OpenAI
```

`OpenAI` ist der Client fuer den LLM-Aufruf.

## Klasse `AnswerGenerator`

```python
class AnswerGenerator:
```

Diese Klasse kapselt Prompt-Erstellung und Antwortgenerierung.

## Konstruktor

```python
def __init__(self, model_name: str | None = None) -> None:
```

Der Konstruktor wird beim Erstellen von `AnswerGenerator()` ausgefuehrt.

```python
load_dotenv()
```

Die `.env`-Datei wird geladen.

```python
self.model_name = model_name or os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
```

Das LLM-Modell wird bestimmt.

Reihenfolge:

1. uebergebener Modellname
2. `OPENAI_LLM_MODEL` aus `.env`
3. Standard `gpt-4.1-mini`

```python
self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

Der OpenAI-Client wird erstellt.

## Methode `build_prompt`

```python
def build_prompt(self, query: str, relevant_chunks: list[str]) -> str:
```

Diese Methode erstellt den Prompt.

`query` ist die Nutzerfrage.

`relevant_chunks` sind die abgerufenen Textstellen.

```python
context = "\n\n".join(
```

Die relevanten Chunks werden zu einem Kontextblock verbunden.

```python
f"Kontext {index + 1}:\n{chunk}"
```

Jeder Chunk bekommt eine Nummer, zum Beispiel `Kontext 1`.

```python
for index, chunk in enumerate(relevant_chunks)
```

Alle Chunks werden mit ihrer Position durchlaufen.

```python
return f"""Beantworte die Frage ausschliesslich auf Basis des folgenden Kontexts.
```

Der finale Prompt beginnt. Das Modell wird angewiesen, nur den gegebenen Kontext zu verwenden.

```python
Wenn die Antwort nicht im Kontext enthalten ist, sage, dass die Information im Kontext nicht vorhanden ist.
```

Diese Zeile verhindert, dass das Modell frei halluziniert.

```python
{context}
```

Hier werden die gefundenen Chunks eingefuegt.

```python
Frage:
{query}
```

Hier wird die Nutzerfrage eingefuegt.

```python
Antwort:"""
```

Das Modell soll danach die Antwort schreiben.

## Methode `generate_answer`

```python
def generate_answer(self, query: str, relevant_chunks: list[str]) -> str:
```

Diese Methode erzeugt die finale Antwort.

```python
prompt = self.build_prompt(query, relevant_chunks)
```

Zuerst wird der Prompt gebaut.

```python
response = self.client.chat.completions.create(
```

Der LLM-Aufruf beginnt.

```python
model=self.model_name,
```

Das festgelegte Modell wird verwendet.

```python
messages=[
    {"role": "user", "content": prompt},
],
```

Der Prompt wird als User-Nachricht gesendet.

```python
temperature=0.2,
```

Eine niedrige Temperatur sorgt fuer eher stabile Antworten.

```python
return response.choices[0].message.content or ""
```

Der Antworttext wird aus der API-Antwort gelesen. Falls kein Text vorhanden ist, wird ein leerer String zurueckgegeben.
