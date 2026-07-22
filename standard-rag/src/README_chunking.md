# README zu `chunking.py`

Dieses Modul teilt einen langen Text in kleinere Textabschnitte. Diese Abschnitte heissen Chunks.

Chunking ist notwendig, weil ein RAG-System nicht den gesamten PDF-Text auf einmal sucht oder in den Prompt packt. Stattdessen werden kleinere Chunks eingebettet und spaeter semantisch durchsucht.

## Funktion `chunk_text`

```python
def chunk_text(
```

Hier beginnt die Funktionsdefinition.

```python
    text: str,
```

`text` ist der lange Eingabetext, zum Beispiel der extrahierte PDF-Inhalt.

```python
    chunk_size: int = 800,
```

`chunk_size` legt fest, wie viele Woerter ein Chunk maximal enthalten soll. Der Standardwert ist 800.

In der UI wird aktuell bewusst `250` verwendet, damit kurze Inhalte wie Tabellen besser auffindbar sind.

```python
    overlap: int = 100,
```

`overlap` legt fest, wie viele Woerter sich zwei benachbarte Chunks teilen. Der Standardwert ist 100.

In der UI wird aktuell `50` verwendet.

```python
) -> list[str]:
```

Die Funktion gibt eine Liste von Strings zurueck. Jeder String ist ein Chunk.

```python
"""
Teilt einen einzelnen Text in einfache, ueberlappende Chunks auf.
...
"""
```

Der Docstring erklaert den Zweck der Funktion.

```python
if chunk_size <= 0:
```

Die Funktion prueft, ob die Chunk-Groesse gueltig ist.

```python
raise ValueError("chunk_size muss groesser als 0 sein.")
```

Wenn `chunk_size` 0 oder kleiner ist, wird ein Fehler ausgelöst.

```python
if overlap < 0:
```

Hier wird geprueft, ob der Overlap negativ ist.

```python
raise ValueError("overlap darf nicht negativ sein.")
```

Ein negativer Overlap ist nicht sinnvoll und fuehrt zu einem Fehler.

```python
if overlap >= chunk_size:
```

Hier wird geprueft, ob der Overlap kleiner als die Chunk-Groesse ist.

```python
raise ValueError("overlap muss kleiner als chunk_size sein.")
```

Wenn der Overlap zu gross ist, wuerde sich der Startpunkt nicht sinnvoll nach vorne bewegen.

```python
words = text.split()
```

Der Eingabetext wird in einzelne Woerter zerlegt.

```python
chunks: list[str] = []
```

Eine leere Liste fuer die spaeteren Chunks wird erstellt.

```python
start = 0
```

Der erste Chunk beginnt beim ersten Wort.

```python
while start < len(words):
```

Die Schleife laeuft, solange noch Woerter uebrig sind.

```python
end = start + chunk_size
```

Das Ende des aktuellen Chunks wird berechnet.

```python
chunk = " ".join(words[start:end])
```

Der Ausschnitt aus der Wortliste wird wieder zu einem normalen Textblock zusammengesetzt.

```python
if chunk:
```

Es wird geprueft, ob der Chunk nicht leer ist.

```python
chunks.append(chunk)
```

Der Chunk wird zur Ergebnisliste hinzugefuegt.

```python
start += chunk_size - overlap
```

Der Startpunkt wird verschoben. Durch `chunk_size - overlap` entsteht eine Ueberlappung zwischen benachbarten Chunks.

```python
return chunks
```

Alle Chunks werden zurueckgegeben.
