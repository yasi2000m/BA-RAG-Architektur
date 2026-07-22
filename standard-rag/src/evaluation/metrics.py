import csv
import sys
import time
from pathlib import Path


def start_timer() -> float:
    """Startet die Zeitmessung."""
    return time.perf_counter()


def stop_timer(start_time: float) -> float:
    """Beendet die Zeitmessung."""
    return time.perf_counter() - start_time


def calculate_text_size(text: str) -> int:
    """Speicherbedarf eines Textes in Byte."""
    return len(text.encode("utf-8"))


def calculate_chunks_size(chunks: list[str]) -> int:
    """Gesamter Speicherbedarf aller Chunks in Byte."""
    return sum(calculate_text_size(chunk) for chunk in chunks)


def format_bytes(size: int) -> str:
    """Formatiert Byte in KB oder MB."""
    if size < 1024:
        return f"{size} B"

    if size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"

    return f"{size / (1024 * 1024):.2f} MB"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Schreibt Ergebnisse in eine CSV-Datei."""

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)


def print_title(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)