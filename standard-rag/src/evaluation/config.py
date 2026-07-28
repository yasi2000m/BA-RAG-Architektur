from pathlib import Path

# ============================
# PDF
# ============================

PDF_PATH = "/Users/yasi/Documents/New project/BA-RAG-Architektur/standard-rag/data/Elektrotechnik 3.pdf"

# ============================
# Aktuelle Standardparameter
# ============================

DOCUMENT_LOADER = "text"

CHUNK_SIZE = 250

OVERLAP = 50

TOP_K = 5

TEMPERATURE = 0.2

# ============================
# Zu testende Werte
# ============================

DOCUMENT_LOADERS = [
    "text",
    "targeted_visual",
    "full_visual",
]

CHUNK_SIZES = [
    100,
    150,
    250,
    400,
    600,
]

OVERLAPS = [
    0,
    25,
    50,
    75,
    100,
]

TOP_K_VALUES = [
    1,
    3,
    5,
    7,
    10,
]

TEMPERATURE_VALUES = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
]

DOCUMENT_LOADER_NAMES = {
    "text": "Nur direkt auslesbarer PDF-Text",
    "targeted_visual": "PDF-Text mit gezielter visueller Analyse",
    "full_visual": "PDF-Text mit vollständiger visueller Analyse",
}

VISUAL_KEYWORDS = [
    "diagramm",
    "abbildung",
    "tabelle",
    "figure",
    "chart",
]

LOADER_TEST_REPETITIONS = 1

# Durchschnittliche geschätzte Kosten je Vision-Aufruf.
# 0.0 bedeutet, dass noch keine Kostenschätzung erfolgt.
VISION_COST_PER_CALL = 0.0

CURRENCY = "USD"

# Anzahl der Wiederholungen je Temperature-Wert.
# Für einen ersten Funktionstest reicht 1.
# Für die spätere Untersuchung sind beispielsweise
# 3 oder 5 Wiederholungen sinnvoll.
TEMPERATURE_TEST_REPETITIONS = 1

# Preise des verwendeten LLM-Modells pro eine Million Tokens.
# Solange beide Werte 0.0 sind, wird der Tokenverbrauch
# gemessen, aber keine Kostensumme berechnet.
LLM_INPUT_PRICE_PER_MILLION = 0.40
LLM_OUTPUT_PRICE_PER_MILLION = 1.60

# ============================
# Ausgabeordner
# ============================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)