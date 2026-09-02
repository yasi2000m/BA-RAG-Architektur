import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# Pfade
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

RAG_PATHS = {
    "Standard RAG": ROOT_DIR / "standard-rag" / "src",
    "Graph RAG": ROOT_DIR / "graph-rag-basic" / "src",
    "Fusion RAG": ROOT_DIR / "fusion-rag" / "src",
}

LOGO_PATH = ROOT_DIR / "assets" / "hs_aalen_logo.png"
ICON_PATH = ROOT_DIR / "assets" / "hs_aalen_icon.png"


# --------------------------------------------------
# Streamlit-Konfiguration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Chatbot | Hochschule Aalen",
    page_icon=str(ICON_PATH),
    layout="centered",
)


# --------------------------------------------------
# Design / CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Gesamter Hintergrund */
    .stApp {
        background-color: #ffffff;
        color: #30343b;
    }


    /* Hauptbereich */
    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* Streamlit Header */
    [data-testid="stHeader"] {
        background-color: #ffffff;
    }


    /* Titel */
    h1 {
        color: #30343b;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }


    /* Hochschule-Aalen Farbverlauf */
    .hs-gradient {
        height: 5px;
        width: 100%;
        border-radius: 10px;

        background: linear-gradient(
            90deg,
            #62b5dc 0%,
            #527cc5 35%,
            #5749a6 65%,
            #bd1585 100%
        );

        margin-top: 15px;
        margin-bottom: 30px;
    }


    /* Untertitel */
    .subtitle {
        color: #666666;
        font-size: 1.05rem;
        margin-bottom: 0;
    }


    /* Allgemeine Input-Felder */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #30343b !important;

        border: 1px solid #d9dce3;
        border-radius: 10px;
    }


    .stTextInput input:focus {
        border-color: #527cc5 !important;
        box-shadow: 0 0 0 1px #527cc5 !important;
    }


    /* Radio Auswahl */
    [data-testid="stRadio"] {
        background-color: #f7f8fb;

        padding: 15px 20px;

        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }


    /* Chat Input */
    [data-testid="stChatInput"] {
        background-color: #ffffff;
    }


    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #30343b !important;

        border-radius: 12px;
    }


    /* Chatnachrichten */
    [data-testid="stChatMessage"] {
        background-color: #f8f9fc;

        border: 1px solid #e7e9ef;
        border-radius: 14px;

        padding: 10px 15px;
        margin-bottom: 10px;
    }


    /* Expander */
    [data-testid="stExpander"] {
        background-color: #fafafa;

        border: 1px solid #e6e6e6;
        border-radius: 10px;
    }


    /* Success Nachricht */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* Caption / Kennzahlen */
    .stCaption {
        color: #5749a6;
    }


    /* Links */
    a {
        color: #527cc5 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.image(LOGO_PATH, width=320)

st.title("RAG Chatbot")

st.markdown(
    """
    <p class="subtitle">
        Vergleich verschiedener Retrieval-Augmented-Generation-Architekturen
    </p>

    <div class="hs-gradient"></div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# RAG auswählen
# --------------------------------------------------

st.markdown("### RAG-Architektur auswählen")

rag = st.radio(
    "RAG-Architektur",
    RAG_PATHS.keys(),
    horizontal=True,
    label_visibility="collapsed",
)


# --------------------------------------------------
# API-Key
# --------------------------------------------------

st.markdown("### OpenAI API-Key")

api_key = st.text_input(
    "API-Key",
    type="password",
    placeholder="API-Key eingeben",
    label_visibility="collapsed",
)

if api_key:
    st.success("API-Key wurde eingegeben.")


# --------------------------------------------------
# Session State
# --------------------------------------------------

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_rag", rag)

if st.session_state.last_rag != rag:
    st.session_state.messages = []
    st.session_state.last_rag = rag


# --------------------------------------------------
# Bisherigen Chat anzeigen
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "metrics" in message:
            st.caption(message["metrics"])


# --------------------------------------------------
# Eingabe
# --------------------------------------------------

query = st.chat_input("Stellen Sie eine Frage...")


if query:

    if not api_key:
        st.error("Bitte zuerst einen API-Key eingeben.")
        st.stop()


    # Nutzerfrage speichern
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })


    with st.chat_message("user"):
        st.markdown(query)


    # --------------------------------------------------
    # RAG vorbereiten
    # --------------------------------------------------

    env = os.environ.copy()

    env["OPENAI_API_KEY"] = api_key


    runner_code = """
import sys
import io
import json
import contextlib

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):
    from main import main
    result = main(sys.argv[1])

print(json.dumps({
    "result": result,
    "logs": buffer.getvalue()
}, ensure_ascii=False))
"""


    # --------------------------------------------------
    # RAG ausführen
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Antwort wird generiert..."):

            process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    runner_code,
                    query,
                ],
                cwd=RAG_PATHS[rag],
                env=env,
                capture_output=True,
                text=True,
            )


        # --------------------------------------------------
        # Fehler prüfen
        # --------------------------------------------------

        if process.returncode != 0:

            st.error(
                "Bei der Ausführung ist ein Fehler aufgetreten."
            )

            st.code(process.stderr)

            st.stop()


        # --------------------------------------------------
        # Ergebnis einlesen
        # --------------------------------------------------

        try:

            payload = json.loads(
                process.stdout
            )

        except json.JSONDecodeError:

            st.error(
                "Die RAG-Ausgabe konnte nicht verarbeitet werden."
            )

            st.code(process.stdout)

            st.stop()


        result = payload["result"]

        logs = payload["logs"]

        answer = result["answer"]


        # --------------------------------------------------
        # Antwort anzeigen
        # --------------------------------------------------

        st.markdown(answer)


        # --------------------------------------------------
        # Kennzahlen
        # --------------------------------------------------

        if rag == "Standard RAG":

            metrics_text = (
                f"Top-k: {result['top_k']} · "
                f"Tokenverbrauch: {result['used_tokens']}"
            )


        elif rag == "Fusion RAG":

            metrics_text = (
                f"Top-k: {result['top_k']} · "
                f"Query-Varianten: "
                f"{result['num_query_variants']} · "
                f"Retrieval-Zeit: "
                f"{result['retrieval_time']:.2f} s · "
                f"Tokenverbrauch: "
                f"{result['used_tokens']}"
            )


        else:  # Graph RAG

            metrics_text = (
                f"Max Depth: {result['max_depth']} · "
                f"Entitäten: {result['entities']} · "
                f"Beziehungen: "
                f"{result['relationships']} · "
                f"Tokenverbrauch: "
                f"{result['used_tokens']}"
            )


        st.caption(metrics_text)


        # Technische Ausgaben eher sekundär darstellen
        with st.expander("Technische Ausgaben"):
            st.text(logs)


    # --------------------------------------------------
    # Antwort speichern
    # --------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "metrics": metrics_text,
    })
