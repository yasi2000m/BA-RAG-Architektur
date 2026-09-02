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

LOGO_PATH = ROOT_DIR / "hs_aalen_logo.png"
ICON_PATH = ROOT_DIR / "hs_aalen_icon.png"


# --------------------------------------------------
# Streamlit-Konfiguration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Chatbot | Hochschule Aalen",
    page_icon=str(ICON_PATH),
    layout="centered",
)


# --------------------------------------------------
# Design
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* --------------------------------------------------
       Allgemein
       -------------------------------------------------- */

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
        color: #262730 !important;
    }


    /* Hauptbereich */
    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }


    /* Streamlit Header */
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }


    /* --------------------------------------------------
       Überschriften und Text
       -------------------------------------------------- */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #2F333B !important;
    }


    h1 {
        font-weight: 700;
        margin-bottom: 0.2rem;
    }


    p,
    li,
    label {
        color: #2F333B;
    }


    .subtitle {
        color: #666A73;
        font-size: 1.05rem;
        margin-bottom: 0;
    }


    /* --------------------------------------------------
       Hochschule-Aalen Farbverlauf
       -------------------------------------------------- */

    .hs-gradient {
        width: 100%;
        height: 5px;

        margin-top: 15px;
        margin-bottom: 35px;

        border-radius: 10px;

        background: linear-gradient(
            90deg,
            #62B5DC 0%,
            #527CC5 35%,
            #5749A6 65%,
            #BD1585 100%
        );
    }


    /* --------------------------------------------------
       Radio / RAG Auswahl
       -------------------------------------------------- */

    [data-testid="stRadio"] {
        background-color: #F6F7FA !important;

        border: 1px solid #DDE1E8;
        border-radius: 12px;

        padding: 15px 20px;
    }


    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] span {
        color: #2F333B !important;
    }


    /* --------------------------------------------------
       Text Input / API Key
       -------------------------------------------------- */

    [data-testid="stTextInput"] div[data-baseweb="input"] {
        background-color: #FFFFFF !important;

        border-color: #CDD1D8 !important;
        border-radius: 10px !important;
    }


    [data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;

        color: #262730 !important;
        -webkit-text-fill-color: #262730 !important;
    }


    [data-testid="stTextInput"] input::placeholder {
        color: #8A8D94 !important;
        opacity: 1 !important;
    }


    /* Passwort-Auge */
    [data-testid="stTextInput"] button {
        background-color: transparent !important;
        color: #5F6368 !important;
    }


    [data-testid="stTextInput"] svg {
        color: #5F6368 !important;
    }


    /* --------------------------------------------------
       Chat Nachrichten
       -------------------------------------------------- */

    [data-testid="stChatMessage"] {
        background-color: #F8F9FC !important;

        border: 1px solid #E2E5EB;
        border-radius: 14px;

        padding: 10px 15px;
        margin-bottom: 10px;
    }


    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: #2F333B;
    }


    /* --------------------------------------------------
       Chat Input unten
       -------------------------------------------------- */

    [data-testid="stBottom"] {
        background-color: #FFFFFF !important;
    }


    [data-testid="stBottomBlockContainer"] {
        background-color: #FFFFFF !important;
    }


    [data-testid="stChatInput"] {
        background-color: #F6F7FA !important;

        border: 1px solid #DDE1E8 !important;
        border-radius: 14px !important;
    }


    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;

        color: #262730 !important;
        -webkit-text-fill-color: #262730 !important;
    }


    [data-testid="stChatInput"] textarea::placeholder {
        color: #8A8D94 !important;
        opacity: 1 !important;
    }


    [data-testid="stChatInput"] button {
        color: #5749A6 !important;
    }


    /* --------------------------------------------------
       Expander
       -------------------------------------------------- */

    [data-testid="stExpander"] {
        background-color: #FAFBFC !important;

        border: 1px solid #E2E5EB;
        border-radius: 10px;
    }


    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #2F333B;
    }


    /* --------------------------------------------------
       Caption / Kennzahlen
       -------------------------------------------------- */

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #666A73 !important;
    }


    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.image(
    LOGO_PATH,
    width=320,
)

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

st.subheader("RAG-Architektur auswählen")

rag = st.radio(
    "RAG-Architektur",
    RAG_PATHS.keys(),
    horizontal=True,
    label_visibility="collapsed",
)


# --------------------------------------------------
# API-Key
# --------------------------------------------------

st.subheader("OpenAI API-Key")

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

st.session_state.setdefault(
    "messages",
    [],
)

st.session_state.setdefault(
    "last_rag",
    rag,
)


# Chat leeren, wenn RAG gewechselt wird
if st.session_state.last_rag != rag:

    st.session_state.messages = []

    st.session_state.last_rag = rag


# --------------------------------------------------
# Bisherigen Chat anzeigen
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if "metrics" in message:

            st.caption(
                message["metrics"]
            )


# --------------------------------------------------
# Eingabe
# --------------------------------------------------

query = st.chat_input(
    "Stellen Sie eine Frage..."
)


# --------------------------------------------------
# Frage verarbeiten
# --------------------------------------------------

if query:

    # API-Key prüfen
    if not api_key:

        st.error(
            "Bitte zuerst einen API-Key eingeben."
        )

        st.stop()


    # --------------------------------------------------
    # Nutzerfrage speichern
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )


    # Nutzerfrage anzeigen
    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------
    # Umgebung für RAG vorbereiten
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

        with st.spinner(
            "Antwort wird generiert..."
        ):

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

            st.code(
                process.stderr
            )

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

            st.code(
                process.stdout
            )

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


        st.caption(
            metrics_text
        )


        # --------------------------------------------------
        # Technische Ausgaben
        # --------------------------------------------------

        with st.expander(
            "Technische Ausgaben"
        ):

            st.text(logs)


    # --------------------------------------------------
    # Antwort im Verlauf speichern
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "metrics": metrics_text,
        }
    )
