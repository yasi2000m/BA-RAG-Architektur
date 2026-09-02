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

# Hochschule-Aalen-Bilder
LOGO_PATH = ROOT_DIR / "hs_aalen_logo.png"
ICON_PATH = ROOT_DIR / "hs_aalen_icon.png"


# --------------------------------------------------
# Streamlit-Oberfläche
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Chatbot | Hochschule Aalen",
    page_icon=str(ICON_PATH),
    layout="centered",
)


# Hochschule-Aalen-Logo
st.image(
    str(LOGO_PATH),
    width=300,
)

st.title("RAG Chatbot")

st.write(
    "Wählen Sie eine RAG-Architektur und stellen Sie eine Frage."
)


# --------------------------------------------------
# RAG auswählen
# --------------------------------------------------

rag = st.radio(
    "RAG-Architektur",
    RAG_PATHS.keys(),
    horizontal=True,
)


# --------------------------------------------------
# API-Key
# --------------------------------------------------

api_key = st.text_input(
    "API-Key",
    type="password",
    placeholder="API-Key eingeben",
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


    # Nutzerfrage anzeigen und speichern
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
            payload = json.loads(process.stdout)

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


        with st.expander("Technische Ausgaben"):
            st.text(logs)


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
                f"Query-Varianten: {result['num_query_variants']} · "
                f"Retrieval-Zeit: {result['retrieval_time']:.2f} s · "
                f"Tokenverbrauch: {result['used_tokens']}"
            )


        else:  # Graph RAG

            metrics_text = (
                f"Max Depth: {result['max_depth']} · "
                f"Entitäten: {result['entities']} · "
                f"Beziehungen: {result['relationships']} · "
                f"Tokenverbrauch: {result['used_tokens']}"
            )


        st.caption(metrics_text)


    # --------------------------------------------------
    # Antwort im Chatverlauf speichern
    # --------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "metrics": metrics_text,
    })
