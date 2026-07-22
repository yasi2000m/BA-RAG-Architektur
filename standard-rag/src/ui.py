import streamlit as st

from chunking import chunk_text
from document_loader import load_pdf_with_visuals
from embeddings import EmbeddingModel
from generation import AnswerGenerator
from retrieval import retrieve_relevant_chunks
from vector_store import LocalVectorStore


DEFAULT_PDF_PATH = "data/elektrotechnik_3.pdf"
VECTOR_DB_PATH = "vector_db_elektrotechnik_3_full_visuals"


def build_vector_store(pdf_path: str) -> tuple[LocalVectorStore, EmbeddingModel, int]:
    """Laedt das PDF, erstellt Chunks, Embeddings und eine lokale Vektordatenbank."""
    document_text = load_pdf_with_visuals(pdf_path)

    if not document_text:
        raise ValueError("Im PDF wurden keine auswertbaren Inhalte gefunden.")

    chunks = chunk_text(document_text, chunk_size=250, overlap=50)
    embedding_model = EmbeddingModel()
    chunk_embeddings = embedding_model.embed_texts(chunks)

    vector_store = LocalVectorStore(VECTOR_DB_PATH)
    vector_store.add(chunks, chunk_embeddings)
    vector_store.save()

    return vector_store, embedding_model, len(chunks)


def load_or_build_vector_store() -> tuple[LocalVectorStore, EmbeddingModel, int, bool]:
    """Laedt eine vorhandene Wissensbasis oder baut sie einmalig aus der integrierten PDF auf."""
    embedding_model = EmbeddingModel()
    vector_store = LocalVectorStore(VECTOR_DB_PATH)

    try:
        vector_store.load()
        return vector_store, embedding_model, len(vector_store.chunks), False
    except FileNotFoundError:
        vector_store, embedding_model, chunk_count = build_vector_store(DEFAULT_PDF_PATH)
        return vector_store, embedding_model, chunk_count, True


st.set_page_config(page_title="Standard-RAG", page_icon="PDF", layout="centered")

st.title("Standard-RAG fuer Elektrotechnik")
st.write("Die Elektrotechnik-PDF ist integriert. Alle Seiten werden inklusive Bilder, Tabellen, Formeln und Diagramme visuell gelesen.")

if "vector_store" not in st.session_state or st.session_state.get("vector_db_path") != VECTOR_DB_PATH:
    with st.spinner("Wissensbasis wird geladen ..."):
        try:
            vector_store, embedding_model, chunk_count, was_built = load_or_build_vector_store()
            st.session_state.vector_store = vector_store
            st.session_state.embedding_model = embedding_model
            st.session_state.chunk_count = chunk_count
            st.session_state.was_built = was_built
            st.session_state.vector_db_path = VECTOR_DB_PATH
        except Exception as error:
            st.session_state.vector_store = None
            st.session_state.embedding_model = None
            st.error(f"Fehler beim Laden der Wissensbasis: {error}")

if st.session_state.vector_store is not None:
    st.success(f"Wissensbasis bereit. Chunks: {st.session_state.chunk_count}")
    st.caption(f"PDF: {DEFAULT_PDF_PATH} | Vektordatenbank: {VECTOR_DB_PATH}")

st.divider()

question = st.text_input("Deine Frage")

if st.button("Antwort generieren", disabled=not question):
    if st.session_state.vector_store is None or st.session_state.embedding_model is None:
        st.warning("Die Wissensbasis konnte nicht geladen werden.")
    else:
        with st.spinner("Relevante Chunks werden gesucht und die Antwort wird generiert ..."):
            relevant_chunks = retrieve_relevant_chunks(
                query=question,
                embedding_model=st.session_state.embedding_model,
                vector_store=st.session_state.vector_store,
                top_k=5,
            )

            generator = AnswerGenerator()
            answer = generator.generate_answer(question, relevant_chunks)

        st.subheader("Antwort")
        st.write(answer)

        with st.expander("Verwendete Chunks anzeigen"):
            for index, chunk in enumerate(relevant_chunks, start=1):
                st.markdown(f"**Chunk {index}**")
                st.write(chunk)
