"""
app.py

Streamlit interface for DocuChat.

Run:
    streamlit run app.py
"""

import os

import streamlit as st

from Qa import load_vector_store, get_answer


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="DocuChat",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Header
# -----------------------------

st.title("📄 DocuChat")

st.markdown(
    """
    **Chat with your documents using Retrieval-Augmented Generation (RAG).**

    Upload a PDF, ask questions, and get answers grounded in the document.
    """
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("📚 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Existing Vector Store
# -----------------------------

try:

    vector_store = load_vector_store()

except FileNotFoundError:

    vector_store = None

    st.info(
        "📄 No document database found. "
        "Please process your PDF using ingest.py first."
    )


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("📚 View Sources"):

                    for source in sources:

                        st.markdown(
                            f"**{source['file']} — Page "
                            f"{source['page']}**"
                        )

                        st.caption(
                            source["content"][:400] + "..."
                        )


# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input(
    "Ask a question about your document..."
)


if question:

    # Display user message

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.write(question)


    # Check vector store

    if vector_store is None:

        with st.chat_message("assistant"):

            st.error(
                "No document database is available. "
                "Please process a PDF first."
            )

    else:

        # Generate answer

        with st.chat_message("assistant"):

            with st.spinner("🔎 Searching document..."):

                try:

                    answer, sources = get_answer(
                        question,
                        vector_store
                    )

                    st.write(answer)

                    # Display sources

                    if sources:

                        with st.expander("📚 View Sources"):

                            for source in sources:

                                st.markdown(
                                    f"**{source['file']} — "
                                    f"Page {source['page']}**"
                                )

                                st.caption(
                                    source["content"][:400]
                                    + "..."
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(
                        "Something went wrong while generating "
                        "the answer."
                    )

                    st.caption(str(e))