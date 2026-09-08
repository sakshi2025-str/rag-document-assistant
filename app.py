"""
app.py
Streamlit chat interface for DocuChat.

Run with:
    streamlit run app.py
"""

import streamlit as st
from Qa import load_vector_store, get_answer

st.set_page_config(page_title="DocuChat", page_icon="📄")
st.title("📄 DocuChat")
st.caption("Ask questions about your uploaded documents. Answers are grounded only in what's in the document.")

# Load the vector store once and cache it across reruns
@st.cache_resource
def get_vector_store():
    return load_vector_store()

vector_store = get_vector_store()

# Keep chat history across turns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box
question = st.chat_input("Ask a question about your document...")

if question:
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(question, vector_store)
            st.write(answer)
            st.caption(f"Source page(s): {sources}")

    st.session_state.messages.append({"role": "assistant", "content": answer})