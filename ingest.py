"""
ingest.py
Loads a PDF, splits it into chunks, creates embeddings,
and stores them in a local ChromaDB vector store.

Run this once (or whenever you add new documents):
    python ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---- Settings ----
DATA_FOLDER = "data"
CHROMA_DB_FOLDER = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_documents():
    """Load all PDFs found in the data folder."""
    all_docs = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DATA_FOLDER, filename)
            print(f"Loading: {filename}")
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

def split_documents(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks):
    """Embed chunks and store them in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_FOLDER
    )
    print(f"Stored embeddings in '{CHROMA_DB_FOLDER}/'")
    return vector_store

if __name__ == "__main__":
    print("Starting ingestion...")
    documents = load_documents()

    if not documents:
        print("No PDFs found in the 'data' folder. Add a PDF and try again.")
    else:
        chunks = split_documents(documents)
        build_vector_store(chunks)
        print("Ingestion complete! Your documents are ready to be queried.")