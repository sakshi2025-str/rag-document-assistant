"""
ingest.py

Loads PDFs from the data folder, splits them into chunks,
creates embeddings, and stores them in ChromaDB.

Run:
    python ingest.py
"""

import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DATA_FOLDER = "data"
CHROMA_DB_FOLDER = "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def load_documents():
    """Load all PDF documents from the data folder."""

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    documents = []

    pdf_files = [
        file for file in os.listdir(DATA_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        return documents

    for filename in pdf_files:

        path = os.path.join(DATA_FOLDER, filename)

        print(f"Loading: {filename}")

        loader = PyPDFLoader(path)
        docs = loader.load()

        # Add filename metadata
        for doc in docs:
            doc.metadata["source_file"] = filename

        documents.extend(docs)

    print(f"Loaded {len(documents)} pages.")

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Add chunk IDs
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(f"Created {len(chunks)} chunks.")

    return chunks


def build_vector_store(chunks):
    """Create embeddings and store documents in ChromaDB."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Remove old database during a fresh ingestion
    if os.path.exists(CHROMA_DB_FOLDER):
        shutil.rmtree(CHROMA_DB_FOLDER)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_FOLDER
    )

    print(f"Vector database created at: {CHROMA_DB_FOLDER}")

    return vector_store


if __name__ == "__main__":

    print("\nStarting DocuChat ingestion...\n")

    documents = load_documents()

    if not documents:
        print("No PDF files found in the data folder.")
        print("Please add a PDF and run the script again.")

    else:
        chunks = split_documents(documents)

        build_vector_store(chunks)

        print("\nIngestion completed successfully!")
        print("Your documents are ready for questions.")