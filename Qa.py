"""
qa.py

Retrieves relevant document chunks from ChromaDB
and uses Groq LLM to generate grounded answers.
"""

import os

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


CHROMA_DB_FOLDER = "chroma_db"

TOP_K = 4


PROMPT_TEMPLATE = """
You are DocuChat, a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not present in the context, say:
   "I don't have enough information in the uploaded documents to answer that."
4. Give a concise and clear answer.
5. Use bullet points when appropriate.

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# Embedding Model
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# -----------------------------
# Load Vector Store
# -----------------------------

def load_vector_store():

    if not os.path.exists(CHROMA_DB_FOLDER):
        raise FileNotFoundError(
            "Vector database not found. Run ingest.py first."
        )

    vector_store = Chroma(
        persist_directory=CHROMA_DB_FOLDER,
        embedding_function=embeddings
    )

    return vector_store


# -----------------------------
# Get Answer
# -----------------------------

def get_answer(question, vector_store):

    # Retrieve documents with similarity scores
    results = vector_store.similarity_search_with_score(
        question,
        k=TOP_K
    )

    if not results:
        return (
            "I don't have enough information in the uploaded documents "
            "to answer that.",
            []
        )

    # Prepare context
    documents = [doc for doc, score in results]

    context_parts = []

    sources = []

    for doc in documents:

        filename = doc.metadata.get(
            "source_file",
            doc.metadata.get("source", "Unknown")
        )

        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"Source: {filename}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

        sources.append({
            "file": filename,
            "page": page,
            "content": doc.page_content
        })

    context = "\n\n---\n\n".join(context_parts)

    # Create prompt
    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    # Generate answer
    response = llm.invoke(formatted_prompt)

    return response.content, sources


# -----------------------------
# Terminal Testing
# -----------------------------

if __name__ == "__main__":

    vector_store = load_vector_store()

    print("DocuChat is ready!")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask a question: ")

        if question.lower() == "exit":
            break

        answer, sources = get_answer(
            question,
            vector_store
        )

        print("\nAnswer:")
        print(answer)

        print("\nSources:")

        for source in sources:
            print(
                f"- {source['file']} "
                f"(Page {source['page']})"
            )

        print()