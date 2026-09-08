"""
qa.py
Loads the existing ChromaDB vector store, retrieves the most relevant
chunks for a question, and asks Groq's LLM to answer using only that context.

Run this to test in the terminal:
    python qa.py
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()  # reads GROQ_API_KEY from .env

CHROMA_DB_FOLDER = "chroma_db"
TOP_K = 3  # how many chunks to retrieve per question

PROMPT_TEMPLATE = """
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up information that isn't in the context.

Context:
{context}

Question:
{question}

Answer:
"""

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_FOLDER,
        embedding_function=embeddings
    )
    return vector_store

def get_answer(question, vector_store):
    # Step 1: retrieve top-k relevant chunks
    results = vector_store.similarity_search(question, k=TOP_K)

    if not results:
        return "I don't have enough information to answer that.", []

    # Step 2: build context string from retrieved chunks
    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Step 3: build the prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context, question=question)

    # Step 4: send to Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )
    response = llm.invoke(formatted_prompt)

    # Step 5: return answer + sources (page numbers) for citation
    sources = [doc.metadata.get("page", "unknown") for doc in results]
    return response.content, sources

if __name__ == "__main__":
    vector_store = load_vector_store()
    print("DocuChat is ready. Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break

        answer, sources = get_answer(question, vector_store)
        print(f"\nAnswer: {answer}")
        print(f"Sources: page(s) {sources}\n")