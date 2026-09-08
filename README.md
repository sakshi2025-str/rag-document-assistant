# DocuChat

DocuChat is an AI-powered RAG (Retrieval-Augmented Generation) application that lets you upload a PDF document and ask natural-language questions about it. Answers are generated using only the content of your document, with the source page cited for every response.

## Features

- Upload a PDF document
- Extract and chunk text from the document
- Convert text chunks into embeddings and store them in a vector database
- Ask questions in a chat interface
- Get answers grounded in the actual document content
- View the source page number for every answer

## Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Groq API
- python-dotenv

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd DocuChat
```

### 2. Create and activate a virtual environment

```bash
conda create -n docuchat python=3.10
conda activate docuchat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root and add :
GROQ_API_KEY=your_groq_api_key_here

### 5. Add a document

Place a PDF file inside the `data/` folder.

### 6. Run ingestion

```bash
python ingest.py
```

### 7. Launch the app

```bash
streamlit run app.py
```

## Project Structure

## How It Works

1. The PDF is split into overlapping text chunks
2. Each chunk is converted into an embedding using Sentence Transformers
3. Embeddings are stored in ChromaDB, a local vector database
4. When a question is asked, the most relevant chunks are retrieved
5. The retrieved chunks are passed to an LLM (via Groq API), which generates an answer based only on that context

## Future Improvements

- Support for multiple documents
- Conversation memory for follow-up questions
- Retrieval accuracy evaluation
- Public deployment