# PDF RAG Assistant

A simple PDF question-answering application built with Streamlit, LangChain, ChromaDB, Hugging Face embeddings, and Google Gemini.

Upload a PDF, create a vector database, and ask questions about the document. The application retrieves relevant text and uses it to generate answers with Gemini.

## Features

* Upload PDF documents.
* Extract and split text into chunks.
* Generate embeddings using Hugging Face.
* Store and retrieve document embeddings using ChromaDB.
* Ask questions through a Streamlit chat interface.
* Generate context-based answers using Google Gemini.

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Hugging Face Sentence Transformers
* Google Gemini

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your `.env` file**

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

**4. Run the application**

```bash
streamlit run app.py
```

## How It Works

1. Upload a PDF.
2. Extract text and split it into chunks.
3. Generate embeddings and store them in ChromaDB.
4. Ask a question about the document.
5. Retrieve relevant chunks and generate an answer using Gemini.

