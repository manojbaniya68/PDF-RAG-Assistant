
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from create_database import create_vector_database


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma-db"
print("db directory is : ", DB_DIR)

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Cached models
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings()


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )


@st.cache_resource
def load_vector_store():
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=load_embedding_model(),
        collection_name="pdf_documents"
    )


# --------------------------------------------------
# Prompt template
# --------------------------------------------------

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful PDF question-answering assistant.

Answer the user's question using only the provided context.
Do not use outside knowledge to fill in missing information.

If the answer is not present in the context, respond exactly:
"I couldn't find the answer in the document."

Be clear and concise."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}"""
    )
])


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None


# --------------------------------------------------
# Sidebar: upload and create database
# --------------------------------------------------

with st.sidebar:
    st.title("📚 PDF Manager")

    st.write("Upload a PDF to create your searchable database.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if st.button(
        "Create Vector Database",
        type="primary",
        use_container_width=True
    ):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")

        elif not os.getenv("GOOGLE_API_KEY"):
            st.error(
                "GOOGLE_API_KEY is missing. Add it to your .env file."
            )

        else:
            temp_path = None

            try:
                with st.spinner(
                    "Loading PDF, splitting text and generating embeddings..."
                ):
                    # PyPDFLoader requires a file path.
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    result = create_vector_database(temp_path)

                    st.session_state.active_pdf = uploaded_file.name
                    st.session_state.messages = []

                    # Clear any cached reference to the old database.
                    load_vector_store.clear()

                st.success("Vector database created successfully!")
                st.write(f"**Pages:** {result[0]}")
                st.write(f"**Chunks:** {result[1]}")

            except Exception as error:
                st.error(f"Database creation failed: {error}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    st.divider()

    if DB_DIR.exists():
        st.success("ChromaDB folder found.")

        if st.session_state.active_pdf:
            st.caption(
                f"Active PDF: {st.session_state.active_pdf}"
            )
        else:
            st.caption(
                "An existing database was found. "
                "Upload a PDF and create a database to replace it."
            )
    else:
        st.info("Upload a PDF and create your first database.")


# --------------------------------------------------
# Main chat interface
# --------------------------------------------------

st.title("🤖 PDF RAG Assistant")
st.caption("Ask questions about your PDF using retrieval-augmented generation.")

if not os.getenv("GOOGLE_API_KEY"):
    st.warning(
        "Set GOOGLE_API_KEY in your .env file before asking questions."
    )

# Render previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


database_exists = (
    DB_DIR.exists()
    and any(DB_DIR.iterdir())
)

if not database_exists:
    st.info(
        "No vector database found. Upload a PDF from the sidebar "
        "and click 'Create Vector Database' to get started."
    )

elif question := st.chat_input("Ask a question about your PDF..."):

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Please configure GOOGLE_API_KEY in your .env file.")

    else:
        # Display and save the user's message.
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the PDF and generating an answer..."):

                    # 1. Load the vector database
                    vector_store = load_vector_store()

                    # 2. Retrieve relevant chunks
                    retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 3,
                            "lambda_mult": 0.5
                        }
                    )

                    documents = retriever.invoke(question)
                    print(documents)

                    if not documents:
                        answer = (
                            "I couldn't find the answer in the document."
                        )

                    else:
                        # 3. Combine retrieved text into context
                        context = "\n\n".join(
                            doc.page_content for doc in documents
                        )

                        # 4. Build the prompt
                        prompt = prompt_template.invoke({
                            "context": context,
                            "question": question
                        })

                        # 5. Ask Gemini
                        response = load_llm().invoke(prompt)
                        answer = response.content

                        # Handle text-block responses if returned as a list.
                        if isinstance(answer, list):
                            answer = "\n".join(
                                block.get("text", "")
                                for block in answer
                                if isinstance(block, dict)
                                and block.get("type") == "text"
                            )

                        if not isinstance(answer, str):
                            answer = str(answer)

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:
                st.error(f"Something went wrong: {error}")