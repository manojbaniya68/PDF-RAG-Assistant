#load pdf
#make chunks
# generate embedding
# add embeddings to vector database(chorma-db)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

#laoddig the pdf
def create_vector_database(pdf_path):
    # data = PyPDFLoader("document_loader/EmergingTrends.pdf")
    data = PyPDFLoader(pdf_path)
    docs = data.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embedding_modal = HuggingFaceEmbeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_modal,
        persist_directory="chroma-db",
        collection_name="pdf_documents"
    )

    return len(docs), len(chunks)



