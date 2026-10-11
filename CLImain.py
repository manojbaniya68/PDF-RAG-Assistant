from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

templates = ChatPromptTemplate.from_messages(
    [
        ("system","""You are an helpful AI assistance.
          Use only the context provided to answer the question.
            If answer is not found in the document then say, 
            "I could't find the answer to the question"
        """),
        ("human","""
            context:{context},
            question:{query}
""")
    ]
)

retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs = {
            "k":3,
            "lambda_mult":0.5
        }
    )

llm = ChatGoogleGenerativeAI(model = "gemini-3.8-flash")

print("RAG system assistance")

print("Enter 0 to exit..")

while True:
    query = input("You: ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n".join(
        [doc.page_content for doc in docs]
    )
    prompt = templates.invoke({
        "context" : context,
        "query" : query
    })

    response = llm.invoke(prompt)
    print(response.content[0]["text"])
    




