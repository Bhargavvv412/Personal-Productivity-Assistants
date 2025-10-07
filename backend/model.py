import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

# Get keys and paths
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")

# Check API key
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found in .env")

# Load embeddings and vector DB
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GEMINI_API_KEY
)

vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)
retriever = vector_db.as_retriever()

# LLM for answering
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GEMINI_API_KEY
)

# QA Chain
qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

# Query handler
def get_response(query: str):
    docs = retriever.get_relevant_documents(query)
    return qa_chain.run(input_documents=docs, question=query)
