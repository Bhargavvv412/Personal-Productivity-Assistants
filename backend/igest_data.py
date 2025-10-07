import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH","db")

if not GEMINI_API_KEY:
    raise ValueError("api key not found")

#load documents
documents = [
    Document(page_content="Meeting notes: Discuss project X deliverables"),
    Document(page_content="Reminder: Submit report by Friday"),
    Document(page_content="Upcoming event: Tech conference next Wensday")
]

# create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)

#split text for better retrieval
text_splitter = CharacterTextSplitter(chunk_size = 200,chunk_overlap=50)
docs = text_splitter.split_documents(documents)

#store embeddings in ChromaDB
vector_db = Chroma.from_documents(docs,embedding=embeddings,persist_directory=CHROMA_DB_PATH)
print("document successfully indexed!")