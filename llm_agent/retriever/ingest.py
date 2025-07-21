import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

# Define output directory for vector store
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")

# List of IRSA URLs to ingest
urls = [
    "https://irsa.ipac.caltech.edu/frontpage/",
    "https://irsa.ipac.caltech.edu/Missions/missions.html",
    "https://irsa.ipac.caltech.edu/documentation.html",
    "https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/",
    "https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/",
    "https://irsa.ipac.caltech.edu/data/SPITZER/docs/seip/",
    "https://irsa.ipac.caltech.edu/data/SPITZER/docs/iglr/",
    "https://irsa.ipac.caltech.edu/data/WISE/docs/allsky/",
]

def ingest_irsa_docs():
    print("🌐 Loading IRSA webpages...")
    loader = WebBaseLoader(urls)
    docs = loader.load()

    print(f"📄 Loaded {len(docs)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    print("🔢 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    print(f"💾 Saving vector store to {VECTOR_DIR}")
    vectorstore.save_local(VECTOR_DIR)

if __name__ == "__main__":
    ingest_irsa_docs()
