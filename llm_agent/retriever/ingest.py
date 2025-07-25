import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY, USER_AGENT)
load_dotenv()
USER_AGENT = os.environ.get("USER_AGENT", "IRSA-RAG-Bot/1.0 (contact: shemmati@caltech.edu)")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# IRSA entry points
START_URLS = [
    "https://irsa.ipac.caltech.edu/frontpage/",
    "https://irsa.ipac.caltech.edu/documentation.html",
]

MAX_DEPTH = 5

def get_irsa_html_links(start_urls, max_depth=2):
    visited = set()
    to_visit = [(url, 0) for url in start_urls]
    html_links = set()
    headers = {"User-Agent": USER_AGENT}

    while to_visit:
        current_url, depth = to_visit.pop()
        if current_url in visited or depth > max_depth:
            continue
        visited.add(current_url)

        try:
            response = requests.get(current_url, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"⚠️ Failed to fetch {current_url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            full_url = urljoin(current_url, href)

            # Keep only IRSA HTML pages
            if "irsa.ipac.caltech.edu" not in full_url:
                continue
            if not full_url.endswith(".html"):
                continue
            if re.search(r"(mailto:|#|logout|login|javascript:)", full_url):
                continue

            if full_url not in visited:
                html_links.add(full_url)
                to_visit.append((full_url, depth + 1))

    return sorted(html_links)

def filter_existing_urls(urls):
    headers = {"User-Agent": USER_AGENT}
    good_urls = []
    print(f"Verifying {len(urls)} links...")

    for url in urls:
        clean_url = url.strip()
        if not clean_url.startswith("http"):
            continue

        try:
            # Follow redirects to catch 301s that land on working pages
            r = requests.head(clean_url, headers=headers, timeout=5, allow_redirects=True)
            if r.status_code == 200:
                good_urls.append(clean_url)
            else:
                print(f"❌ Skipping {clean_url}: Status {r.status_code}")
        except Exception as e:
            print(f"⚠️ Error checking {clean_url}: {e}")

    return good_urls

def ingest_irsa_docs():
    print("🌐 Crawling IRSA URLs...")
    irsa_links = get_irsa_html_links(START_URLS, max_depth=MAX_DEPTH)
    print(f"🔗 Found {len(irsa_links)} HTML links")

    irsa_links = filter_existing_urls(irsa_links)
    print(f"✅ {len(irsa_links)} links verified as live")

    print("Loading pages with WebBaseLoader...")
    loader = WebBaseLoader(irsa_links)
    docs = loader.load()

    print(f"📄 Loaded {len(docs)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    print("Generating embeddings...")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    model_kwargs={"device": "cuda"},  # Use GPU if available
    encode_kwargs={"normalize_embeddings": True})

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    print(f"Saving vector store to {VECTOR_DIR}")
    vectorstore.save_local(VECTOR_DIR)

if __name__ == "__main__":
    ingest_irsa_docs()
