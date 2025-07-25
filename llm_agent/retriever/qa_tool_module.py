import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
torch.cuda.empty_cache()

print("🔄 Loading vector store...")
VECTOR_DIR = "../data/vector_store"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

print("🧠 Loading FLAN-ul2 model...")
model_id = "google/flan-t5-base"  # or "google/flan-t5-base"
device = 0 if torch.cuda.is_available() else -1  
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=256,
    truncation=True
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Chain setup
prompt = PromptTemplate.from_template(
    "Answer the following question using the provided context.\n\nContext:\n{context}\n\nQuestion:\n{input}"
)
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

# Function for FastAPI/Gradio
def ask_irsa(question: str):
    result = qa_chain.invoke({"input": question})
    answer = result["answer"]
    sources = [doc.metadata.get("source") for doc in result["context"] if "source" in doc.metadata]
    return answer, sources
