import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
torch.cuda.empty_cache()

# === 1. Load FAISS vector store ===
print("🔄 Loading vector store...")
VECTOR_DIR = "data/vector_store"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# === 2. Load FLAN-T5 model ===
print("🧠 Loading FLAN-ul2 model...")

tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2")

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    truncation=True  # Important to prevent token overflow
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# === 3. Build retrieval QA chain ===
prompt = PromptTemplate.from_template(
    "Answer the following question using the provided context.\n\nContext:\n{context}\n\nQuestion:\n{input}"
)

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain
)

# === 4. Ask Function ===
def ask_irsa(question):
    result = qa_chain.invoke({"input": question})
    answer = result["answer"]
    sources = [doc.metadata.get("source") for doc in result["context"] if "source" in doc.metadata]
    return answer, sources

# === 5. CLI Loop ===
if __name__ == "__main__":
    while True:
        try:
            q = input("❓ Ask IRSA something: ")
            if q.lower() in {"exit", "quit"}:
                break
            answer, sources = ask_irsa(q)
            print("🤖", answer)
            if sources:
                print("\n📚 Sources:")
                for src in set(sources):
                    print("  -", src)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
