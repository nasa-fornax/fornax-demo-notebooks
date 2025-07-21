import os
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings  # updated import

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vector store
doc_store_path = "data/vector_store"
print("🔄 Loading vector store...")
vectorstore = FAISS.load_local(doc_store_path, embeddings, allow_dangerous_deserialization=True)

# Set up retriever
retriever = vectorstore.as_retriever()

# Load local FLAN-T5 model via transformers
print("🧠 Loading local FLAN-T5 model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define prompt template
prompt_template = PromptTemplate.from_template(
    "Answer the following question using the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}"
)

# Assemble the chain manually to support LangChain >=0.1.0
combine_docs_chain = StuffDocumentsChain(
    llm_chain=LLMChain(llm=llm, prompt=prompt_template),
    document_variable_name="context",
)

qa_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=combine_docs_chain,
    return_source_documents=False,
)

# Function to ask a question
def ask_irsa(question):
    result = qa_chain.invoke({"query": question})
    return result["result"]

# CLI loop
if __name__ == "__main__":
    while True:
        try:
            q = input("❓ Ask IRSA something: ")
            if q.lower() in {"exit", "quit"}:
                break
            print("🤖", ask_irsa(q))
        except KeyboardInterrupt:
            break
