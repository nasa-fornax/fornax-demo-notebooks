# IRSA LLM Agent (MVP)

This repository contains a prototype agent that uses a Large Language Model (LLM) to assist users with IRSA tools and missions.

---

### Goals
- Answer questions about IRSA missions and services
- Point users to relevant tools or documentation
- Suggest science projects based on available data  
  _(Integration planned with [project_idea_agent](https://github.com/xoubish/project_idea_agent))_

---

### How It Works

The ideal training approach would use curated IRSA documentation files. However, for this MVP, we crawl the IRSA website starting from key entry points. The crawler recursively follows links (up to 5 levels deep) and collects all `.html` pages (~4526 in total).  

These pages are processed using a **retrieval-augmented generation (RAG)** pipeline:
- Chunks are embedded using `intfloat/e5-base-v2`, a high-performing dense retrieval model.
- Embeddings are stored in a FAISS vector store.
- At query time, relevant documents are retrieved and passed to an LLM (default: `flan-t5-base`).

You can modify the embedding model by changing the `model_name` in `retriever/ingest.py`.

---

### System Flow

```text
[User] → (Gradio UI) → [chat_interface()] → POST → (FastAPI /chat endpoint)
                                                ↓
                                [ask_irsa()] → FAISS + LLM → response
                                                ↑
                      JSON ←---------------------← Answer + Sources
                          ↑
             Displayed in Gradio chat bubble



### How to Run

#### Step 1: Ingest the IRSA documentation

python retriever/ingest.py

#### Step 2: Start the backend API (Terminal 1)

uvicorn app:app --reload

#### Step 3: Launch the Gradio UI (Terminal 2)

python gradio_ui.py
