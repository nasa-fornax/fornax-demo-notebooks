# IRSA LLM Agent (MVP)

This folder contains a prototype agent that uses a Large Language Model (LLM) to assist users with IRSA tools and missions.

### Goals:
- Answer questions about IRSA missions and services
- Point users to the right tools or documentation
- Suggest science projects based on available data (https://github.com/xoubish/project_idea_agent) to be transferred and connected somehow

### How to Run

The best way to train on IRSA documentation is to start from curated or existing local files, if available. However, for this MVP, instead of waiting for that, we crawl the IRSA website directly. Starting from key entry pages, we recursively follow links (up to 5 levels deep) and collect all .html pages. These pages (~926) are then vectorized using a retrieval-augmented generation (RAG) pipeline (see model details below), allowing them to be queried by an agent or LLM.  By default, the embedding model used is intfloat/e5-base-v2, a strong general-purpose model for dense retrieval. You can change this in ingest.py by modifying the model_name in the HuggingFaceEmbeddings call.

To run the ingestion pipeline:
python retriever/ingest.py

You can then run a simple LLM via the following two terminal commands:

Open 2 terminals, Terminal 1 – Start the API backend and Terminal 2 – Launch the Gradio interface:


```bash
uvicorn app:app --reload

python gradio_ui.py
