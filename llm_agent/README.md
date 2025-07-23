# IRSA LLM Agent (MVP)

This folder contains a prototype agent that uses a Large Language Model (LLM) to assist users with IRSA tools and missions.

### Goals:
- Answer questions about IRSA missions and services
- Point users to the right tools or documentation
- Suggest science projects based on available data

### How to Run

You can run the assistant via the following two terminal commands:

Open 2 terminals, Terminal 1 – Start the API backend and Terminal 2 – Launch the Gradio interface:


```bash
uvicorn app:app --reload

python gradio_ui.py
