from fastapi import FastAPI, Request
from pydantic import BaseModel
from qa_tool_module import ask_irsa
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Optional: allow requests from Gradio or localhost for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    answer, sources = ask_irsa(query.question)
    return {"answer": answer, "sources": sources}
