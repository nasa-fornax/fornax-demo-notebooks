import os
import gradio as gr
import requests

# 👇 For Jupyter notebooks; remove or comment this if not needed
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

API_URL = "http://localhost:8000/chat"  # Change this if deploying elsewhere

def chat_interface(message, history):
    try:
        response = requests.post(API_URL, json={"question": message})
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "No answer returned.")
        sources = data.get("sources", [])
        if sources:
            answer += "\n\n📚 Sources:\n" + "\n".join(f"🔗 {src}" for src in sources)
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}"

gr.ChatInterface(
    fn=chat_interface,
    chatbot=gr.Chatbot(label="IRSA Assistant"),
    textbox=gr.Textbox(placeholder="Ask IRSA anything...", container=False, scale=7),
    title="🔭 Ask IRSA",
    description="Ask questions about IRSA services, documentation, and tools.",
    theme="soft"
).launch(inline=True, share=True)
