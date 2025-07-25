# Base Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Run ingestion script at build time to create the vector store
RUN python retriever/ingest.py

# Expose FastAPI port and Gradio port
EXPOSE 8000 7860

# Set working directory to where app.py and gradio_ui.py live
WORKDIR /app/retriever

# Default command to run FastAPI and Gradio together
CMD bash -c "uvicorn app:app --host 0.0.0.0 --port 8000 & \
python gradio_ui.py & \
sleep 2 && echo -e '\n🚀 IRSA Agent is running! Access the chat UI at: http://localhost:7860\n' && wait"
