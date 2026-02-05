# GenAI Knowledge Assistant

A minimal Retrieval-Augmented Generation (RAG) service built with Python and FastAPI.

## Features
- Semantic search using embeddings + FAISS
- REST API for Q&A
- Dockerized service
- Infrastructure defined using Terraform
- Cloud-ready (AWS / Azure)

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload


## ARCHITECTURE

User
 → FastAPI (Python)
 → Query Embedding
 → Vector Search (FAISS)
 → Context Retrieval
 → LLM (or Mock)
 → Response


## Project Structure 

genai-knowledge-assistant/
│
├── app/
│   ├── main.py
│   ├── rag.py
│   └── llm.py
│
├── data/
│   └── knowledge.txt
│
├── requirements.txt
├── Dockerfile
├── README.md
└── terraform/
    └── main.tf
