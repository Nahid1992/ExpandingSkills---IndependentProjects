# MedRAG — Medical Literature Q&A with RAG

An independent learning project exploring Retrieval-Augmented Generation (RAG), 
LangChain, vector databases, FastAPI, and Docker — built to develop hands-on 
skills in modern AI/ML engineering and GenAI application development.

---

## What This Project Does

MedRAG is a domain-specific question-answering system that answers questions 
grounded in medical research papers. Instead of relying on an LLM's training 
data, it retrieves relevant passages from a local document collection and uses 
them as context for generation — ensuring answers are grounded in the actual 
source material.

---

## Why I Built This

My PhD research focused on building foundation models for medical image analysis. 
While that work gave me deep expertise in model architecture and large-scale 
pretraining, I wanted to develop practical skills in the modern AI/ML engineering 
stack — specifically RAG pipelines, LangChain, REST API development, and 
containerization with Docker. MedRAG is the first project in a series of 
independent builds toward that goal.

---

## Architecture
PDF Documents
↓
[ingest.py]
Load → Chunk → Embed → Save FAISS Index
↓
FAISS Vector Store (local)
↓
[app.py — FastAPI Server]
/query endpoint
↓
User Question → Embed → Retrieve Top-4 Chunks
↓
LangChain LCEL Chain
{context + question} → Prompt → LLM → Answer
↓
JSON Response

---

## Tech Stack

| Component | Technology |
|---|---|
| RAG Framework | LangChain (LCEL) |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM Backend | Ollama (llama3.2) — runs locally |
| API Layer | FastAPI |
| Containerization | Docker |

---

## Project Structure
MedRAG/
├── data/               # Source PDF documents
├── faiss_index/        # Generated vector index (after running ingest.py)
├── ingest.py           # Load, chunk, embed PDFs → build FAISS index
├── app.py              # FastAPI app with /query and /health endpoints
├── requirements.txt
└── Dockerfile

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker Desktop
- Ollama installed and running locally

### 1. Install Ollama and pull the model
```bash
brew install ollama
ollama pull llama3.2
ollama serve
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your documents
Place PDF files in the `data/` directory.

### 4. Build the vector index
```bash
python ingest.py
```

### 5. Run locally
```bash
uvicorn app:app --reload
```

### 6. Run with Docker
```bash
docker build -t medrag .
docker run -p 8000:8000 medrag
```

---

## Usage

**Query endpoint:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Lock-Release Pretraining?"}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

**Example response:**
```json
{
  "answer": "Lock-Release Pretraining is a strategy that alternates between 
  freezing and unfreezing model components during cyclic multi-dataset training, 
  designed to stabilize multi-task optimization and prevent task-specific 
  overfitting across heterogeneous annotation types."
}
```

---

## Skills Developed

- **RAG pipeline design** — ingestion, chunking, embedding, retrieval, generation
- **LangChain LCEL** — chaining retrievers, prompt templates, LLMs, output parsers
- **FAISS** — vector indexing and semantic similarity search
- **FastAPI** — REST API development with Pydantic request validation
- **Docker** — containerization, image builds, container networking

---

## Part of AI Portfolio

This project is part of an ongoing series of independent builds focused on 
developing industry AI/ML engineering skills:

| Project | Focus |
|---|---|
| MedRAG | RAG, LangChain, FAISS, FastAPI, Docker |
| JobScout *(coming soon)* | LangGraph, Agentic AI, Automated Pipelines |

---

## Author
**Nahid Ul Islam, Ph.D.**  
