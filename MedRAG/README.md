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

## Architecture
PDF Documents </br>
↓ </br>
[ingest.py] </br>
Load → Chunk → Embed → Save FAISS Index </br>
↓ </br>
FAISS Vector Store (local) </br>
↓ </br>
[app.py — FastAPI Server] </br>
/query endpoint </br>
↓ </br>
User Question → Embed → Retrieve Top-4 Chunks </br>
↓ </br>
LangChain LCEL Chain </br>
{context + question} → Prompt → LLM → Answer </br>
↓ </br>
JSON Response </br>
 
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
MedRAG/ </br>
├── data/               # Source PDF documents </br>
├── faiss_index/        # Generated vector index (after running ingest.py) </br>
├── ingest.py           # Load, chunk, embed PDFs → build FAISS index </br>
├── app.py              # FastAPI app with /query and /health endpoints </br>
├── requirements.txt </br>
└── Dockerfile </br>

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

## Tools Used

- **RAG pipeline design** — ingestion, chunking, embedding, retrieval, generation
- **LangChain LCEL** — chaining retrievers, prompt templates, LLMs, output parsers
- **FAISS** — vector indexing and semantic similarity search
- **FastAPI** — REST API development with Pydantic request validation
- **Docker** — containerization, image builds, container networking

---


## Author
**Nahid Ul Islam, Ph.D.**  
