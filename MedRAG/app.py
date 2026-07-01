"""
app.py — FastAPI Server for MedRAG

Serves a REST API that answers questions grounded in indexed medical literature
using a Retrieval-Augmented Generation (RAG) pipeline.

At Startup:
    - Loads the FAISS vector index built by ingest.py
    - Initializes the HuggingFace embedding model for query encoding
    - Connects to the Ollama LLM backend (llama3.2)
    - Assembles the LangChain LCEL retrieval-generation chain

Query Pipeline (/query endpoint):
    1. Embed    — Encodes the incoming question into a dense vector
    2. Retrieve — Searches FAISS for the 4 most semantically similar chunks
    3. Prompt   — Injects retrieved chunks as context into the prompt template
    4. Generate — Sends the prompt to llama3.2 via Ollama and returns the answer

Endpoints:
    POST /query   — Accepts a JSON question, returns a grounded answer
    GET  /health  — Returns service status for health monitoring

Usage:

    Local (without Docker):
        # Terminal 1 — start Ollama
        ollama serve

        # Terminal 2 — start FastAPI server
        uvicorn app:app --reload

        # Terminal 3 — send a query
        curl -X POST http://localhost:8000/query \
             -H "Content-Type: application/json" \
             -d '{"question": "What is Lock-Release Pretraining?"}'

    With Docker:
        docker run -p 8000:8000 medrag

Note:
    Ensure ingest.py has been run and the faiss_index/ directory exists
    before starting the server. Ollama must be running and accessible
    at host.docker.internal:11434 when running inside Docker.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

## For Local Run
# llm = OllamaLLM(model="llama3.2", temperature=0)

## For Docker
llm = OllamaLLM(
    model="llama3.2",
    temperature=0,
    base_url="http://host.docker.internal:11434"
)


prompt = ChatPromptTemplate.from_template("""
You are a medical AI assistant. Answer the question based only on the following context from medical research papers.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class Query(BaseModel):
    question: str

@app.post("/query")
def query(q: Query):
    answer = qa_chain.invoke(q.question)
    return {"answer": answer}

@app.get("/health")
def health():
    return {"status": "ok"}
