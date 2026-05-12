<p align="center">
  <img src="assets/logo.png" width="140"/>
</p>

<h1 align="center">AetherRAG</h1>

<p align="center">
  <b>Aether — Your Local, Private & Offline AI Assistant</b>
</p>

<p align="center">
  Retrieval-Augmented Generation (RAG) powered by FAISS, LangChain, Ollama and Mistral 7B.
</p>

---

## 👨‍💻 Author

### Hritaansh Mehra  
GitHub: [Hritshhh](https://github.com/Hritshhh)

---

# 📖 About AetherRAG

AetherRAG is a fully local AI assistant that allows users to upload documents and interact with them conversationally using Retrieval-Augmented Generation (RAG).

Unlike cloud-based AI systems, Aether runs entirely on the user's machine:

- 🔒 No external API calls
- 📡 Offline capable
- 🧠 Local LLM inference
- 📄 Local document indexing
- 🔐 Private semantic retrieval

The assistant combines:

- **Mistral 7B** for language generation
- **FAISS** for vector similarity search
- **BAAI embeddings** for semantic understanding
- **LangChain** for orchestration
- **Ollama** for local LLM serving
- **Streamlit** for the interactive UI

---

# ✨ Features

- 🔒 Fully local & private AI assistant
- 📄 PDF / TXT document ingestion
- 🧠 Semantic search using vector embeddings
- ⚡ Fast FAISS vector retrieval
- 🤖 Mistral 7B inference via Ollama
- 🐳 Dockerized deployment
- 💬 Chat-style Streamlit interface
- 📊 Confidence scoring & source highlighting
- 📡 Offline-first architecture
- ♻️ Incremental FAISS indexing
- 🧵 Streaming token generation
- 🧠 Context-aware greetings & acknowledgements

---

# 🧠 Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Streamlit |
| Application Logic | Python + LangChain |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | FAISS |
| LLM Runtime | Ollama |
| Language Model | Mistral 7B Instruct |
| Containerization | Docker + Docker Compose |

---

# 🏗️ System Workflow

<p align="center">
  <img src="assets/system_workflow.png" width="850"/>
</p>

The workflow begins with document upload and user queries through the Streamlit interface. Documents are embedded locally using BAAI embeddings and stored inside a FAISS vector database. Queries are embedded similarly and semantically matched against stored chunks before context is sent to the local Mistral 7B model through Ollama.

---

# 🐳 Docker Architecture

<p align="center">
  <img src="assets/docker_architecture.png" width="900"/>
</p>

Docker isolates all dependencies and services into reproducible containers. This solves:

- ✅ Dependency conflicts
- ✅ Environment inconsistencies
- ✅ OS compatibility issues
- ✅ Easier deployment
- ✅ Portable local AI setup
- ✅ Faster reproducible setup for other users

The architecture separates:

- Streamlit application container
- Ollama inference container
- Persistent FAISS vector storage
- Internal Docker networking

---

# 🔒 Offline & Private RAG Pipeline

<p align="center">
  <img src="assets/offline_pipeline.png" width="500"/>
</p>

AetherRAG is designed with an offline-first philosophy:

- No cloud APIs
- No external data transfer
- All embeddings generated locally
- All retrieval performed locally
- All inference performed locally

This makes the system highly suitable for:

- Academic usage
- Sensitive documents
- Research workflows
- Air-gapped environments
- Privacy-focused AI applications

---

# ⚙️ Confidence Scoring

AetherRAG computes semantic relevance using FAISS similarity search.

The UI displays confidence badges based on retrieval strength:

| Confidence Range | Badge |
| --- | --- |
| High relevance | 🟢 High Confidence |
| Moderate relevance | 🟡 Medium Confidence |
| Weak relevance | 🔴 Low Confidence |

The scoring is derived from semantic similarity between:

- User query embeddings
- Retrieved document chunk embeddings

This helps users understand retrieval reliability during inference.

---

# 📂 Supported Documents

Currently supported:

- PDF (`.pdf`)
- Text (`.txt`)

Documents are:

1. Loaded locally
2. Chunked into semantic segments
3. Embedded using BAAI embeddings
4. Stored inside FAISS index
5. Retrieved during querying

---

# 🚀 Running the Project

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Hritshhh/AetherRAG.git
cd AetherRAG
```

---

# ▶️ Method 1 — Run Normally (Streamlit)

## Install Requirements

```bash
pip install -r requirements.txt
```

## Install Ollama

Download Ollama from:

https://ollama.com

---

## Pull Mistral Model

```bash
ollama pull mistral:7b-instruct-v0.3-q4_K_M
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 🐳 Method 2 — Run Using Docker

## Start Docker Desktop

Ensure Docker Desktop is running.

---

## Build & Start Containers

```bash
docker compose up --build
```

---

## Pull Mistral Model Inside Container

```bash
docker exec -it ollama ollama pull mistral:7b-instruct-v0.3-q4_K_M
```

---

## Access Application

Open in browser:

```bash
http://localhost:8501
```

---

# 📁 Project Structure

```bash
AetherRAG/
│
├── app.py
├── ingestion.py
├── utils.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
│
├── faiss_index/
├── data/
│
└── assets/
    ├── logo.png
    ├── system_workflow.png
    ├── docker_architecture.png
    └── offline_pipeline.png
```

---

# 🎯 Key Highlights

- Fully local RAG assistant
- Runs without internet
- Semantic retrieval using FAISS
- Local Mistral 7B inference
- Modern chat interface
- Incremental vector indexing
- Dockerized deployment
- Streamed token generation
- Confidence-aware retrieval

---

# 📚 References

- LangChain  
https://www.langchain.com/

- Ollama  
https://ollama.com/

- FAISS  
https://github.com/facebookresearch/faiss

- Mistral AI  
https://mistral.ai/

- BAAI Embeddings  
https://huggingface.co/BAAI/bge-small-en-v1.5

---

# 📜 License

This project is intended for academic and educational purposes.

---

# ⭐ AetherRAG

> Private. Offline. Local. Yours.
