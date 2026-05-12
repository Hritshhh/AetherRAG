<p align="center">

&#x20; <img src="assets/logo.png" width="140"/>

</p>



<h1 align="center">AetherRAG</h1>



<p align="center">

&#x20; <b>Aether — Your Local, Private \& Offline AI Assistant</b>

</p>



<p align="center">

&#x20; Retrieval-Augmented Generation (RAG) powered by FAISS, LangChain, Ollama and Mistral 7B.

</p>



\---



\# 🌌 About AetherRAG



AetherRAG is a fully local AI assistant that allows users to upload documents and interact with them conversationally using Retrieval-Augmented Generation (RAG).



Unlike cloud-based AI systems, Aether runs entirely on the user's machine:

\- 🔒 No external API calls

\- 📡 Offline capable

\- 🧠 Local LLM inference

\- 📄 Local document indexing

\- 🔐 Private semantic retrieval



The assistant combines:

\- \*\*Mistral 7B\*\* for language generation

\- \*\*FAISS\*\* for vector similarity search

\- \*\*BAAI embeddings\*\* for semantic understanding

\- \*\*LangChain\*\* for orchestration

\- \*\*Ollama\*\* for local LLM serving

\- \*\*Streamlit\*\* for the interactive UI



\---



\# ✨ Features



\- 🔒 Fully local \& private AI assistant

\- 📄 PDF / TXT document ingestion

\- 🧠 Semantic search using vector embeddings

\- ⚡ Fast FAISS vector retrieval

\- 🤖 Mistral 7B inference via Ollama

\- 🐳 Dockerized deployment

\- 💬 Chat-style Streamlit interface

\- 📊 Confidence scoring \& source highlighting

\- 📡 Offline-first architecture

\- ♻️ Incremental FAISS indexing

\- 🧵 Streaming token generation



\---



\# 🧠 Tech Stack



| Layer | Technology |

|---|---|

| Frontend | Streamlit |

| Application Logic | Python + LangChain |

| Embedding Model | BAAI/bge-small-en-v1.5 |

| Vector Database | FAISS |

| LLM Runtime | Ollama |

| Language Model | Mistral 7B Instruct |

| Containerization | Docker + Docker Compose |



\---



\# 🏗️ System Workflow



<p align="center">

&#x20; <img src="assets/system\_workflow.png" width="850"/>

</p>



\### Workflow Overview



1\. User uploads documents or asks a query

2\. Documents are chunked and converted into embeddings

3\. FAISS stores semantic vectors locally

4\. User query is embedded

5\. Top-K relevant chunks are retrieved

6\. Retrieved context is passed into Mistral 7B

7\. Aether streams a grounded response with sources \& confidence scoring



\---



\# 🐳 Docker Architecture



<p align="center">

&#x20; <img src="assets/docker\_architecture.png" width="950"/>

</p>



\## Why Docker?



Docker solves several major deployment problems:



\### ✅ Dependency Isolation

Avoids Python package conflicts across machines.



\### ✅ Cross-Platform Consistency

Runs identically on Windows, Linux and macOS.



\### ✅ Faster Setup

No manual environment configuration required.



\### ✅ Service Separation

\- Streamlit app runs independently

\- Ollama model server runs independently



\### ✅ Persistent Model Storage

GGUF models remain cached inside Docker volumes.



\### ✅ Clean Deployment

Anyone can clone and launch the project with minimal setup.



\---



\# 🔒 Local \& Offline Architecture



<p align="center">

&#x20; <img src="assets/offline\_architecture.png" width="500"/>

</p>



\## Privacy Advantages



\- ❌ No cloud APIs

\- ❌ No OpenAI dependency

\- ❌ No external document transfer

\- ✅ Local embeddings

\- ✅ Local vector search

\- ✅ Local LLM inference

\- ✅ User-controlled data



All uploaded documents remain entirely on the local machine.



\---



\# ⚙️ How Retrieval Works



\## 1. Document Ingestion



Uploaded documents are:

\- parsed

\- chunked

\- embedded using BGE embeddings



\---



\## 2. Vector Storage



Embeddings are stored locally inside:



```bash

faiss\_index/

```



using FAISS similarity indexing.



\---



\## 3. Query Embedding



The user query is converted into semantic vectors.



\---



\## 4. Retrieval



Top-K relevant chunks are retrieved using cosine similarity.



\---



\## 5. Generation



Retrieved context is injected into the prompt template and passed to Mistral 7B through Ollama.



\---



\# 📊 Confidence Scoring



AetherRAG includes confidence estimation based on semantic similarity between the user query and retrieved document chunks.



The score is normalized for intuitive readability while preserving retrieval quality.



\## Confidence Ranges



| Badge | Range | Meaning |

|---|---|---|

| 🟢 High Confidence | 75% – 100% | Strong semantic relevance |

| 🟡 Medium Confidence | 45% – 74% | Partial contextual relevance |

| 🔴 Low Confidence | Below 45% | Weak retrieval match |



Confidence scores help users quickly estimate how strongly the retrieved context supports the generated response.



\---



\# 📁 Project Structure



```bash

AetherRAG/

│

├── app.py

├── ingestion.py

├── utils.py

├── requirements.txt

├── Dockerfile

├── docker-compose.yml

│

├── faiss\_index/

│

├── assets/

│   ├── logo.png

│   ├── system\_workflow.png

│   ├── docker\_architecture.png

│   └── offline\_architecture.png

│

└── README.md

```



\---



\# 🚀 Installation



\## 1. Clone Repository



```bash

git clone https://github.com/Hritshhh/AetherRAG.git

cd AetherRAG

```



\---



\# 🖥️ Method 1 — Run Normally (Without Docker)



\## Install Dependencies



```bash

pip install -r requirements.txt

```



\---



\## Install Ollama



Download from:



https://ollama.com



\---



\## Pull Mistral Model



```bash

ollama pull mistral:7b-instruct-v0.3-q4\_K\_M

```



\---



\## Run Streamlit App



```bash

streamlit run app.py

```



\---



\# 🐳 Method 2 — Run Using Docker (Recommended)



\## Build \& Start Containers



```bash

docker-compose up --build

```



\---



\## Pull Mistral Model Inside Container



```bash

docker exec -it ollama ollama pull mistral:7b-instruct-v0.3-q4\_K\_M

```



\---



\## Access Application



Open:



```bash

http://localhost:8501

```



\---



\# 🛑 Stop Containers



```bash

docker-compose down

```



\---



\# 🧹 Clear Stored Embeddings



The application includes:

\- Delete Chat

\- Clear All Data



which remove:

\- session chat history

\- FAISS embeddings

\- indexed document vectors



\---



\# 📌 Future Improvements



\- OCR support

\- Hybrid retrieval

\- Metadata filtering

\- Multi-user support

\- GPU acceleration

\- Agentic workflows

\- Conversation memory



\---



\# 📚 References



\- LangChain

\- FAISS

\- Ollama

\- Streamlit

\- HuggingFace Embeddings

\- Mistral AI



Research Areas:

\- Retrieval-Augmented Generation (RAG)

\- Local LLM Inference

\- Semantic Search Systems

\- Vector Database Retrieval



\---



\# 👨‍💻 Author



<p align="center">

&#x20; <b>Hritaansh Mehra</b><br>

&#x20; Engineering Project — AetherRAG<br><br>

&#x20; 

&#x20; 🔗 <a href="https://github.com/Hritshhh">GitHub Profile</a>

</p>



\# ⭐ Aether



> “AI that is local, private and truly yours.”

