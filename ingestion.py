import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = "./faiss_index"

# ---------------- Embeddings (CACHED) ----------------
_embedding_model = None

def get_embeddings():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embedding_model


# ---------------- Ingestion ----------------
def ingest_documents(documents, rebuild=False):
    """
    Add documents to FAISS index.
    """

    if not documents:
        print("⚠️ No documents to ingest.")
        return

    embeddings = get_embeddings()
    os.makedirs(INDEX_PATH, exist_ok=True)

    index_file = os.path.join(INDEX_PATH, "index.faiss")

    try:
        # ---- Create new index ----
        if rebuild or not os.path.exists(index_file):
            vectorstore = FAISS.from_documents(documents, embeddings)

        # ---- Append ----
        else:
            vectorstore = FAISS.load_local(
                INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(documents)

        vectorstore.save_local(INDEX_PATH)

        print(f"✅ Added {len(documents)} chunks")
        print(f"📦 Total chunks: {vectorstore.index.ntotal}")

    except Exception as e:
        print(f"❌ Ingestion error: {e}")