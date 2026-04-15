import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = "./faiss_index"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def ingest_documents(documents, rebuild=False):
    """Add documents to FAISS index. Set rebuild=True to start fresh."""
    embeddings = get_embeddings()
    os.makedirs(INDEX_PATH, exist_ok=True)

    if rebuild or not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        vectorstore = FAISS.from_documents(documents, embeddings)
    else:
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(documents)

    vectorstore.save_local(INDEX_PATH)
    print(f"✅ Ingested {len(documents)} chunks. Total in index: {vectorstore.index.ntotal}")