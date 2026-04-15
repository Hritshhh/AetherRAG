import os
import tempfile
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents_from_files(uploaded_files) -> List:
    """Load and chunk documents from various file types."""
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    for uploaded_file in uploaded_files:
        # Save to temp file because loaders expect a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        # Choose loader based on extension
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        try:
            if ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif ext == ".csv":
                loader = CSVLoader(tmp_path)
            elif ext == ".json":
                loader = JSONLoader(tmp_path, jq_schema=".", text_content=False)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(tmp_path)
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(tmp_path)
            elif ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                print(f"Unsupported file type: {ext}")
                os.unlink(tmp_path)
                continue

            docs = loader.load()
            # Add source filename metadata
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            # Split into chunks
            split_docs = text_splitter.split_documents(docs)
            all_docs.extend(split_docs)

        except Exception as e:
            print(f"Error loading {uploaded_file.name}: {e}")
        finally:
            os.unlink(tmp_path)

    return all_docs