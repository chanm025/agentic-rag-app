"""Local PDF knowledge base: load PDFs -> chunk -> embed -> FAISS index -> similarity search.

Same pipeline as the notebook (analysis3.ipynb): PyPDFDirectoryLoader, 1000-character chunks with
200 overlap, sentence-transformers/all-MiniLM-L6-v2 embeddings, FAISS vector store.
"""
import os
from functools import lru_cache
from pathlib import Path

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def find_knowledge_dir():
    """First folder containing PDFs: $KNOWLEDGE_DIR, then ./data, then ./crew_data."""
    for folder in (os.getenv("KNOWLEDGE_DIR"), "data", "crew_data"):
        if folder and Path(folder).is_dir() and any(Path(folder).glob("*.pdf")):
            return folder
    return None


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings   # local model, no API quota needed
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore():
    """Build the FAISS index once per process (None if there are no PDFs)."""
    folder = find_knowledge_dir()
    if folder is None:
        return None
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = PyPDFDirectoryLoader(folder).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    return FAISS.from_documents(chunks, get_embeddings())


def search_pdfs(query: str, k: int = 4) -> str:
    """Return the k most relevant chunks as text, each labelled with its source file and page."""
    store = get_vectorstore()
    if store is None:
        return "The PDF knowledge base is empty: add .pdf files to the ./data folder and restart."
    parts = []
    for i, doc in enumerate(store.similarity_search(query, k=k), 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page")
        label = f"{source}, p.{page + 1}" if isinstance(page, int) else source
        parts.append(f"[{i}] ({label})\n{doc.page_content.strip()}")
    return "\n\n".join(parts)
