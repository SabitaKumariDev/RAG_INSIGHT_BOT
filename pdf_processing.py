import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Windows + Python 3.12: ensure compatible event loop policy (harmless on non-Windows)
import asyncio
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Read env once
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Optional: small guardrails for missing keys
if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY is not set. Please add it to your .env")
if not QDRANT_URL:
    st.warning("QDRANT_URL is not set. Please add it to your .env")
if not QDRANT_API_KEY:
    st.info("QDRANT_API_KEY not set. If your Qdrant is public or local without auth, this may be fine.")

def _make_embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,  
        transport="rest",              
    )

def process_pdf(uploaded_file, collection_name):
    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load & split
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(docs)

    # Limit work to keep Streamlit responsive
    MAX_CHUNKS = 20
    if len(split_docs) > MAX_CHUNKS:
        st.warning(f"Large PDF detected — processing first {MAX_CHUNKS} chunks to stay responsive.")
        split_docs = split_docs[:MAX_CHUNKS]

    # Create embedder lazily (inside the function) so imports don’t spin up clients prematurely
    embedding_model = _make_embedder()

    try:
        st.write("Starting embedding…")
        vector_db = QdrantVectorStore.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=collection_name,
            force_recreate=False,
        )
        st.success("Embedding complete and uploaded to Qdrant.")
        return vector_db
    except Exception as e:
        st.error(f"Error during embedding or Qdrant upload: {e}")
        return None
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
