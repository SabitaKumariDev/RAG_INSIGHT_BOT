import os
import asyncio
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Windows + Python 3.12 gRPC/asyncio gotcha: use selector policy (harmless elsewhere)
# ──────────────────────────────────────────────────────────────────────────────
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Env
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_OR_GEMINI_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # optional for local/no-auth

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def collection_exists(qdrant_url: str, qdrant_api_key: str | None, collection_name: str) -> bool:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    try:
        collections = client.get_collections().collections
    except Exception:
        # If listing fails (e.g., perms), assume it doesn't exist so we try to create it.
        return False
    return any(col.name == collection_name for col in collections)

def make_embedder() -> GoogleGenerativeAIEmbeddings:
    if not GOOGLE_OR_GEMINI_KEY:
        st.error("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env")
        st.stop()
    # Critical: force REST to avoid gRPC asyncio event loop errors
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_OR_GEMINI_KEY,
        transport="rest",
    )

def make_chat_client() -> OpenAI:
    if not GOOGLE_OR_GEMINI_KEY:
        st.error("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env")
        st.stop()
    return OpenAI(
        api_key=GOOGLE_OR_GEMINI_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────
def website_chat_workflow():
    st.title("Chat with Website")

    url = st.text_input("Enter website URL to chat with:")
    process_button = st.button("Process Website", type="primary")

    # session state
    st.session_state.setdefault("chat_ready", False)
    st.session_state.setdefault("vector_db", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("show_chat", False)

    if url and process_button:
        url_hash = get_url_hash(url)
        collection_name = f"website_{url_hash}"

        embedding_model = make_embedder()

        if collection_exists(QDRANT_URL, QDRANT_API_KEY, collection_name):
            st.success("Website already processed. Using existing embeddings.")
            st.session_state.vector_db = QdrantVectorStore.from_existing_collection(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=collection_name,
                embedding=embedding_model,
            )
            st.session_state.chat_ready = True
            st.session_state.show_chat = False
            st.rerun()
        else:
            with st.spinner("Loading and embedding website content..."):
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                except Exception as e:
                    st.error(f"Failed to load the URL: {e}")
                    return
                
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
                )
                split_docs = splitter.split_documents(docs)

                # keep Streamlit responsive
                MAX_CHUNKS = 20
                if len(split_docs) > MAX_CHUNKS:
                    st.warning(
                        f"Website is large; only processing the first {MAX_CHUNKS} chunks to avoid timeout."
                    )
                    split_docs = split_docs[:MAX_CHUNKS]

                try:
                    vector_db = QdrantVectorStore.from_documents(
                        documents=split_docs,
                        embedding=embedding_model,
                        url=QDRANT_URL,
                        api_key=QDRANT_API_KEY,
                        collection_name=collection_name,
                        force_recreate=False,
                    )
                    st.success("Website processed and indexed!")
                    st.session_state.vector_db = vector_db
                    st.session_state.chat_ready = True
                    st.session_state.show_chat = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during embedding or Qdrant upload: {e}")

    if st.session_state.get("chat_ready"):
        if st.button("Chat with Website"):
            st.session_state.show_chat = True

    if st.session_state.get("show_chat"):
        handle_website_chat()

# ──────────────────────────────────────────────────────────────────────────────
# Chat UI
# ──────────────────────────────────────────────────────────────────────────────
def handle_website_chat():
    client = make_chat_client()

    st.subheader("Ask a question about the website")
    user_query = st.text_input("You:", key="website_user_input")

    if user_query:
        try:
            search_results = st.session_state.vector_db.similarity_search(query=user_query, k=5)
        except Exception as e:
            st.error(f"Vector search failed: {e}")
            return

        context = "\n\n\n".join(
            f"Content: {res.page_content}\nSource: {res.metadata.get('source', 'N/A')}"
            for res in search_results
        )

        system_prompt = f"""
You are a helpful AI assistant who answers user queries based ONLY on the provided context.

Context:
{context}
If the answer is not in the context, say you don't know and suggest what page/section to look at.
"""

        try:
            response = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.2,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return

        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("assistant", answer))

    # Show chat history (newest first)
    for role, msg in reversed(st.session_state.chat_history):
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

    # Export chat
    if st.button("Export Chat History"):
        text_blob = "\n".join(f"{role.capitalize()}: {msg}" for role, msg in st.session_state.chat_history)
        st.download_button(
            label="Download Chat History",
            data=text_blob,
            file_name="website_chat_history.txt",
            mime="text/plain",
        )


