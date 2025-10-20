import streamlit as st
# --- event loop fix for Streamlit + gRPC on Windows ---
import sys, asyncio, os

# (optional) quiet some libs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# gRPC's asyncio client needs a compatible loop policy on Windows
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Streamlit runs code in a worker thread; ensure that thread has a loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
# --- end fix ---

st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("RAG AI Insight Bot")

# Main page selector
st.header("Choose a Chat Mode")
option = st.radio(
    "What would you like to do?",
    ("Chat with PDF", "Chat with Website"),
    index=0,
    horizontal=True
)

if option == "Chat with PDF":
    from pdf_chat_page import pdf_chat_workflow
    pdf_chat_workflow()
elif option == "Chat with Website":
    from website_chat_page import website_chat_workflow
    website_chat_workflow()
