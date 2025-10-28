# 📚 PDF Chat Assistant with Gemini

An **AI-powered Document & Website Assistant** built using **Streamlit, Gemini API, LangChain, and Qdrant**.  
This app lets you **chat with your PDFs or website content** using a RAG (Retrieval-Augmented Generation) pipeline for grounded, context-aware answers.

---

## 🎥 Demo Video

▶ [Watch the Full Demo on YouTube](https://youtu.be/Tc1cKGNg34I)

---

## 🚀 What I’ve Built

You’ve created an **AI-powered assistant** that can understand, index, and answer questions about **PDF documents** and **websites** — all in natural language.

Here’s the flow in simple words 👇

### 1️⃣ Upload or Crawl
- Upload **one or more PDF files**  
- OR enter a **website URL**  
The app will ingest, parse, and prepare the content for intelligent querying.

### 2️⃣ Ingestion & Indexing
- Each document or webpage is **split into smaller text chunks** for efficient search.  
- Text is embedded into vectors using **Google Gemini Embeddings**.  
- These embeddings are stored in a **Qdrant Vector Database** for fast semantic similarity search.

### 3️⃣ Query & Retrieval
- When a user asks a question, the query is also embedded.  
- Qdrant retrieves the **most semantically relevant chunks** from all stored documents.  
- The system ensures that the model always references your data, not generic web content.

### 4️⃣ LLM Response Generation
- The relevant chunks are combined with the user’s question into a **context-rich prompt**.  
- The **Gemini LLM** then generates a response grounded in the actual document/website content.  
- This minimizes hallucinations and increases factual accuracy.

### 5️⃣ Interactive Streamlit UI
All this is delivered in a smooth, intuitive Streamlit app where you can:
- 🧠 Upload PDFs or Enter URLs  
- 💬 Ask questions interactively  
- 🪄 Get AI-generated answers grounded in your content  
- 🌓 Switch between dark/light mode  
- 💾 Download chat history for later use  

---

## 🌟 Key Features

- 📄 **Multi-PDF Support** – Upload and query multiple documents at once  
- 🔍 **Context-Aware Responses** – Answers grounded in your data  
- ⚡ **RAG Pipeline** – Retrieval-Augmented Generation with Gemini + Qdrant  
- 🌐 **Website Mode** – Chat with any webpage content  
- 🧠 **Gemini Embeddings** – Accurate semantic similarity search  
- 🧩 **LangChain Integration** – Manages prompt templates, chains, and memory  
- 💬 **Interactive Chat UI** – Real-time responses in Streamlit  
- 🌓 **Dark/Light Theme Toggle**  
- 💾 **Export Chat History** – Save your interactions as `.txt`

---

## 🧩 Tech Stack

| Layer | Tools |
|--------|-------|
| **Frontend** | Streamlit |
| **LLM** | Google Gemini Pro API |
| **Framework** | LangChain |
| **Vector Database** | Qdrant |
| **Embeddings** | Gemini Text Embeddings |
| **Language** | Python 3.10+ |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
git clone https://github.com/SabitaKumariDev/PDF-Chat-Assistant-with-Gemini.git
cd PDF-Chat-Assistant-with-Gemini


2️⃣ Create a Virtual Environment
python -m venv .venv
. .venv/Scripts/activate   # (Windows)
# source .venv/bin/activate  (macOS/Linux)

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Create a .env File
Create a .env file in your root directory with:
GOOGLE_API_KEY=your_gemini_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=https://your-instance-name.qdrant.tech

🚀 Run the App
streamlit run app.py

Once it’s running, open the local URL (e.g. http://localhost:8501) in your browser.


💬 How It Works (RAG Pipeline Summary)

Document Ingestion – PDFs/websites are uploaded or crawled.

Chunking – Text is split into smaller segments.

Vectorization – Segments converted into embeddings via Gemini API.

Storage – Embeddings stored in Qdrant vector database.

Retrieval – On each user query, similar chunks are retrieved using cosine similarity.

LLM Response – Retrieved chunks + question passed to Gemini LLM → grounded, accurate answers.

git clone https://github.com/SabitaKumariDev/PDF-Chat-Assistant-with-Gemini.git

cd PDF-Chat-Assistant-with-Gemini
