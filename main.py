import streamlit as st
import time
import tempfile
import numpy as np

# ---------------- IMPORTS ----------------
from ingestion.pdf_loader import load_and_chunk_pdfs

from embeddings.embedder import Embedder
from embeddings.openai_embedder import OpenAIEmbedder

from vectorstores.chromadb_store import ChromaStore
from vectorstores.pinecone_store import PineconeStore

from retrieval.bm25_search import BM25Search
import retrieval.brute_force as bf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Vector DB Benchmark", layout="wide")
st.title("📊 Interactive Vector Database Benchmarking Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configuration")

embedding_type = st.sidebar.selectbox(
    "Embedding Type",
    ["Sentence Transformers", "OpenAI Embeddings"]
)

embedding_model = None
openai_api_key = None

if embedding_type == "Sentence Transformers":
    embedding_model = st.sidebar.selectbox(
        "Sentence Transformer Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    )

if embedding_type == "OpenAI Embeddings":
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key (Required)",
        type="password"
    )

retrieval_method = st.sidebar.selectbox(
    "Retrieval Method",
    ["BM25", "Brute Force", "HNSW (ChromaDB)", "HNSW (Pinecone)"]
)

pinecone_api_key = None
if retrieval_method == "HNSW (Pinecone)":
    pinecone_api_key = st.sidebar.text_input(
        "Pinecone API Key",
        type="password"
    )

top_k = st.sidebar.slider("Top-K Results", 1, 10, 5)

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- MAIN PIPELINE ----------------
if uploaded_files and st.button("🚀 Run Benchmark"):

    # Save PDFs safely (Windows compatible)
    pdf_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_paths.append(tmp.name)

    # Load & chunk
    with st.spinner("📄 Loading and chunking PDFs..."):
        documents = load_and_chunk_pdfs(pdf_paths)
        st.write(f"Total text chunks: **{len(documents)}**")

    # ---------------- EMBEDDINGS ----------------
    try:
        with st.spinner("🧠 Generating embeddings..."):
            if embedding_type == "OpenAI Embeddings":
                if not openai_api_key:
                    st.error("❌ OpenAI API Key is REQUIRED")
                    st.stop()

                embedder = OpenAIEmbedder(openai_api_key)
                embeddings = np.array(embedder.embed(documents))

            else:
                embedder = Embedder(embedding_model)
                embeddings = embedder.embed(documents)

    except Exception as e:
        st.warning("⚠️ OpenAI failed — falling back to Sentence Transformers")
        embedder = Embedder("all-MiniLM-L6-v2")
        embeddings = embedder.embed(documents)
        st.code(str(e))

    # ---------------- RETRIEVAL SETUP ----------------
    if retrieval_method == "BM25":
        engine = BM25Search(documents)

    elif retrieval_method == "Brute Force":
        engine = bf.BruteForceSearch(embeddings)

    elif retrieval_method == "HNSW (ChromaDB)":
        chroma = ChromaStore("vector-db-benchmark")
        chroma.add(embeddings, documents)

    else:
        if not pinecone_api_key:
            st.error("❌ Pinecone API Key required")
            st.stop()

        pinecone = PineconeStore(
            index_name="vector-db-benchmark",
            dimension=embeddings.shape[1],
            api_key=pinecone_api_key
        )
        pinecone.add(embeddings, documents)

    # ---------------- QUERY ----------------
    st.subheader("🔍 Query the System")
    query = st.text_input("Enter your query")

    if query:
        start = time.time()

        # Query embedding
        query_embedding = embedder.embed([query])[0]

        # Search
        if retrieval_method == "BM25":
            idxs = engine.search(query, top_k)
            results = [documents[i] for i in idxs]

        elif retrieval_method == "Brute Force":
            idxs = engine.search(query_embedding, top_k)
            results = [documents[i] for i in idxs]

        elif retrieval_method == "HNSW (ChromaDB)":
            res = chroma.search(query_embedding, top_k)
            results = res["documents"][0]

        else:
            res = pinecone.search(query_embedding, top_k)
            results = [m["metadata"]["text"] for m in res["matches"]]

        latency = (time.time() - start) * 1000

        st.success(f"⏱ Response Time: {latency:.2f} ms")

        st.subheader("📄 Retrieved Results")
        for i, doc in enumerate(results, 1):
            st.markdown(f"**Result {i}:**")
            st.write(doc[:400] + "...")
