import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

def retrieve_top_k(query_embedding, embeddings, k=10):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

def get_query_embedding(query):
    return model.encode(query)

# Streamlit UI
st.set_page_config(page_title="IR App", layout="wide")
st.title("📄 Information Retrieval using Document Embeddings")

# Input query
query = st.text_input("Enter your query:")

# Top-K slider
k = st.slider("Number of results to display", min_value=1, max_value=20, value=10)

if st.button("Search") and query.strip():
    start_time = time.time()
    
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings, k)
    
    end_time = time.time()
    st.write(f"⏱ Search completed in {end_time - start_time:.2f} seconds")
    
    st.write(f"### Top {k} Relevant Documents:")
    for i, (doc, score) in enumerate(results, start=1):
        st.write(f"{i}. **{doc}** (Score: {score:.4f})")
