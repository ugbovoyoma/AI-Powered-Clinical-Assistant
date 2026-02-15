import json, pandas as pd
import os, json, re, math, uuid
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Embeddings / retrieval
from sentence_transformers import SentenceTransformer


# FAISS is optional; if unavailable, we'll fall back to cosine search
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


load_dotenv()


# API KEYS
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME_FOR_CHAT = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

df= pd.read_json(r"C:\Users\CB\Desktop\AI Powered App\data\cleanedData\cleaned_symptom_disease.json", lines=True)
df.head()

#delete tokens column

data = df.drop(columns=["tokens"])

#extract the disease and symptom column as a list

corpus_texts = data["symptoms"].astype(str).str.strip().tolist()
labels = data["diseases"].astype(str).str.strip().tolist()

embeddingModel= "sentence-transformers/all-MiniLM-L6-v2"

embedder= SentenceTransformer(embeddingModel)

# Create one vector per record’s text (Encoding the dataset)

emb_matrix = embedder.encode(
    corpus_texts, 
    convert_to_numpy=True, 
    show_progress_bar=True
).astype("float32")

# If FAISS is available, build a cosine-similarity index. Otherwise, we’ll search with numpy.

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

if FAISS_AVAILABLE:
    faiss.normalize_L2(emb_matrix) # normalize so inner product == cosine
    index = faiss.IndexFlatIP(emb_matrix.shape[1]) # IP = inner product
    index.add(emb_matrix)
    print("Vector index: FAISS (cosine)")
else:
    index = None
    print("Vector index: numpy cosine (FAISS not installed)")

print(f"Embedding matrix shape: {emb_matrix.shape}")

from numpy.linalg import norm

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    denom = (norm(a) * norm(b)) or 1e-12
    return float(np.dot(a, b) / denom)

def retrieve(query_text: str, top_k: int = 5):
    """
    Quick test retriever:
    - Encodes the query
    - Searches FAISS if available, otherwise uses a NumPy cosine loop
    - Returns a list of dicts: {text, label, score, idx}
    """
    q_vec = embedder.encode([query_text], convert_to_numpy=True)[0].astype("float32")

    if index is not None:
        # FAISS path (inner product on normalized vectors == cosine)
        q_norm = q_vec / (norm(q_vec) + 1e-12)
        scores, idxs = index.search(q_norm.reshape(1, -1), min(top_k, len(corpus_texts))) # this is just like asking FAISS Which vectors in my 
                                                                                          # database are closest to my query?
        scores = scores[0]
        idxs = idxs[0]
        hits = [
            {
                "idx": int(i),
                "text": corpus_texts[i],
                "label": labels[i],
                "score": float(scores[j]),
            }
            for j, i in enumerate(idxs)
        ]
    else:
        # NumPy fallback
        sims = [ _cosine_sim(q_vec, emb_matrix[i]) for i in range(len(corpus_texts)) ]
        idxs = np.argsort(sims)[::-1][:top_k]
        hits = [
            {
                "idx": int(i),
                "text": corpus_texts[i],
                "label": labels[i],
                "score": float(sims[i]),
            }
            for i in idxs
        ]
    return hits

# Try a quick search

example_query = 'love'  
top_hits = retrieve(example_query, top_k=3)
print(f"\n🔎 Query: {example_query}")
for rank, h in enumerate(top_hits, start=1):
    print(f"{rank}. [{h['score']:.3f}] {h['label']}  →  {h['text'][:120]}...")

import requests

RAG_SYSTEM = (
    "You are a cautious clinical QA assistant. Use ONLY the provided context to answer. "
    "If the context is insufficient, say you cannot answer. Keep answers under 6 sentences."
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# API key loaded from .env file via load_dotenv() at top of file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("No OPENROUTER_API_KEY found in environment.")

MODEL_NAME_FOR_CHAT =  "openai/gpt-oss-20b:free"

def _build_context(hits):
    """
    Turn retrieve() hits into a readable context block + a list of source IDs.
    We synthesize a simple 'source' like json:{idx}:{label} so you can cite it.
    """
    lines, sources = [], []
    for i, h in enumerate(hits, start=1):
        src = f"json:{h['idx']}:{h['label']}"
        lines.append(f"[{i}] {h['text']} (source={src})")
        sources.append(src)
    return "\n\n".join(lines), sources

def answer_with_context(question: str, k: int = 5):
    hits = retrieve(question, top_k=k)
    context, sources = _build_context(hits)

    if OPENROUTER_API_KEY:
        payload = {
            "model": MODEL_NAME_FOR_CHAT,
            "messages": [
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://local",
            "X-Title": "rag_test",
        }
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip()
    else:
        # No key? Show the snippets so you still learn what retrieval found.
        answer = "(No API key) Top snippets:\n" + "\n".join([f"• {h['text'][:160]}..." for h in hits])

    return answer, sources


# Test 
q = "What are malaria symptoms?"
text, srcs = answer_with_context(q, k=5)
print(text)
print("\nSources:")
for s in srcs:
    print("-", s)
