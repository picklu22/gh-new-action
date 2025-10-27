"""
pdf_gemini_chatbot.py

Simple Retrieval-Augmented Chatbot that:
- Loads multiple PDFs from a folder
- Extracts and chunks text
- Creates embeddings using Google Gemini embeddings (google-genai)
- Stores embeddings in a local FAISS index (numpy + faiss)
- Answers user queries by retrieving relevant chunks and calling Gemini to generate the final answer

Prereqs:
    pip install -U google-genai pypdf faiss-cpu numpy python-dotenv

Environment:
    Set your Gemini API key in GEMINI_API_KEY environment variable (or use .env)

Notes:
- This is a minimal, readable example. For production, add error handling, batching, persistence, metadata, and rate-limit/backoff logic.
- Based on Google Gemini quickstart and embeddings docs.
"""

import os
import glob
import json
import math
import numpy as np
from typing import List, Tuple
from pypdf import PdfReader
from dotenv import load_dotenv

# FAISS optional import; if not available the code will raise an informative error
try:
    import faiss
except Exception as e:
    raise ImportError("faiss is required. Install with: pip install faiss-cpu")

# Google GenAI client
from google import genai

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Please set GEMINI_API_KEY in your environment (or .env file)")

# Initialize Gemini client
client = genai.Client()

# ---------- PDF loading and text chunking ----------

def load_pdfs_from_folder(folder: str) -> List[Tuple[str, str]]:
    """Return list of (filename, text) for every PDF in folder."""
    files = glob.glob(os.path.join(folder, "*.pdf"))
    docs = []
    for f in files:
        reader = PdfReader(f)
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                # skip pages that fail
                continue
        full_text = "\n".join(text_parts)
        docs.append((os.path.basename(f), full_text))
    return docs


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple character-based chunking with overlap. Returns list of text chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

# ---------- Embeddings and FAISS index ----------

def embed_texts(texts: List[str], model: str = "gemini-embedding-001") -> np.ndarray:
    """Call Gemini embeddings API and return numpy array of shape (n, dim)."""
    # The docs show client.models.embed_content usage.
    resp = client.models.embed_content(model=model, contents=texts)
    # resp.embeddings is a list of lists
    embeddings = np.array(resp.embeddings, dtype=np.float32)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS index (inner product) and normalize vectors for cosine similarity."""
    # Normalize to unit length to use inner-product as cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# ---------- Small persistence helpers ----------

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

# ---------- Putting it all together: index PDFs ----------

def index_pdfs(folder: str, index_path: str = "faiss.index", meta_path: str = "meta.json"):
    """Load PDFs, chunk, embed, build FAISS index and save metadata.
    meta.json stores a list of dicts: [{"source": filename, "chunk": text, "start": int, "end": int}, ...]
    """
    docs = load_pdfs_from_folder(folder)
    all_chunks = []
    meta = []
    for fname, full_text in docs:
        chunks = chunk_text(full_text)
        for c in chunks:
            meta.append({"source": fname, "chunk": c[:2000]})
            all_chunks.append(c)

    if not all_chunks:
        raise ValueError("No text found in PDFs in folder: %s" % folder)

    print(f"Embedding {len(all_chunks)} chunks...")
    embeds = embed_texts(all_chunks)
    idx = build_faiss_index(embeds)
    save_index(idx, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # Save embeddings too (useful for re-ranking or inspection)
    np.save("embeddings.npy", embeds)
    print("Indexing complete. Saved:", index_path, meta_path, "embeddings.npy")

# ---------- Querying / Chat ----------

def retrieve_top_k(query: str, k: int = 4, model: str = "gemini-embedding-001") -> List[Tuple[float, dict]]:
    """Return top-k (score, metadata) results for the query."""
    # Load meta and index (for simplicity we re-load every time; optimize as needed)
    meta = json.load(open("meta.json", "r", encoding="utf-8"))
    index = load_index("faiss.index")
    # Embed query
    q_emb = embed_texts([query], model=model)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((float(score), meta[idx]))
    return results


def build_prompt_with_context(query: str, retrieved: List[Tuple[float, dict]]) -> str:
    """Create a prompt that provides the retrieved chunks as context to Gemini."""
    context_parts = []
    for score, m in retrieved:
        context_parts.append(f"Source: {m['source']}\nContent:\n{m['chunk']}\n---\n")
    context = "\n".join(context_parts)
    prompt = (
        "You are an assistant that answers user questions based only on the provided source snippets. "
        "If the answer is not present, say 'I don't know' and offer to search the documents for related info.\n\n"
        "Provided snippets:\n" + context + "\nUser question: " + query + "\n\nAnswer concisely and cite the source file names in square brackets."
    )
    return prompt


def ask_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Call Gemini to generate an answer using the supplied prompt."""
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text


def chat_loop():
    print("PDF Gemini Chat — type 'exit' to quit")
    while True:
        q = input("\nYou: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        retrieved = retrieve_top_k(q, k=4)
        prompt = build_prompt_with_context(q, retrieved)
        answer = ask_gemini(prompt)
        print("\nAssistant:\n", answer)

# ---------- Example CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple PDF -> Gemini RAG chatbot")
    parser.add_argument("--index", action="store_true", help="Index PDFs in ./pdfs folder")
    parser.add_argument("--pdf-folder", default="./pdfs", help="Folder with PDFs to index")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat (requires existing index)")

    args = parser.parse_args()

    if args.index:
        index_pdfs(args.pdf_folder)
    if args.chat:
        chat_loop()

    if not args.index and not args.chat:
        parser.print_help()
