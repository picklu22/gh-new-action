# Simple PDF Chatbot with Gemini (Google GenAI)
# This version is easier to understand and use

import os
import glob
import json
from pypdf import PdfReader
import numpy as np
from google import genai
import faiss

# Load Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client()

# Step 1: Load PDFs and Extract Text
def load_pdfs(folder_path):
    pdf_files = glob.glob(folder_path + "/*.pdf")
    all_text = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        all_text.append({"file": os.path.basename(pdf), "text": text})
    return all_text

# Step 2: Split text into smaller chunks
def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Step 3: Generate Embeddings
def get_embeddings(text_list):
    response = client.models.embed_content(model="gemini-embedding-001", contents=text_list)
    return np.array(response.embeddings, dtype="float32")

# Step 4: Create FAISS index
def create_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# Step 5: Query the chatbot
def chatbot_query(query, index, chunks, files):
    query_emb = get_embeddings([query])
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, 3)

    context = ""
    for i in indices[0]:
        context += f"{files[i]}: {chunks[i]}\n"

    prompt = f"You are an AI answering based on PDF content only.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

# Main process
def main():
    print("Indexing documents...")
    docs = load_pdfs("./pdfs")

    chunks, files = [], []
    for doc in docs:
        for chunk in split_text(doc["text"]):
            chunks.append(chunk)
            files.append(doc["file"])

    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)
    print("Chatbot ready! Type your questions.")

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        print("Bot:", chatbot_query(query, index, chunks, files))

if __name__ == "__main__":
    main()
