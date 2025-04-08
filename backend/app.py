from flask import Flask, request, jsonify
import faiss
import json
import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()

# Google API Key setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embedding_model = genai.embed_content

# Paths
INDEX_PATH = "index.faiss"
META_PATH = "shldataset.json"

def get_embedding(text):
    """Get embedding using Gemini embedding model"""
    response = embedding_model(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

def build_or_load_faiss():
    """Load FAISS index and metadata or build it from metadata.json"""
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("Loading saved FAISS index and metadata...")
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
    else:
        print("Building FAISS index from metadata.json...")
        with open(META_PATH, "r") as f:
            metadata = json.load(f)

        vectors = []
        for entry in metadata:
            # Concatenate all relevant fields into a single string for embedding
            concat_text = " ".join([
                str(entry.get("title", "")),
                str(entry.get("description", "")),
                str(entry.get("job_level", "")),
                str(entry.get("language", "")),
                str(entry.get("duration", "")),
                str(entry.get("adaptive_irt", "")),
                str(entry.get("remote_testing", ""))
            ])
            vec = get_embedding(concat_text)
            vectors.append(vec)

        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors).astype("float32"))

        faiss.write_index(index, INDEX_PATH)

    return index, metadata

index, metadata = build_or_load_faiss()

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_query = data.get("query", "")
    if not user_query:
        return jsonify([])

    query_vec = np.array(get_embedding(user_query)).astype("float32").reshape(1, -1)
    top_k = 5
    D, I = index.search(query_vec, top_k)

    results = [metadata[i] for i in I[0] if i < len(metadata)]
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

