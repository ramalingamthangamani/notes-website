from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Enable CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "https://notes-website.vercel.app"],  # Add Vercel URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained sentence-transformers model from environment variable
model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# In-memory storage for notes (replace with database in production)
notes = []

class Note(BaseModel):
    id: int
    content: str
    timestamp: str

class SearchQuery(BaseModel):
    query: str

# Function to get embeddings for text
def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512, clean_up_tokenization_spaces=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Save note endpoint
@app.post("/notes")
async def save_note(note: Note):
    notes.append(note.dict())
    return {"message": "Note saved successfully"}

# Search notes endpoint
@app.post("/search")
async def search_notes(query: SearchQuery):
    if not notes:
        return {"results": []}
    
    # Get query embedding
    query_embedding = get_embedding(query.query)
    
    # Compute cosine similarity for each note
    results = []
    for note in notes:
        note_embedding = get_embedding(note["content"])
        similarity = np.dot(query_embedding, note_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(note_embedding)
        )
        results.append({"note": note, "score": float(similarity)})
    
    # Sort by similarity score and filter relevant results
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    results = [r["note"] for r in results if r["score"] > 0.5]  # Threshold for relevance
    
    return {"results": results}

# Delete note endpoint
@app.delete("/notes/{note_id}")
async def delete_note(note_id: int):
    global notes
    notes = [note for note in notes if note["id"] != note_id]
    return {"message": "Note deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render-assigned PORT or default to 8000
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)