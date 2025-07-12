import os
import uvicorn

# your FastAPI setup
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Backend is running successfully!"}

# Run if it's the main module
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
git add app.py
git commit -m "Fix: Bind to Render's port using env"
git push origin main
