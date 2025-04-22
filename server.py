from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from Gemini_agents import process_user_input
import local
import shutil
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pyngrok import ngrok, conf
from API_keys import key_ngrok

# Define absolute path for results directory
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
def recommend(req: QueryRequest):
    processed = process_user_input(req.query)
    if not processed:
        return {"error": "Could not understand the query."}

    need_template = processed["need_template"]
    has_topic = processed["has_topics"]
    has_usage = processed["has_usages"]
    details = processed.get("details") or {}

    topics_str = " ".join(details.get("topics", [])) if has_topic else None
    usages_str = "|".join(details.get("usages", [])) if has_usage else None
    search_by = details.get("recommendation_focus", "Global").capitalize() if has_topic else None
    sentiment_preference = details.get("sentiment_preference", "neutral") if not has_topic and not has_usage else None
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    else:
        for entry in os.listdir(RESULTS_DIR):
            path = os.path.join(RESULTS_DIR, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    recommended_memes = local.get_similar_memes(
        topics=topics_str,
        need_template=need_template,
        usages=usages_str,
        search_by=search_by,
        sentiment_preference=sentiment_preference
    )

    meme_urls = [f"/static/{filename}" for filename in recommended_memes]

    return {
        "memes": meme_urls,
        "need_template": need_template,
        "details": details
    }

app.mount("/static", StaticFiles(directory=RESULTS_DIR), name="static")
app.mount("/UI_images", StaticFiles(directory=os.path.join(BASE_DIR, "UI_images")), name="UI_images")

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "index.html"), "r") as f:
        return f.read()

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)