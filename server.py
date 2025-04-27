from fastapi import FastAPI, File, UploadFile, Form
import uuid
import shutil
import cv2
from PIL import Image
import numpy as np
from easyocr import Reader as EasyOCRReader
from paddleocr import PaddleOCR
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from pydantic import BaseModel
from typing import List, Optional
from Gemini_agents import process_user_input
import local
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pyngrok import ngrok, conf
from API_keys import key_ngrok

# Define absolute path for results directory
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

app = FastAPI()

# instantiate OCR and caption models
easyocr_reader = EasyOCRReader(['en'], gpu=True)
paddle_reader = PaddleOCR(use_angle_cls=True, lang='en')
blip_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
blip_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b', torch_dtype=torch.float16).eval()

class QueryRequest(BaseModel):
    query: str
    top_n: Optional[int] = 10
    top_n_template: Optional[int] = 5

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
        sentiment_preference=sentiment_preference,
        top_n=req.top_n,
        top_n_template=req.top_n_template
    )

    meme_urls = [f"/static/{filename}" for filename in recommended_memes]

    return {
        "memes": meme_urls,
        "need_template": need_template,
        "details": details
    }

@app.post('/recommend/upload')
async def recommend_upload(
    context: str = Form(...),
    file: UploadFile = File(...)
):
    # save incoming file
    tmp_filename = f"tmp_{uuid.uuid4().hex}_{file.filename}"
    tmp_path = os.path.join(BASE_DIR, tmp_filename)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    else:
        for entry in os.listdir(RESULTS_DIR):
            path = os.path.join(RESULTS_DIR, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    with open(tmp_path, 'wb') as out:
        shutil.copyfileobj(file.file, out)
    try:
        if context == 'local':
            # extract text via EasyOCR
            results = easyocr_reader.readtext(tmp_path, detail=0)
            text = ' '.join(results)
            rec_files = local.get_similar_memes(
                topics=text, usages=False, need_template=False, search_by='Local'
            )
            with open("log/error_log.txt", "a") as log_file:
                log_file.write(f"Extracted text: {text}\n")
                log_file.write(f"Recommended files: {rec_files}\n")
        else:
            # mask text regions via PaddleOCR, then caption via BLIP-2
            img = cv2.imread(tmp_path)
            ocr_res = paddle_reader.ocr(tmp_path, cls=True)
            for line in ocr_res:
                for word in line:
                    x1, y1 = map(int, word[0][0])
                    x2, y2 = map(int, word[0][2])
                    cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), -1)
            # convert to PIL image for BLIP
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = blip_processor(images=pil_img, return_tensors='pt').to(blip_model.device, torch.float16)
            with torch.no_grad():
                out_ids = blip_model.generate(**inputs)
            caption = blip_processor.decode(out_ids[0], skip_special_tokens=True)
            rec_files = local.get_similar_memes(
                topics=caption, usages=False, need_template=False, search_by='Global'
            )
            with open("log/error_log.txt", "a") as log_file:
                log_file.write(f"Caption: {caption}\n")
                log_file.write(f"Recommended files: {rec_files}\n")

        # build URLs
        meme_urls = [f"/static/{fname}" for fname in rec_files]
        return {"memes": meme_urls}
    finally:
        # cleanup temp file
        try: os.remove(tmp_path)
        except: pass

app.mount("/static", StaticFiles(directory=RESULTS_DIR), name="static")
app.mount("/UI_images", StaticFiles(directory=os.path.join(BASE_DIR, "UI_images")), name="UI_images")

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "index.html"), "r") as f:
        return f.read()

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)