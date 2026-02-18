import os
import time
import io
import re
import requests
import numpy as np
import cv2
import pytesseract
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from supabase import create_client, Client

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
HF_API_URL = os.environ.get("HF_API_URL")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase Init Error: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

http_session = requests.Session()

def calculate_usage_from_text(text):
    clean_text = text.lower().replace(',', '.')
    gb_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)'
    matches = re.findall(gb_pattern, clean_text)
    values = [float(m) for m in matches if 0.01 < float(m) < 2000]
    if len(values) >= 2:
        return round(max(values) - min(values), 2), round(min(values), 2), "Max-Min Calc"
    elif len(values) == 1:
        return 0.0, values[0], "Single Value"
    return 0.0, 0.0, "No Data"

@app.get("/")
def home():
    return {"status": "active"}

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(None),
    text_input: str = Form(None),
    task_type: str = Form("quota") 
):
    raw_text = ""
    try:
        if text_input:
            raw_text = text_input
        elif file:
            content = await file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            raw_text = pytesseract.image_to_string(adaptive, config="--psm 6")
        else:
            return {"error": "No input provided"}

        if task_type == "raw_ocr":
            lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) > 3]
            return {"type": "raw_ocr", "lines": lines}

        if task_type == "quota":
            used, rem, method = calculate_usage_from_text(raw_text)
            return {"used": used, "remaining": rem, "method": method}

        elif task_type == "invoice":
            if not HF_API_URL:
                return {"error": "HF_API_URL_MISSING"}
            
            lines = [l.strip() for l in re.split('\n|,', raw_text) if len(l.strip()) > 3]
            matches = []
            for line in lines:
                try:
                    hf_res = http_session.get(HF_API_URL, params={"text": line}, timeout=5)
                    data = hf_res.json()
                    if data.get("matched_product") != "No Match":
                        matches.append({
                            "original_text": line,
                            "master_name": data.get("matched_product"),
                            "confidence": data.get("confidence")
                        })
                except: continue
            return {"type": "invoice", "matches": matches}
            
    except Exception as e:
        return {"error": str(e)}

@app.post("/submit-report")
async def submit_report(
    file: UploadFile = File(...),
    report_id: str = Form(...),      
    phone_number: str = Form(...),
    outlet_id: str = Form(None),
    outlet_name_manual: str = Form(None),
    user_corrected_usage: float = Form(...)
):
    if not supabase: return {"status": "error", "message": "Database not connected"}
    content = await file.read()
    try:
        current_year = time.strftime('%Y')
        current_week = time.strftime('%W')
        filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
        folder_name = outlet_name_manual or outlet_id or "Unsorted"
        clean_folder = re.sub(r'[^a-zA-Z0-9_-]', '', folder_name)
        storage_path = f"Quota/{clean_folder}/{current_year}/Week_{current_week}/{filename}"
        bucket_name = "Screenshots"
        
        supabase.storage.from_(bucket_name).upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type}
        )
        
        public_url_resp = supabase.storage.from_(bucket_name).get_public_url(storage_path)
        final_image_url = public_url_resp if isinstance(public_url_resp, str) else public_url_resp.get("publicUrl")

        data_payload = {
            "image_url": final_image_url,
            "confirmation": True,                 
            "confirmed_at": "now()"
        }
        
        response = supabase.table("quota_reports").update(data_payload).eq("id", report_id).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}