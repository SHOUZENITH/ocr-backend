import os
import time
import io
import re
import requests
import numpy as np
import cv2
import pytesseract
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
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
    except Exception:
        pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_size_to_gb(value_str, unit_str="GB"):
    try:
        val = float(value_str.replace(',', '.'))
        unit = unit_str.upper().strip()
        if "MB" in unit: return val / 1024.0
        if "KB" in unit: return val / (1024.0 * 1024.0)
        return val
    except: return 0.0

def calculate_usage_from_text(text):
    clean_text = text.lower().replace(',', '.')
    gb_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)'
    matches = re.findall(gb_pattern, clean_text)
    values = [float(m) for m in matches if 0.01 < float(m) < 2000]
    if len(values) >= 2:
        return round(max(values) - min(values), 2), round(min(values), 2), "Max-Min Calc"
    return (0.0, values[0], "Single Value") if values else (0.0, 0.0, "No Data")

@app.get("/")
def home():
    return {"status": "active"}

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    task_type: str = Form("quota")
):
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        raw_text = pytesseract.image_to_string(adaptive, config="--psm 6")

        if task_type == "quota":
            used, rem, method = calculate_usage_from_text(raw_text)
            return {"used": used, "remaining": rem, "method": method}

        elif task_type == "invoice":
            if not HF_API_URL:
                return {"error": "HF_API_URL_MISSING", "raw_text": raw_text}
            
            lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) > 3]
            matches = []
            for line in lines:
                try:
                    hf_res = requests.get(HF_API_URL, params={"text": line}, timeout=5)
                    data = hf_res.json()
                    if data.get("matched_product") != "No Match":
                        matches.append(data)
                except: continue
            return {"type": "invoice", "matches": matches}
    except Exception as e:
        return {"error": str(e)}

@app.post("/submit-report")
async def submit_report(
    file: UploadFile = File(...),
    outlet_id: str = Form(None),
    outlet_name_manual: str = Form(None),
    user_corrected_usage: float = Form(...)
):
    if not supabase: return {"error": "no_db"}
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        text = pytesseract.image_to_string(adaptive, config="--psm 4")
        audit_used, _, _ = calculate_usage_from_text(text)
        
        filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
        folder = outlet_name_manual if outlet_name_manual else (outlet_id or "Unsorted")
        clean_folder = re.sub(r'[^a-zA-Z0-9_-]', '', folder)
        storage_path = f"{clean_folder}/{filename}"
        
        supabase.storage.from_("Screenshots").upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type}
        )

        data_payload = {
            "outlet_id": outlet_id if outlet_id != "OTHER" else None,
            "outlet_name_manual": outlet_name_manual,
            "week": f"Week {time.strftime('%U')}",
            "ocr_used_gb": audit_used,
            "final_used_gb": user_corrected_usage,
            "verified": True,
            "image_url": storage_path,
            "created_at": time.strftime('%Y-%m-%dT%H:%M:%S')
        }
        supabase.table("quota_reports").insert(data_payload).execute()
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}