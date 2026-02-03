import os
import time
import io
import re
import numpy as np
import cv2
import pytesseract
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from supabase import create_client, Client

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase Connected Successfully")
    except Exception as e:
        print(f"❌ Supabase Connection Failed: {e}")

origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

def parse_size_to_gb(value_str, unit_str="GB"):
    try:
        val = float(value_str.replace(',', '.'))
        unit = unit_str.upper().strip()
        
        if "MB" in unit:
            return val / 1024.0
        elif "KB" in unit:
            return val / (1024.0 * 1024.0)
        return val
    except ValueError:
        return 0.0

def calculate_usage_from_text(text):
    clean_text = text.replace(',', '.').lower()
    
    total_used = 0.0
    total_remaining = 0.0
    
    used_pattern = r'(?:terpakai|used|penggunaan|pemakaian).*?(\d+(?:\.\d+)?)\s*(gb|mb|kb)'
    used_matches = re.findall(used_pattern, clean_text)
    
    if used_matches:
        found_valid = False
        for val, unit in used_matches:
            gb_val = parse_size_to_gb(val, unit)
            if gb_val < 1000:
                total_used += gb_val
                found_valid = True
        
        if found_valid:
            return round(total_used, 2), 0.0, "Strategy 1 (Explicit Labels)"

    slash_pattern = r'(\d+(?:\.\d+)?)\s*(gb|mb|kb)?\s*\/\s*(\d+(?:\.\d+)?)\s*(gb|mb|kb)'
    slash_matches = re.findall(slash_pattern, clean_text)
    
    if slash_matches:
        s2_used = 0.0
        s2_rem = 0.0
        valid_pair = False
        
        for rem_val, rem_unit, tot_val, tot_unit in slash_matches:
            if not rem_unit: rem_unit = tot_unit
            
            rem_gb = parse_size_to_gb(rem_val, rem_unit)
            tot_gb = parse_size_to_gb(tot_val, tot_unit)
            
            if tot_gb >= rem_gb and tot_gb < 1000:
                s2_used += (tot_gb - rem_gb)
                s2_rem += rem_gb
                valid_pair = True
        
        if valid_pair:
            return round(s2_used, 2), round(s2_rem, 2), "Strategy 2 (Slash Pairs)"

    gb_pattern = r'(\d+(?:\.\d+)?)\s*(gb)'
    matches = re.findall(gb_pattern, clean_text)
    
    values = []
    for val, unit in matches:
        v = float(val)
        if v < 1000: values.append(v)
        
    if len(values) >= 2:
        used = max(values) - min(values)
        rem = min(values)
        return round(used, 2), round(rem, 2), "Strategy 3 (Max - Min)"
    elif len(values) == 1:
        val = values[0]
        if "sisa" in clean_text or "rem" in clean_text:
            return 0.0, round(val, 2), "Strategy 3 (Single Remaining)"
        return round(val, 2), 0.0, "Strategy 3 (Single Usage)"

    return 0.0, 0.0, "No Data Found"

@app.get("/")
def home():
    return {"status": "OCR Service Operational", "mode": "Secure"}

@app.post("/preview-ocr")
async def preview_ocr(
    file: UploadFile = File(...),
    content_length: int = Header(None)
):
    if content_length and content_length > 5 * 1024 * 1024:
        raise HTTPException(413, "File too large (Max 5MB)")

    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Invalid Image File")

    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    
    used, remaining, method = calculate_usage_from_text(text)
    
    return {
        "used": used,
        "remaining": remaining,
        "strategy": method,
        "debug_text_snippet": text[:100]
    }

@app.post("/submit-report")
async def submit_report(
    file: UploadFile = File(...),
    outlet_id: str = Form(None),
    outlet_name_manual: str = Form(None),
    user_corrected_usage: float = Form(...)
):
    if not supabase:
        raise HTTPException(500, "Server Error: Database key not configured.")

    content = await file.read()
    
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        
        audit_used, _, _ = calculate_usage_from_text(text)
    except Exception:
        audit_used = 0.0

    try:
        filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
        folder = outlet_name_manual if outlet_name_manual else (outlet_id or "Unsorted")
        folder = re.sub(r'[^a-zA-Z0-9_-]', '', folder) 
        
        storage_path = f"{folder}/{filename}"
        
        supabase.storage.from_("Screenshots").upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type}
        )
    except Exception as e:
        print(f"Storage Error: {e}")
        storage_path = "upload_failed.jpg"

    try:
        data_payload = {
            "outlet_id": outlet_id if outlet_id != "OTHER" else None,
            "outlet_name_manual": outlet_name_manual,
            "week": f"Week {time.strftime('%U')}",
            "ocr_used_gb": audit_used,
            "final_used_gb": user_corrected_usage,
            "verified": True,
            "image_url": storage_path
        }
        
        supabase.table("quota_reports").insert(data_payload).execute()
        
        return {
            "status": "success",
            "message": "Report secured.",
            "data": data_payload
        }
    except Exception as e:
        raise HTTPException(500, f"Database Save Failed: {str(e)}")