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
        print("✅ Supabase Connected")
    except Exception as e:
        print(f"❌ Supabase Connection Failed: {e}")

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
        elif "KB" in unit: return val / (1024.0 * 1024.0)
        return val
    except ValueError:
        return 0.0

def calculate_usage_from_text(text):
    clean_text = text.lower().replace(',', '.')
    
    is_remaining_context = bool(re.search(r's[i1l]sa|rem|left|kuota', clean_text))
    
    total_used = 0.0
    total_remaining = 0.0
    
    slash_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)?\s*[\\\/|1lI]\s*(\d+(?:\.\d+)?)\s*(?:gb|mb)'
    slash_matches = re.findall(slash_pattern, clean_text)
    
    if slash_matches:
        valid_pair = False
        for val1, val2 in slash_matches:
            try:
                n1 = float(val1)
                n2 = float(val2)
                if n1 < n2 and n2 < 2000:
                    used = n2 - n1
                    total_used += used
                    total_remaining += n1
                    valid_pair = True
            except: continue
        
        if valid_pair:
            return round(total_used, 2), round(total_remaining, 2), "Strategy 1 (Slash Pairs)"

    used_pattern = r'(?:terpakai|used|pemakaian|usage).*?(\d+(?:\.\d+)?)\s*(gb|mb)'
    used_matches = re.findall(used_pattern, clean_text)
    if used_matches:
        explicit_used = 0.0
        for val, unit in used_matches:
            explicit_used += parse_size_to_gb(val, unit)
        if explicit_used > 0:
            return round(explicit_used, 2), 0.0, "Strategy 2 (Explicit Used)"

    gb_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)'
    matches = re.findall(gb_pattern, clean_text)
    
    values = []
    for m in matches:
        try:
            v = float(m)
            if v < 2000 and v > 0.01: values.append(v)
        except: pass
        
    if not values:
        return 0.0, 0.0, "No Data Found"

    if len(values) >= 2:
        total = max(values)
        rem = min(values)
        used = total - rem
        return round(used, 2), round(rem, 2), "Strategy 3 (Max-Min)"
    
    elif len(values) == 1:
        val = values[0]
        if is_remaining_context:
            return 0.0, round(val, 2), "Strategy 3 (Single Remaining)"
        else:
            return round(val, 2), 0.0, "Strategy 3 (Single Usage)"

    return 0.0, 0.0, "Failed"

@app.get("/")
def home():
    return {"status": "OCR Service Operational", "mode": "Winner Takes All"}

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
        img_np = np.array(img)
        
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        
        inverted = cv2.bitwise_not(adaptive)
        
        gray_eq = cv2.equalizeHist(gray)
        _, standard = cv2.threshold(gray_eq, 150, 255, cv2.THRESH_BINARY)
        
        config = "--psm 6"
        results = []
        
        t1 = pytesseract.image_to_string(adaptive, config=config)
        results.append(calculate_usage_from_text(t1))
        
        t2 = pytesseract.image_to_string(inverted, config=config)
        results.append(calculate_usage_from_text(t2))
        
        t3 = pytesseract.image_to_string(standard, config=config)
        results.append(calculate_usage_from_text(t3))
        
        best_result = max(results, key=lambda x: (x[0], x[1]))
        
        used, remaining, method = best_result
        
        return {
            "used": used,
            "remaining": remaining,
            "debug_method": method,
            "debug_text_snippet": t1[:50] 
        }
    except Exception as e:
        return {"error": str(e), "used": 0, "remaining": 0}

@app.post("/submit-report")
async def submit_report(
    file: UploadFile = File(...),
    outlet_id: str = Form(None),
    outlet_name_manual: str = Form(None),
    user_corrected_usage: float = Form(...)
):
    if not supabase: raise HTTPException(500, "Database Not Configured")

    content = await file.read()
    
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        text = pytesseract.image_to_string(adaptive, config="--psm 6")
        
        audit_used, _, _ = calculate_usage_from_text(text)
    except:
        audit_used = 0.0

    try:
        filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
        folder = outlet_name_manual if outlet_name_manual else (outlet_id or "Unsorted")
        clean_folder = re.sub(r'[^a-zA-Z0-9_-]', '', folder)
        storage_path = f"{clean_folder}/{filename}"
        
        supabase.storage.from_("Screenshots").upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type}
        )
    except Exception as e:
        print(f"Storage Error: {e}")
        storage_path = "error_upload_failed.jpg"

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
        return {"status": "success", "data": data_payload}
        
    except Exception as e:
        raise HTTPException(500, f"DB Error: {str(e)}")