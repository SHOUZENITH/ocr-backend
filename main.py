import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import pytesseract
import cv2
import numpy as np
import re
from PIL import Image
import io
import time

app = FastAPI()

# --- SETUP ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

origins = ["http://localhost:3000", "https://quota-report.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# --- HELPER: The Logic from your Notebook ---
def run_ocr_logic(content):
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', text, re.IGNORECASE)
    values = [float(m) for m in matches]
    
    used = 0.0
    remaining = 0.0
    
    if len(values) >= 2:
        used = max(values) - min(values)
        remaining = min(values)
    elif len(values) == 1:
        used = values[0]
        
    return round(used, 2), round(remaining, 2)

# --- ENDPOINT 1: PREVIEW (Just shows the number, doesn't save) ---
@app.post("/preview-ocr")
async def preview_ocr(file: UploadFile = File(...)):
    content = await file.read()
    used, remaining = run_ocr_logic(content)
    return {"used": used, "remaining": remaining}

# --- ENDPOINT 2: SECURE SUBMIT (Saves & Audits) ---
@app.post("/submit-report")
async def submit_report(
    file: UploadFile = File(...),
    outlet_id: str = Form(None),
    outlet_name_manual: str = Form(None),
    user_corrected_usage: float = Form(...) # <--- The number the USER typed
):
    if not supabase: raise HTTPException(500, "DB Config Missing")
    
    # 1. Re-Verify Image Integrity
    content = await file.read()
    
    # 2. Re-Run OCR (The "Audit" Check)
    # We do this secretly to compare against what the user typed
    system_detected_usage, _ = run_ocr_logic(content)

    # 3. Upload Image
    filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
    folder = outlet_name_manual if outlet_name_manual else outlet_id
    storage_path = f"{folder}/{filename}"
    
    supabase.storage.from_("Screenshots").upload(
        path=storage_path,
        file=content,
        file_options={"content-type": file.content_type}
    )

    # 4. Save to DB (Recording BOTH numbers)
    data_payload = {
        "outlet_id": outlet_id if outlet_id != "OTHER" else None,
        "outlet_name_manual": outlet_name_manual,
        "week": f"Week {time.strftime('%U')}",
        "ocr_used_gb": system_detected_usage,   # What the Machine saw
        "final_used_gb": user_corrected_usage,  # What the Human typed
        "verified": True,
        "image_url": storage_path
    }
    
    supabase.table("quota_reports").insert(data_payload).execute()

    return {"status": "success", "data": data_payload}