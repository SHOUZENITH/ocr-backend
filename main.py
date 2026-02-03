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

# --- 1. CONFIGURATION ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# Initialize Supabase
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase Connected")
    except Exception as e:
        print(f"❌ Supabase Connection Failed: {e}")

# CORS Config (Open for testing, restrict for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. ADVANCED LOGIC ENGINE ---

def parse_size_to_gb(value_str, unit_str="GB"):
    """
    Converts "500 MB", "2.5 GB", "1024 KB" to a float GB value.
    """
    try:
        val = float(value_str.replace(',', '.'))
        unit = unit_str.upper().strip()
        
        if "MB" in unit:
            return val / 1024.0
        elif "KB" in unit:
            return val / (1024.0 * 1024.0)
        return val # Default is GB
    except ValueError:
        return 0.0

def calculate_usage_from_text(text):
    """
    The 'Smart Sisa' Engine.
    Prioritizes slash patterns (Remaining/Total) and 'Sisa' keywords.
    """
    clean_text = text.lower().replace(',', '.')
    
    # Includes common OCR typos for "Sisa" (Slsa, S1sa)
    sisa_keywords = ['sisa', 'slsa', 's1sa', 'rem', 'left', 'kuota']
    
    total_used = 0.0
    total_remaining = 0.0
    
    # Most accurate. Logic: If N1 < N2, then N1 is Remaining.
    slash_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)?\s*[\/|]\s*(\d+(?:\.\d+)?)\s*(?:gb|mb)'
    slash_matches = re.findall(slash_pattern, clean_text)
    
    if slash_matches:
        valid_pair = False
        for val1, val2 in slash_matches:
            try:
                n1 = float(val1)
                n2 = float(val2)
                
                # Sanity Check: Total (n2) must be bigger than Remaining (n1)
                # And ignore years (e.g. 2026)
                if n1 < n2 and n2 < 2000:
                    used = n2 - n1
                    total_used += used
                    total_remaining += n1
                    valid_pair = True
            except:
                continue
        
        if valid_pair:
            return round(total_used, 2), round(total_remaining, 2), "Strategy 1 (Slash Pairs)"

    # Looks for "Terpakai: 5 GB"
    used_pattern = r'(?:terpakai|used|pemakaian|usage).*?(\d+(?:\.\d+)?)\s*(gb|mb)'
    used_matches = re.findall(used_pattern, clean_text)
    if used_matches:
        explicit_used = 0.0
        for val, unit in used_matches:
            explicit_used += parse_size_to_gb(val, unit)
        
        if explicit_used > 0:
            return round(explicit_used, 2), 0.0, "Strategy 2 (Explicit Used)"

    # Looks for any GB/MB numbers
    gb_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)'
    matches = re.findall(gb_pattern, clean_text)
    
    # Filter out years (like 2026) and tiny noise
    values = []
    for m in matches:
        try:
            v = float(m)
            if v < 2000: values.append(v)
        except: pass
        
    if not values:
        return 0.0, 0.0, "No Data Found"


    has_sisa = any(k in clean_text for k in sisa_keywords)

    if len(values) >= 2:
        # Multiple numbers: Assume Max is Total, Min is Remaining
        total = max(values)
        rem = min(values)
        used = total - rem
        return round(used, 2), round(rem, 2), f"Strategy 3 (Max-Min)"
    
    elif len(values) == 1:
        val = values[0]
        if has_sisa:
            # We found "Sisa 37.41 GB". Used is unknown (set to 0 for user to fill), NOT 37.41.
            return 0.0, round(val, 2), "Strategy 3 (Single Remaining)"
        else:
            # Only assume Usage if no "Sisa" keyword exists
            return round(val, 2), 0.0, "Strategy 3 (Single Usage)"

    return 0.0, 0.0, "Failed"

# --- 3. ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "OCR Service Operational", "logic": "Smart Sisa Detection"}

@app.post("/preview-ocr")
async def preview_ocr(
    file: UploadFile = File(...),
    content_length: int = Header(None)
):
    """
    Step 1: Analyzes image and tells Frontend what it sees.
    Does NOT save to database yet.
    """
    if content_length and content_length > 5 * 1024 * 1024:
        raise HTTPException(413, "File too large (Max 5MB)")

    content = await file.read()
    
    # Image Pre-processing
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray) # Contrast boost
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # OCR
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        
        # Logic
        used, remaining, method = calculate_usage_from_text(text)
        
        return {
            "used": used,
            "remaining": remaining,
            "debug_method": method,
            "debug_text": text[:100]
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
    """
    Step 2: Securely saves the report.
    """
    if not supabase:
        raise HTTPException(500, "Database Not Configured")

    content = await file.read()
    
    # 1. Audit Check (Silent Re-Run)
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        audit_used, _, _ = calculate_usage_from_text(text)
    except:
        audit_used = 0.0

    # 2. Upload to Storage
    try:
        filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
        folder = outlet_name_manual if outlet_name_manual else (outlet_id or "Unsorted")
        clean_folder = re.sub(r'[^a-zA-Z0-9_-]', '', folder) # Sanitize
        storage_path = f"{clean_folder}/{filename}"
        
        supabase.storage.from_("Screenshots").upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type}
        )
    except Exception as e:
        print(f"Storage Upload Failed: {e}")
        storage_path = "error_upload_failed.jpg"

    # 3. Save to Database
    try:
        data_payload = {
            "outlet_id": outlet_id if outlet_id != "OTHER" else None,
            "outlet_name_manual": outlet_name_manual,
            "week": f"Week {time.strftime('%U')}",
            "ocr_used_gb": audit_used,            # Audit Trail
            "final_used_gb": user_corrected_usage, # Billing Amount
            "verified": True,
            "image_url": storage_path
        }
        
        supabase.table("quota_reports").insert(data_payload).execute()
        
        return {"status": "success", "data": data_payload}
        
    except Exception as e:
        raise HTTPException(500, f"DB Error: {str(e)}")