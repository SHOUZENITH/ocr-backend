import os
import time
import io
import re
import numpy as np
import cv2
import pytesseract
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from supabase import create_client, Client

app = FastAPI()

# --- 1. CONFIGURATION ---
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

# --- 2. LOGIC ENGINE ---

def parse_size_to_gb(value_str, unit_str="GB"):
    """Converts MB/KB strings to GB floats."""
    try:
        val = float(value_str.replace(',', '.'))
        unit = unit_str.upper().strip()
        if "MB" in unit: return val / 1024.0
        if "KB" in unit: return val / (1024.0 * 1024.0)
        return val
    except ValueError:
        return 0.0

def calculate_usage_from_text(text):
    """
    Analyzes text to find data usage.
    Returns: (used_gb, remaining_gb, detection_method)
    """
    clean_text = text.lower().replace(',', '.')
    
    # Context Check: Is this describing remaining data?
    is_remaining_context = bool(re.search(r's[i1l]sa|rem|left|kuota|bal', clean_text))

    # STRATEGY 1: SLASH PATTERN (Highest Confidence)
    # Looks for: "21.98 GB / 28 GB"
    slash_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)?\s*[\\\/|1lI]\s*(\d+(?:\.\d+)?)\s*(?:gb|mb)'
    slash_matches = re.findall(slash_pattern, clean_text)
    
    if slash_matches:
        for val1, val2 in slash_matches:
            try:
                n1 = float(val1)
                n2 = float(val2)
                # n1 is usually Remaining, n2 is Total. 
                # Constraint: Total must be realistic (< 2000 GB)
                if n1 < n2 and n2 < 2000:
                    used = n2 - n1
                    return round(used, 2), round(n1, 2), "Strategy 1 (Slash Logic)"
            except: continue

    # STRATEGY 2: EXPLICIT KEYWORDS (Medium Confidence)
    # Looks for: "Used 15 GB" or "Terpakai 2 GB"
    used_pattern = r'(?:terpakai|used|pemakaian|usage).*?(\d+(?:\.\d+)?)\s*(gb|mb)'
    used_matches = re.findall(used_pattern, clean_text)
    if used_matches:
        explicit_used = 0.0
        for val, unit in used_matches:
            explicit_used += parse_size_to_gb(val, unit)
        if explicit_used > 0:
            return round(explicit_used, 2), 0.0, "Strategy 2 (Explicit Keyword)"

    # STRATEGY 3: NUMBER ANALYSIS (Fallback)
    gb_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)'
    matches = re.findall(gb_pattern, clean_text)
    
    values = []
    for m in matches:
        try:
            v = float(m)
            # Filter: Ignore years (2024-2030) and tiny noise
            if 0.01 < v < 2000: 
                values.append(v)
        except: pass
        
    if not values:
        return 0.0, 0.0, "No Data Found"

    if len(values) >= 2:
        # Assume Max = Total, Min = Remaining
        total = max(values)
        rem = min(values)
        used = total - rem
        if used == 0: return 0.0, total, "Strategy 3 (Duplicate Read)"
        return round(used, 2), round(rem, 2), "Strategy 3 (Max-Min Calc)"
    
    elif len(values) == 1:
        val = values[0]
        if is_remaining_context:
            return 0.0, round(val, 2), "Strategy 3 (Single Remaining)"
        else:
            return round(val, 2), 0.0, "Strategy 3 (Single Usage - Risky)"

    return 0.0, 0.0, "Failed"

def get_strategy_score(method_name):
    """Higher score = More reliable method."""
    if "Strategy 1" in method_name: return 10  # Best: Found "X / Y"
    if "Strategy 2" in method_name: return 8   # Good: Found "Used X"
    if "Max-Min" in method_name: return 6      # Okay: Found two numbers
    if "Single Remaining" in method_name: return 5 # Okay: Found "Sisa X"
    if "Single Usage" in method_name: return 1 # Bad: Just guessing
    return 0

# --- 3. ENDPOINTS ---

@app.head("/") # Fix for UptimeRobot
@app.get("/")
def home():
    return {"status": "OCR Service Operational", "version": "5.0 (Smart Scorer)"}

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
        
        # Create 3 Versions of the image to catch different visual styles
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        inverted = cv2.bitwise_not(adaptive)
        gray_eq = cv2.equalizeHist(gray)
        _, standard = cv2.threshold(gray_eq, 150, 255, cv2.THRESH_BINARY)
        
        config = "--psm 6"
        results_list = []
        
        # Run OCR on all 3 versions
        filters = [("adaptive", adaptive), ("inverted", inverted), ("standard", standard)]
        
        for name, processed_img in filters:
            raw_text = pytesseract.image_to_string(processed_img, config=config)
            used, rem, method = calculate_usage_from_text(raw_text)
            score = get_strategy_score(method)
            
            results_list.append({
                "used": used,
                "remaining": rem,
                "method": method,
                "score": score,
                "text": raw_text[:50].replace('\n', ' ') 
            })

        # WINNER SELECTION:
        # Sort by Score (High to Low), then by Used Amount (High to Low)
        results_list.sort(key=lambda x: (x['score'], x['used']), reverse=True)
        best = results_list[0]
        
        return {
            "used": best["used"],
            "remaining": best["remaining"],
            "debug_method": best["method"],
            "debug_score": best["score"],
            "debug_text": best["text"]
        }

    except Exception as e:
        print(f"OCR Error: {e}")
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
    
    # Quick Audit Scan (Adaptive only)
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        text = pytesseract.image_to_string(adaptive, config="--psm 6")
        audit_used, _, _ = calculate_usage_from_text(text)
    except:
        audit_used = 0.0

    # Upload Image
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
    except Exception:
        storage_path = "error_upload_failed.jpg"

    # Save Data
    try:
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
        return {"status": "success", "data": data_payload}
        
    except Exception as e:
        raise HTTPException(500, f"DB Error: {str(e)}")