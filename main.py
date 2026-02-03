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

# 1. SETUP: Supabase Connection (Server-Side)
# These will be set in your Render Dashboard Environment Variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # Use Service Role Key for full access

# Initialize client only if keys exist (prevents crash during build)
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. CORS: Restrict access
origins = [
    "http://localhost:3000",
    "https://quota-report.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Secure OCR Backend Live"}

@app.post("/submit-report")
async def submit_report(
    file: UploadFile = File(...),
    outlet_id: str = Form(None),       # Receive Outlet ID
    outlet_name_manual: str = Form(None), # Receive Manual Name
    content_length: int = Header(None)
):
    # --- SECURITY CHECKS ---
    if not supabase:
        raise HTTPException(500, "Server DB not configured")
        
    if content_length and content_length > 5 * 1024 * 1024:
        raise HTTPException(413, "File too large (Max 5MB)")

    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Invalid file type. Only JPG/PNG allowed")

    # --- PROCESS IMAGE ---
    try:
        content = await file.read()
        
        # Verify Image
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img.verify()
            img = Image.open(io.BytesIO(content)).convert("RGB") # Re-open
        except:
            raise HTTPException(400, "Corrupted image file")

        # OCR Logic (Notebook Style)
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        config = "--psm 6"
        text = pytesseract.image_to_string(thresh, config=config)
        
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', text, re.IGNORECASE)
        values = [float(m) for m in matches]
        
        used = 0.0
        remaining = 0.0
        
        if len(values) >= 2:
            total = max(values)
            remaining = min(values)
            used = total - remaining
        elif len(values) == 1:
            used = values[0]

        # --- DATABASE & STORAGE SAVE ---
        
        # 1. Upload Image to Supabase Storage
        filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
        folder = outlet_name_manual if outlet_name_manual else outlet_id
        storage_path = f"{folder}/{filename}"
        
        # Reset pointer to start of file for upload
        file_obj = io.BytesIO(content)
        
        # Upload
        supabase.storage.from_("Screenshots").upload(
            path=storage_path,
            file=file_obj.read(),
            file_options={"content-type": file.content_type}
        )

        # 2. Insert Data into DB
        # The backend acts as the source of truth for "ocr_used_gb"
        data_payload = {
            "outlet_id": outlet_id if outlet_id != "OTHER" else None,
            "outlet_name_manual": outlet_name_manual,
            "week": f"Week {time.strftime('%U')}",
            "ocr_used_gb": round(used, 2),
            "final_used_gb": round(used, 2), # Auto-verified by system
            "verified": True,
            "image_url": storage_path
        }
        
        db_res = supabase.table("quota_reports").insert(data_payload).execute()

        return {
            "status": "success", 
            "message": "Report secured and saved.",
            "data": data_payload
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")