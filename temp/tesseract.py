import os
import time
import re
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Gemini OCR Gateway")

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
HF_API_URL = os.environ.get("HF_API_URL")

# Initialize Supabase Client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase connection failed: {e}")

# Enable CORS for frontend/n8n access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared HTTPX client for efficiency (Connection Pooling)
async_client = httpx.AsyncClient(timeout=90.0)

class MatchRequest(BaseModel):
    lines: List[str]

@app.on_event("shutdown")
async def shutdown_event():
    await async_client.aclose()

# --- Gateway Endpoints ---

@app.get("/")
def home():
    return {
        "status": "online",
        "mode": "bridge",
        "brain_target": HF_API_URL,
        "database_connected": supabase is not None
    }

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    task_type: str = Form("invoice")
):
    """
    Bridge: Receives image from n8n and proxies it to Hugging Face Brain.
    """
    if not HF_API_URL:
        raise HTTPException(status_code=500, detail="HF_API_URL is not configured.")

    # Route mapping to Brain endpoints
    endpoint_map = {
        "invoice": "process-invoice",
        "quota": "process-quota",
        "raw_ocr": "get-raw-ocr"
    }
    
    target_path = endpoint_map.get(task_type, "get-raw-ocr")
    target_url = f"{HF_API_URL}/{target_path}"

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        # Forwarding the request to HF
        response = await async_client.post(target_url, files=files)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Brain Communication Error: {str(e)}")

@app.post("/match-items")
async def match_items(data: MatchRequest):
    """
    Bridge: Forwards textual lines to HF for SBERT/Fuzzy matching.
    """
    if not HF_API_URL:
        raise HTTPException(status_code=500, detail="HF_API_URL is not configured.")
    
    try:
        target_url = f"{HF_API_URL}/match-items"
        response = await async_client.post(target_url, json=data.dict())
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Matching Service Error: {str(e)}")

# --- Finalization & Storage ---

@app.post("/submit-report")
async def submit_report(
    file: UploadFile = File(...),
    report_id: str = Form(...),      
    phone_number: str = Form(...),
    outlet_id: Optional[str] = Form(None),
    outlet_name_manual: Optional[str] = Form(None),
    user_corrected_usage: float = Form(...)
):
    """
    Gateway: Handles image archival in Supabase Storage and updates the database.
    """
    if not supabase: 
        return {"status": "error", "message": "Database connection unavailable."}
        
    try:
        content = await file.read()
        
        # 1. Generate clean path
        current_time = time.localtime()
        year = time.strftime('%Y', current_time)
        week = time.strftime('%W', current_time)
        timestamp = int(time.time())
        
        folder_name = outlet_name_manual or outlet_id or "Unsorted"
        clean_folder = re.sub(r'[^a-zA-Z0-9_-]', '', folder_name)
        
        filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        storage_path = f"Quota/{clean_folder}/{year}/Week_{week}/{filename}"
        
        # 2. Upload to Supabase Bucket
        supabase.storage.from_("Screenshots").upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type}
        )
        
        # 3. Get Public URL
        url_resp = supabase.storage.from_("Screenshots").get_public_url(storage_path)
        final_url = url_resp if isinstance(url_resp, str) else url_resp.get("publicUrl")

        # 4. Update Database Record
        data_payload = {
            "image_url": final_url,
            "confirmation": True,                
            "confirmed_at": "now()"
        }
        
        db_response = supabase.table("quota_reports").update(data_payload).eq("id", report_id).execute()
        return {"status": "success", "image_url": final_url, "db_data": db_response.data}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}