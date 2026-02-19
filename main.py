import os
import time
import re
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/")
def home():
    return {"status": "active", "mode": "bridge"}

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(None),
    task_type: str = Form("invoice")
):
    try:
        if not HF_API_URL:
            return {"error": "HF_API_URL_MISSING"}

        if not file:
            return {"error": "No file provided"}

        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        if task_type == "invoice":
            target_url = f"{HF_API_URL}/process-invoice"
            
        elif task_type == "quota":
            target_url = f"{HF_API_URL}/process-quota"
            
        elif task_type == "raw_ocr":
            target_url = f"{HF_API_URL}/process-ocr"
            
        else:
            target_url = f"{HF_API_URL}/process-ocr"
        
        hf_res = requests.post(target_url, files=files, timeout=30)
        
        return hf_res.json()

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
    if not supabase: 
        return {"status": "error", "message": "Database not connected"}
        
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
        return {"status": "error", "message": str(e)}