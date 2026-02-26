import os
import time
import re
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="OCR Gateway")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
HF_API_URL = os.environ.get("HF_API_URL")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase connection failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async_client = httpx.AsyncClient(timeout=90.0)

class MatchRequest(BaseModel):
    lines: List[str]

@app.on_event("shutdown")
async def shutdown_event():
    await async_client.aclose()

def calculate_hex_distance(hash1_hex: str, hash2_hex: str) -> int:
    if not hash1_hex or not hash2_hex: 
        return 100
    try:
        bin1 = bin(int(hash1_hex, 16))[2:].zfill(64)
        bin2 = bin(int(hash2_hex, 16))[2:].zfill(64)
        return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
    except ValueError:
        return 100

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
    task_type: str = Form("invoice"),
    outlet_id: Optional[str] = Form(None)
):
    if not HF_API_URL:
        raise HTTPException(status_code=500, detail="HF_API_URL is not configured.")

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
        
        response = await async_client.post(target_url, files=files)
        api_data = response.json()

        if task_type == "quota" and outlet_id and supabase:
            db_res = supabase.table("quota_reports").select(
                "phash, md5, rem_gb, expiry_date"
            ).eq("outlet_id", outlet_id).order("created_at", desc=True).limit(1).execute()

            last_report = db_res.data[0] if db_res.data else None

            if last_report:
                api_exp = api_data.get("expiry_date")
                api_quo = api_data.get("remaining_quota")
                db_exp = last_report.get("expiry_date")
                db_quo = last_report.get("rem_gb") 
                distance = calculate_hex_distance(api_data.get("phash"), last_report.get("phash"))

                verdict = "APPROVED (Valid weekly quota update)"
                is_valid = True

                if api_data.get("md5") == last_report.get("md5"):
                    verdict, is_valid = "REJECTED (Layer 1: Exact file duplicate uploaded)", False
                elif db_exp != "Unknown" and api_exp != "Unknown" and db_exp != api_exp:
                    verdict, is_valid = "APPROVED (New billing cycle / Quota repurchased)", True
                elif db_exp == api_exp and api_quo > db_quo:
                    verdict, is_valid = "REJECTED (Layer 4: Logical Error - Quota increased without renewal)", False
                elif db_exp == api_exp and api_quo == db_quo:
                    verdict, is_valid = "REJECTED (Layer 4: Exact same quota numbers. Cropped or stale duplicate detected)", False
                elif distance <= 8:
                    verdict, is_valid = "REJECTED (Layer 2: Structural layout is identical. Manual Photoshop detected)", False

                api_data["anti_cheat"] = {
                    "is_valid": is_valid,
                    "verdict": verdict,
                    "visual_distance": distance,
                    "previous_quota": db_quo,
                    "previous_expiry": db_exp
                }
            else:
                api_data["anti_cheat"] = {
                    "is_valid": True,
                    "verdict": "APPROVED (First time submission)"
                }

        return api_data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Brain Communication Error: {str(e)}")

@app.post("/match-items")
async def match_items(data: MatchRequest):
    if not HF_API_URL:
        raise HTTPException(status_code=500, detail="HF_API_URL is not configured.")
    
    try:
        target_url = f"{HF_API_URL}/match-items"
        response = await async_client.post(target_url, json=data.dict())
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Matching Service Error: {str(e)}")

@app.post("/submit-report")
async def submit_report(
    file: UploadFile = File(...),
    report_id: str = Form(...),      
    phone_number: str = Form(...),
    outlet_id: Optional[str] = Form(None),
    outlet_name_manual: Optional[str] = Form(None),
    user_corrected_usage: float = Form(...)
):
    if not supabase: 
        return {"status": "error", "message": "Database connection unavailable."}
        
    try:
        content = await file.read()
        
        current_time = time.localtime()
        year = time.strftime('%Y', current_time)
        week = time.strftime('%W', current_time)
        timestamp = int(time.time())
        
        folder_name = outlet_name_manual or outlet_id or "Unsorted"
        clean_folder = re.sub(r'[^a-zA-Z0-9_-]', '', folder_name)
        
        filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        storage_path = f"Quota/{clean_folder}/{year}/Week_{week}/{filename}"
        
        supabase.storage.from_("Screenshots").upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type}
        )
        
        url_resp = supabase.storage.from_("Screenshots").get_public_url(storage_path)
        final_url = url_resp if isinstance(url_resp, str) else url_resp.get("publicUrl")

        data_payload = {
            "image_url": final_url,
            "confirmation": True,                
            "confirmed_at": "now()"
        }
        
        db_response = supabase.table("quota_reports").update(data_payload).eq("id", report_id).execute()
        return {"status": "success", "image_url": final_url, "db_data": db_response.data}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}