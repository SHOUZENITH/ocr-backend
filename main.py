import os
import time
import re
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="OCR Gateway v3")

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

@app.get("/")
def home():
    return {
        "status": "online",
        "version": "v3",
        "mode": "bridge",
        "brain_target": HF_API_URL,
        "database_connected": supabase is not None,
        "standardized_variables": {
            "quota_metrics": ["total", "used", "remaining"],
            "identifiers": ["outlet_code", "user_id", "report_id", "detected_phone"]
        }
    }

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    task_type: str = Form("quota"),
    outlet_code: Optional[str] = Form(None),
    outlet_id: Optional[str] = Form(None)  # Backward-compatible alias
):
    """
    v3: Multi-Provider Gateway
    - Proxies document to HF OCR engine (v3)
    - Receives standardized variables (total, used, remaining, detected_phone)
    - Cross-references with historical reports in Supabase for single-card UIs (e.g. Bima+ Tri)
    """
    if not HF_API_URL:
        raise HTTPException(status_code=500, detail="HF_API_URL is not configured.")

    endpoint_map = {
        "invoice": "process-invoice",
        "quota": "process-quota",
        "raw_ocr": "get-raw-ocr"
    }
    
    target_path = endpoint_map.get(task_type, "get-raw-ocr")
    target_url = f"{HF_API_URL}/{target_path}"
    effective_outlet = outlet_code or outlet_id

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        response = await async_client.post(target_url, files=files)
        api_data = response.json()

        # Anti-cheat & Historical reconciliation in Supabase
        if task_type == "quota" and effective_outlet and supabase:
            try:
                db_res = supabase.table("quota_reports").select(
                    "total, used, remaining, outlet_code, created_at"
                ).eq("outlet_code", effective_outlet).order("created_at", desc=True).limit(1).execute()

                last_report = db_res.data[0] if db_res.data else None

                if last_report:
                    curr_rem = api_data.get("remaining")
                    prev_rem = last_report.get("remaining")
                    prev_tot = last_report.get("total")

                    # If UI only shows remaining quota (e.g. Bima+) and total was not on screen,
                    # infer total from previous report if within reasonable range
                    if api_data.get("used") == 0.0 and prev_tot and curr_rem and prev_tot >= curr_rem:
                        inferred_used = round(prev_tot - curr_rem, 2)
                        api_data["total"] = prev_tot
                        api_data["used"] = inferred_used
                        api_data["total_quota"] = prev_tot
                        api_data["used_quota"] = inferred_used

                    verdict = "APPROVED (Valid quota report)"
                    is_valid = True

                    if prev_rem is not None and curr_rem is not None:
                        if curr_rem > prev_rem:
                            verdict = "APPROVED (New billing cycle / Quota renewed or repurchased)"
                        elif curr_rem == prev_rem:
                            verdict = "WARNING (Identical remaining quota to previous submission)"

                    api_data["anti_cheat"] = {
                        "is_valid": is_valid,
                        "verdict": verdict,
                        "outlet_code": effective_outlet,
                        "previous_remaining": prev_rem,
                        "current_remaining": curr_rem,
                        "detected_phone": api_data.get("detected_phone")
                    }
                else:
                    api_data["anti_cheat"] = {
                        "is_valid": True,
                        "verdict": "APPROVED (First time submission for this outlet)",
                        "detected_phone": api_data.get("detected_phone")
                    }
            except Exception as db_err:
                api_data["anti_cheat"] = {
                    "is_valid": True,
                    "verdict": f"INFO (Database check skipped: {str(db_err)})"
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
    outlet_code: Optional[str] = Form(None),
    outlet_id: Optional[str] = Form(None),               # Backward-compatible alias
    user_id: Optional[str] = Form(None),
    phone_number: Optional[str] = Form(None),          # Backward-compatible alias
    used: Optional[float] = Form(None),
    user_corrected_usage: Optional[float] = Form(None), # Backward-compatible alias
    remaining: Optional[float] = Form(None),
    total: Optional[float] = Form(None)
):
    """
    v3: Stores screenshot in Supabase storage and finalizes quota_reports record.
    Supports both new standardized field names and legacy n8n aliases.
    """
    if not supabase: 
        return {"status": "error", "message": "Database connection unavailable."}
        
    try:
        content = await file.read()
        
        effective_outlet = outlet_code or outlet_id or "Unsorted"
        effective_user = user_id or phone_number or "unknown"
        effective_used = used if used is not None else user_corrected_usage
        
        current_time = time.localtime()
        year = time.strftime('%Y', current_time)
        week = time.strftime('%W', current_time)
        timestamp = int(time.time())
        
        clean_folder = re.sub(r'[^a-zA-Z0-9_-]', '', effective_outlet)
        filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        storage_path = f"Quota/{clean_folder}/{year}/Week_{week}/{filename}"
        
        supabase.storage.from_("Screenshots").upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type}
        )
        
        url_resp = supabase.storage.from_("Screenshots").get_public_url(storage_path)
        final_url = url_resp if isinstance(url_resp, str) else url_resp.get("publicUrl")

        # Standardized payload matching current Supabase schema
        data_payload = {
            "image_url": final_url,
            "confirmation": True,                
            "confirmed_at": "now()"
        }
        
        if effective_used is not None:
            data_payload["used"] = effective_used
        if remaining is not None:
            data_payload["remaining"] = remaining
        if total is not None:
            data_payload["total"] = total
        if effective_outlet != "Unsorted":
            data_payload["outlet_code"] = effective_outlet
            
        db_response = supabase.table("quota_reports").update(data_payload).eq("id", report_id).execute()
        
        return {
            "status": "success", 
            "image_url": final_url, 
            "report_id": report_id,
            "outlet_code": effective_outlet,
            "used": effective_used,
            "remaining": remaining,
            "total": total,
            "db_data": db_response.data
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
