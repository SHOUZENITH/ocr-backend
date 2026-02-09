import os
import time
import io
import re
import numpy as np
import cv2
import pytesseract
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process

app = FastAPI()

# Database Config
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Connection Error: {e}")

# Global Variables
nlp_model = None
MASTER_PRODUCTS = []
MASTER_EMBEDDINGS = None

def fetch_master_products():
    """Retrieve product list from supabase"""
    if not supabase:
        return ["Produk Contoh A", "Produk Contoh B"]
    
    try:
        response = supabase.table("master_products").select("name").execute()
        return [item['name'] for item in response.data]
    except Exception as e:
        print(f"Fetch Error: {e}")
        return []

@app.on_event("startup")
async def startup_event():
    """Initialize model and product embeddings on startup"""
    global nlp_model, MASTER_PRODUCTS, MASTER_EMBEDDINGS
    
    try:
        nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        MASTER_PRODUCTS = fetch_master_products()
        
        if MASTER_PRODUCTS and nlp_model:
            MASTER_EMBEDDINGS = nlp_model.encode(MASTER_PRODUCTS, convert_to_tensor=True)
    except Exception as e:
        print(f"Startup Error: {e}")

# Quota OCR Logic
def parse_size_to_gb(value_str, unit_str="GB"):
    """Convert size strings to float GB"""
    try:
        val = float(value_str.replace(',', '.'))
        unit = unit_str.upper().strip()
        if "MB" in unit: return val / 1024.0
        if "KB" in unit: return val / (1024.0 * 1024.0)
        return val
    except ValueError:
        return 0.0

def calculate_usage_from_text(text):
    """Analyze text for data usage metrics"""
    clean_text = text.lower().replace(',', '.')
    
    # Slash Format Strategy
    slash_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)?\s*[\\\/|1lI]\s*(\d+(?:\.\d+)?)\s*(?:gb|mb)?'
    slash_matches = re.finditer(slash_pattern, clean_text)
    valid_candidates = []
    for match in slash_matches:
        full_str = match.group(0)
        val1, val2 = match.group(1), match.group(2)
        if not re.search(r'[gm]', full_str): continue
        try:
            n1, n2 = float(val1), float(val2)
            if n1 <= n2 and n2 < 5000:
                valid_candidates.append({'used': round(n2 - n1, 2), 'rem': round(n1, 2), 'total': n2})
        except: continue
    
    if valid_candidates:
        best = max(valid_candidates, key=lambda x: x['total'])
        return best['used'], best['rem'], "Slash Format"

    # Explicit Keyword Strategy
    used_pattern = r'(?:terpakai|used|pemakaian|usage).*?(\d+(?:\.\d+)?)\s*(gb|mb)'
    used_matches = re.findall(used_pattern, clean_text)
    if used_matches:
        explicit_used = sum(parse_size_to_gb(val, unit) for val, unit in used_matches)
        if explicit_used > 0:
            return round(explicit_used, 2), 0.0, "Explicit Keyword"

    # Heuristic Strategy
    gb_pattern = r'(\d+(?:\.\d+)?)\s*(?:gb|mb)'
    values = []
    for m in re.findall(gb_pattern, clean_text):
        try:
            v = float(m)
            if 0.01 < v < 2000: values.append(v)
        except: pass
        
    if not values: return 0.0, 0.0, "No Data"

    if len(values) >= 2:
        total, rem = max(values), min(values)
        return round(total - rem, 2), round(rem, 2), "Max-Min Calc"
    elif len(values) == 1:
        if re.search(r's[i1l]sa|rem|left|kuota|bal', clean_text):
            return 0.0, round(values[0], 2), "Single Remaining"
        return round(values[0], 2), 0.0, "Single Usage"
            
    return 0.0, 0.0, "Failed"

# Invoice NLP Logic
def match_product_name(ocr_text_line: str):
    """Hybrid matching using Fuzzy String and Semantic Similarity"""
    if nlp_model is None or MASTER_EMBEDDINGS is None or not MASTER_PRODUCTS:
        return None
    
    # Semantic Search
    input_embedding = nlp_model.encode(ocr_text_line, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, MASTER_EMBEDDINGS)
    sbert_idx = torch.argmax(cosine_scores).item()
    sbert_score = cosine_scores[0][sbert_idx].item()
    
    # Fuzzy Search
    fuzzy_res = process.extractOne(ocr_text_line, MASTER_PRODUCTS, scorer=fuzz.token_set_ratio)
    if not fuzzy_res: return None
    
    fuzzy_match, fuzzy_score = fuzzy_res[0], fuzzy_res[1] / 100
    
    # Hybrid Calculation (70% Fuzzy, 30% SBERT)
    final_score = (fuzzy_score * 0.7) + (sbert_score * 0.3)
    
    if final_score > 0.50:
        return {
            "original_text": ocr_text_line,
            "matched_product": fuzzy_match,
            "confidence": round(final_score, 2),
            "match_type": "Hybrid"
        }
    return None

# Endpoints
@app.get("/")
def home():
    return {
        "status": "Operational",
        "products_loaded": len(MASTER_PRODUCTS),
        "ai_ready": nlp_model is not None
    }

@app.post("/refresh-products")
def refresh_products():
    """Update local product cache from database"""
    global MASTER_PRODUCTS, MASTER_EMBEDDINGS
    new_products = fetch_master_products()
    if not new_products:
        return {"status": "failed"}
        
    MASTER_PRODUCTS = new_products
    if nlp_model:
        MASTER_EMBEDDINGS = nlp_model.encode(MASTER_PRODUCTS, convert_to_tensor=True)
    return {"status": "success", "total_products": len(MASTER_PRODUCTS)}

@app.post("/preview-ocr")
async def preview_ocr(file: UploadFile = File(...)):
    """Extract quota data from screenshot"""
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    _, standard = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(standard, config="--psm 6")
    
    used, rem, method = calculate_usage_from_text(text)
    return {"used": used, "remaining": rem, "method": method}

@app.post("/scan-invoice")
async def scan_invoice(file: UploadFile = File(...)):
    """Extract and match product items from physical receipt"""
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Adaptive preprocessing for textured paper
    processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    raw_text = pytesseract.image_to_string(processed_img, config="--psm 6")
    
    detected_items = []
    for line in raw_text.split('\n'):
        line = line.strip()
        if len(line) < 4 or line.isdigit(): 
            continue
            
        match = match_product_name(line)
        if match:
            detected_items.append(match)
            
    return {
        "status": "success",
        "detected_items_count": len(detected_items),
        "items": detected_items
    }