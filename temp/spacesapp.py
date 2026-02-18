import os
import json
import torch
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
from supabase import create_client, Client

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = None
model = None
MASTER_PRODUCTS = []
MASTER_SKUS = []
MASTER_EMBEDDINGS = None

@app.on_event("startup")
async def startup_event():
    global supabase, model
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception:
            pass
    model = SentenceTransformer('all-MiniLM-L6-v2')
    await refresh_data_internal()

async def refresh_data_internal():
    global MASTER_PRODUCTS, MASTER_EMBEDDINGS, MASTER_SKUS
    if not supabase:
        return
    try:
        response = supabase.table("master_products").select("id, name, sku, embedding").execute()
        data = response.data
    except Exception:
        return
    if not data:
        MASTER_PRODUCTS = []
        MASTER_SKUS = []
        MASTER_EMBEDDINGS = None
        return
    
    names_to_encode = []
    ids_to_update = []
    loaded_names = []
    loaded_skus = []
    loaded_embeddings = []

    for item in data:
        if item['name']:
            loaded_names.append(item['name'])
            loaded_skus.append(item.get('sku', 'N/A'))
            emb = item.get('embedding')
            if emb:
                try:
                    if isinstance(emb, str):
                        clean_emb = emb.replace('{', '[').replace('}', ']')
                        emb = json.loads(clean_emb)
                    loaded_embeddings.append(emb)
                except Exception:
                    names_to_encode.append(item['name'])
                    ids_to_update.append(item['id'])
            else:
                names_to_encode.append(item['name'])
                ids_to_update.append(item['id'])

    if names_to_encode:
        new_embeddings = model.encode(names_to_encode).tolist()
        for i, uid in enumerate(ids_to_update):
            try:
                supabase.table("master_products").update(
                    {"embedding": new_embeddings[i]}
                ).eq("id", uid).execute()
                loaded_embeddings.append(new_embeddings[i])
            except Exception:
                pass

    MASTER_PRODUCTS = loaded_names
    MASTER_SKUS = loaded_skus
    if loaded_embeddings:
        try:
            MASTER_EMBEDDINGS = torch.tensor(loaded_embeddings, dtype=torch.float)
        except Exception:
            pass

@app.get("/")
def home():
    return {
        "status": "active",
        "count": len(MASTER_PRODUCTS),
        "db": supabase is not None
    }

@app.get("/match")
def get_match(text: str):
    if not MASTER_PRODUCTS or MASTER_EMBEDDINGS is None:
        return {"matched_product": "No Match", "sku": "N/A", "confidence": 0, "status": "no_data"}
    
    input_embedding = model.encode(text, convert_to_tensor=True)
    cos_scores = util.cos_sim(input_embedding, MASTER_EMBEDDINGS)
    best_idx = torch.argmax(cos_scores).item()
    sbert_score = cos_scores[0][best_idx].item()
    sbert_name = MASTER_PRODUCTS[best_idx]
    
    fuzzy_res = process.extractOne(text, MASTER_PRODUCTS, scorer=fuzz.token_set_ratio)
    fuzzy_name = fuzzy_res[0]
    fuzzy_score_norm = fuzzy_res[1] / 100.0
    
    if sbert_score > 0.85:
        final_score = (sbert_score * 0.8) + (fuzzy_score_norm * 0.2)
        best_name = sbert_name
    else:
        final_score = (fuzzy_score_norm * 0.6) + (sbert_score * 0.4)
        best_name = fuzzy_name

    try:
        final_idx = MASTER_PRODUCTS.index(best_name)
        matched_sku = MASTER_SKUS[final_idx]
    except Exception:
        matched_sku = "N/A"

    if final_score >= 0.30:
        return {
            "matched_product": best_name,
            "sku": matched_sku,
            "confidence": round(final_score * 100, 2),
            "status": "success"
        }
    else:
        return {
            "matched_product": "No Match",
            "sku": matched_sku if final_score > 0.1 else "N/A",
            "confidence": round(final_score * 100, 2),
            "status": "low_confidence"
        }

@app.post("/refresh-data")
async def trigger_refresh():
    await refresh_data_internal()
    return {"status": "success", "total_products": len(MASTER_PRODUCTS)}