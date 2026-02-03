from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import numpy as np
import re
from PIL import Image
import io

app = FastAPI()

# IMPORTANT: Allow your Next.js app to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "OCR API is Live"}

@app.post("/process-ocr")
async def process_ocr(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Preprocessing for better accuracy
    _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)
    
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', text, re.IGNORECASE)
    values = [float(m) for m in matches]
    
    return {
        "used": max(values) if values else 0,
        "remaining": min(values) if len(values) > 1 else 0
    }
