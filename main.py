from fastapi import FastAPI, UploadFile, File
import pytesseract
import cv2
import numpy as np
import re
from PIL import Image
import io

app = FastAPI()

@app.post("/process-ocr")
async def process_ocr(file: UploadFile = File(...)):
    # Read image
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Preprocessing
    _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)
    
    # OCR Logic
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*GB', text, re.IGNORECASE)
    values = [float(m) for m in matches]
    
    return {
        "used": max(values) if values else 0,
        "remaining": min(values) if len(values) > 1 else 0,
        "raw_text": text
    }