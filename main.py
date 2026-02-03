from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import numpy as np
import re
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Vines OCR API is Live"}

@app.post("/process-ocr")
async def process_ocr(file: UploadFile = File(...)):
    # Read image
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    img_np = np.array(img)

    # Preprocessing
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # OCR
    text = pytesseract.image_to_string(thresh, config="--psm 6")

    # Parse GB values
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

    return {
        "used": round(used, 2),
        "remaining": round(remaining, 2)
    }
