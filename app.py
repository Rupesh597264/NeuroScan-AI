
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import cv2
import joblib
import csv

# ---------------------------------------
# 1. Project paths
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "model.joblib"
STATIC_DIR = BASE_DIR / "static"
LOG_FILE = BASE_DIR / "prediction_logs.csv"

# ---------------------------------------
# 2. Initialize FastAPI
# ---------------------------------------

app = FastAPI(
    title="NeuroScan AI",
    description="Educational Alzheimer's MRI Classification System",
    version="1.0.0"
)

# ---------------------------------------
# 3. Load model once at startup
# ---------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

# ---------------------------------------
# 4. Mount static folder
# ---------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

# ---------------------------------------
# 5. Homepage
# ---------------------------------------

@app.get("/")
def home():

    return FileResponse(
        STATIC_DIR / "index.html"
    )

# ---------------------------------------
# 6. Health check
# ---------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "project": "NeuroScan AI",
        "model_loaded": True
    }

# ---------------------------------------
# 7. Image preprocessing
# ---------------------------------------

def preprocess_image(image_bytes):

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Invalid or unreadable image")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    normalized = resized.astype(
        np.float32
    ) / 255.0

    flattened = normalized.flatten()

    return flattened.reshape(1, -1)

# ---------------------------------------
# 8. Prediction logging
# ---------------------------------------

def log_prediction(prediction, features):

    file_exists = LOG_FILE.exists()

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "prediction",
                "input_mean",
                "input_std"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            prediction,
            float(np.mean(features)),
            float(np.std(features))
        ])

# ---------------------------------------
# 9. Prediction endpoint
# ---------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp"
    }

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=422,
            detail="Only JPG, JPEG, PNG and BMP images are accepted."
        )

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=422,
            detail="Image size must not exceed 10 MB."
        )

    if not image_bytes:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty."
        )

    try:
        features = preprocess_image(image_bytes)

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    prediction = int(
        model.predict(features)[0]
    )

    if prediction == 0:
        result = "Non-Demented"

        tips = [
            "This model classified the uploaded image as Non-Demented.",
            "This experimental result should not replace professional assessment.",
            "Consult a qualified healthcare professional for medical concerns."
        ]

    else:
        result = "Demented"

        tips = [
            "This model classified the uploaded image as Demented.",
            "This experimental result is not a clinical diagnosis.",
            "Consult a qualified healthcare professional for further evaluation."
        ]

    log_prediction(result, features)

    return {
        "prediction": result,
        "class": prediction,
        "model": "NeuroScan AI",
        "tips": tips,
        "message": "Educational prediction only. Not a medical diagnosis."
    }
