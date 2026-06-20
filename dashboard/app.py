import os
import sys
import base64
import numpy as np
import cv2
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras import models

# Ensure UTF-8 console output on Windows to prevent UnicodeEncodeError
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = FastAPI(title="Clinical UNet MRI Segmentation Workstation API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom metrics for loading model
def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def dice_coef_numpy(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return float((2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))

# Load pre-trained model
MODEL_PATH = 'brain_tumor_unet_final.keras'
print("🧠 Loading clinical U-Net segmentation model...")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Please train it first.")
model = models.load_model(MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
print("✅ Model loaded successfully!")

# Define data directories
IMAGE_DIR = 'Brain_Data_Clean/images'
MASK_DIR = 'Brain_Data_Clean/masks'

# Fallback to sample data if full clean data is missing
if not os.path.exists(IMAGE_DIR):
    print("📁 Fallback: using sample data registry...")
    IMAGE_DIR = 'sample_data/images'
    MASK_DIR = 'sample_data/masks'

# Build in-memory case list cache at startup
print("📁 Pre-loading and caching case registry...")
CASES_CACHE = []
if os.path.exists(IMAGE_DIR):
    _files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.npy')])
    for file in _files:
        case_id = file[:-4]  # Remove '.npy'
        parts = case_id.split('_')
        if len(parts) >= 3:
            patient_num = parts[1]
            slice_num = parts[2].replace('slice', '')
            name = f"Patient BRATS-{patient_num} (Slice {slice_num})"
        else:
            name = case_id
        CASES_CACHE.append({"id": case_id, "name": name})
    print(f"✅ Cached {len(CASES_CACHE)} cases.")
else:
    print("⚠️ WARNING: Neither clean data nor sample data directories were found.")


def array_to_base64_png(arr: np.ndarray, is_mask: bool = False, color: list = None) -> str:
    """Converts a 2D numpy array to a base64 encoded PNG string."""
    if is_mask:
        # Create an RGBA image
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        if color is None:
            color = [0, 243, 255, 160] # Default Neon Cyan
        # Set alpha and color where mask is active
        rgba[arr > 0] = color
        img_to_encode = rgba
    else:
        # Normalize MRI array to 0-255 grayscale
        mri_gray = (arr * 255).astype(np.uint8)
        img_to_encode = cv2.cvtColor(mri_gray, cv2.COLOR_GRAY2BGR)
        
    _, buffer = cv2.imencode('.png', img_to_encode)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

@app.get("/api/cases")
def get_cases():
    """Lists all available patient cases from the cached registry."""
    if not os.path.exists(IMAGE_DIR):
        raise HTTPException(status_code=404, detail="Clean data directory not found. Please run data_clean.py.")
    return CASES_CACHE

# In-memory prediction cache to eliminate CPU latency during case switching
PREDICTION_CACHE = {}

@app.get("/api/predict/case/{case_id}")
def predict_case(case_id: str):
    """Loads a specific case image and mask, runs inference, and returns base64 overlays."""
    if case_id in PREDICTION_CACHE:
        return PREDICTION_CACHE[case_id]
        
    img_path = os.path.join(IMAGE_DIR, f"{case_id}.npy")
    mask_path = os.path.join(MASK_DIR, f"{case_id}.npy")
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Case data not found.")
    
    # Load slice data
    img = np.load(img_path)
    mask = np.load(mask_path)
    
    # Preprocess for model (batch size 1, 128x128, 1 channel)
    input_tensor = np.expand_dims(img, axis=(0, -1))
    
    # Run prediction using direct call to avoid model.predict iterator overhead (3s -> 10ms)
    pred_tensor = model(input_tensor, training=False)
    pred = pred_tensor[0, :, :, 0].numpy()
    pred_binary = (pred > 0.5).astype(np.uint8)
    
    # Calculate slice Dice score using pure numpy for speed
    dice = dice_coef_numpy(mask, pred)
    
    # Calculate tumor areas (in pixels)
    gt_area = int(np.sum(mask > 0))
    pred_area = int(np.sum(pred_binary > 0))
    
    # Generate base64 images
    mri_b64 = array_to_base64_png(img)
    # Emerald green for ground truth [B, G, R, A] (OpenCV uses BGR)
    gt_b64 = array_to_base64_png(mask, is_mask=True, color=[46, 204, 113, 160])
    # Neon cyan for prediction [B, G, R, A]
    pred_b64 = array_to_base64_png(pred_binary, is_mask=True, color=[255, 243, 0, 160]) 
    
    result = {
        "case_id": case_id,
        "metrics": {
            "dice_score": round(float(dice), 4),
            "ground_truth_area_px": gt_area,
            "predicted_area_px": pred_area,
            "tumor_detected": bool(pred_area > 0)
        },
        "images": {
            "mri": mri_b64,
            "ground_truth": gt_b64,
            "prediction": pred_b64
        }
    }
    PREDICTION_CACHE[case_id] = result
    return result

@app.post("/api/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Handles raw image uploads, resizes to 128x128, runs inference, and returns results."""
    contents = await file.read()
    
    # Try reading as standard image file
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        # Check if it's a numpy file upload
        try:
            # Save temporarily to load
            temp_path = "temp_uploaded.npy"
            with open(temp_path, "wb") as f:
                f.write(contents)
            img = np.load(temp_path)
            os.remove(temp_path)
            if img.ndim != 2:
                raise HTTPException(status_code=400, detail="Numpy array must be a 2D slice.")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid file format. Upload a standard image or a 2D numpy slice.")
    else:
        # Convert BGR to grayscale and normalize
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img = img_gray.astype(np.float32) / 255.0
        
    # Resize to 128x128
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Run prediction using direct call to avoid model.predict iterator overhead
    input_tensor = np.expand_dims(img_resized, axis=(0, -1))
    pred_tensor = model(input_tensor, training=False)
    pred = pred_tensor[0, :, :, 0].numpy()
    pred_binary = (pred > 0.5).astype(np.uint8)
    pred_area = int(np.sum(pred_binary > 0))
    
    mri_b64 = array_to_base64_png(img_resized)
    pred_b64 = array_to_base64_png(pred_binary, is_mask=True, color=[255, 243, 0, 160])
    
    return {
        "metrics": {
            "predicted_area_px": pred_area,
            "tumor_detected": bool(pred_area > 0)
        },
        "images": {
            "mri": mri_b64,
            "prediction": pred_b64
        }
    }

# Mount static folder for frontend hosting
app.mount("/", StaticFiles(directory="dashboard/static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
