import io
import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from transformers import AutoModel
import base64
import uuid
from datetime import datetime
from typing import Dict, Any
import logging

import traceback



# Add this to your FastAPI app initialization for Swagger UI
app = FastAPI(
    title="Pneumonia-CXR Detection API",
    description="""
## Pneumonia Detection and Segmentation API

Upload chest X-ray images to detect pneumonia and visualize affected lung regions.

### Quick Start
1. **Upload Image**: Use POST endpoints with PNG/JPG chest X-ray files (max 10MB)
2. **Get Results**: Receive pneumonia probability and highlighted segmentation overlay
3. **Choose Format**: HTML page (`/predict`) or JSON response (`/predict_json`)

### Important Disclaimers
- **Not a medical device** - For research/educational purposes only
- **Not for clinical diagnosis** - Always consult healthcare professionals
- **Dataset limitations** - Trained on specific datasets, may not generalize

### Usage Tips
- Use frontal chest X-ray images for best results
- Ensure images are clear and properly oriented
- Check `/health` endpoint for service status
- View `/example` for sample prediction

**Model**: [ianpan/pneumonia-cxr](https://huggingface.co/ianpan/pneumonia-cxr) from Hugging Face
    """,
    version="1.0.0"
)
model = None  # Global model reference

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Updated model loading with error handling
@app.on_event("startup")
async def load_model():
    global model
    try:
        logger.info("Loading ianpan/pneumonia-cxr model from Hugging Face...")
        model = AutoModel.from_pretrained("ianpan/pneumonia-cxr", trust_remote_code=True)
        model.eval()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None  # Ensure model is None if loading fails
        # Don't raise here - let the app start but health check will show unhealthy
def validate_upload_file(file: UploadFile, file_bytes: bytes) -> None:
    """Comprehensive file validation for medical imaging"""
    
    # Check filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    allowed_extensions = ['.png', '.jpg', '.jpeg']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="File must be PNG or JPG format")
    
    # Check content type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid content type - must be image")
    
    # Check file size
    file_size = len(file_bytes)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
  # File header validation (works without magic)
    if len(file_bytes) < 8:
        raise HTTPException(status_code=400, detail="File too small to be valid image")
    
    header = file_bytes[:8]
    is_png = header.startswith(b'\x89PNG\r\n\x1a\n')
    is_jpeg = header.startswith(b'\xff\xd8\xff')
    
    if not (is_png or is_jpeg):
        raise HTTPException(status_code=400, detail="Invalid image file format")
def log_request_start(request_id: str, endpoint: str, filename: str, file_size: int) -> None:
    """Structured logging for request start"""
    try:
      logger.info(
          "REQUEST_START",
          extra={
              "request_id": request_id,
              "endpoint": endpoint,
              "upload_filename": filename,
              "file_size_bytes": file_size,
              "timestamp": datetime.utcnow().isoformat()
          }
      )
    except Exception as e:
        logger.error("Failed to log request start.")

def log_prediction_result(request_id: str, label: str, probability: float, processing_time: float) -> None:
    """Structured logging for prediction results"""
    logger.info(
        "PREDICTION_COMPLETE",
        extra={
            "request_id": request_id,
            "predicted_label": label,
            "pneumonia_probability": round(probability, 4),
            "processing_time_seconds": round(processing_time, 3),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

def log_error(request_id: str, error_type: str, error_message: str, endpoint: str) -> None:
    """Structured logging for errors"""
    logger.error(
        "REQUEST_ERROR",
        extra={
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message,
            "endpoint": endpoint,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

def preprocess_image(file_bytes: bytes):
    """
    Reads image bytes, converts to grayscale, resizes to 512x512,
    and returns (tensor, resized PIL).
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
    arr = np.array(pil_img)
    arr = A.Resize(512, 512)(image=arr)["image"]
    resized_pil = Image.fromarray(arr)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor, resized_pil


def create_overlay(base_gray: Image.Image, seg_mask: np.ndarray) -> Image.Image:
    """
    Overlays the (binary) segmentation mask in red at a fixed size (512x512).
    This is the original approach used by /predict and /predict_json.
    """
    bin_mask = (seg_mask >= 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(bin_mask, mode="L")

    base_rgb = base_gray.convert("RGB")
    # The alpha=128 is ignored with Image.composite if mask is strictly 0/255,
    # so it appears fully opaque in practice. We leave it as-is for demonstration.
    overlay_rgba = Image.new("RGBA", base_rgb.size, (255, 0, 0, 85))
    base_rgba = base_rgb.convert("RGBA")

    result_rgba = Image.composite(overlay_rgba, base_rgba, mask_image)
    return result_rgba.convert("RGB")


def create_overlay_original(base_gray: Image.Image, seg_mask: np.ndarray) -> Image.Image:
    """
    Overlays the segmentation mask onto the original grayscale image dimensions.
    1) We assume seg_mask is 0–1 (float) or 0–255 (uint8) at 512x512.
    2) We upsample it to base_gray.size, then alpha-blend.
    """
    orig_w, orig_h = base_gray.size

    # Convert seg_mask to [0..255]
    if seg_mask.dtype != np.uint8:
        seg_mask_255 = (seg_mask * 255).astype(np.uint8)
    else:
        seg_mask_255 = seg_mask

    # Create a PIL image for the 512x512 mask
    mask_pil_512 = Image.fromarray(seg_mask_255, mode="L")

    # Upsample mask to original size
    mask_pil_orig = mask_pil_512.resize((orig_w, orig_h), resample=Image.NEAREST)

    # Convert the base to RGBA
    base_rgba = base_gray.convert("RGBA")

    # Create a red overlay (fully transparent initially)
    overlay_rgba = Image.new("RGBA", base_rgba.size, (255, 0, 0, 0))

    # We'll do a simple binary threshold: alpha=128 where mask>128
    mask_array = np.array(mask_pil_orig)
    alpha_data = np.where(mask_array > 128, 128, 0).astype(np.uint8)
    alpha_image = Image.fromarray(alpha_data, mode="L")

    # Put that alpha channel into the red overlay
    overlay_rgba.putalpha(alpha_image)

    # Alpha-composite the overlay onto the base
    result_rgba = Image.alpha_composite(base_rgba, overlay_rgba)

    # Return as RGB (flatten alpha)
    return result_rgba.convert("RGB")


def pil_to_base64(img: Image.Image) -> str:
    """
    Converts a PIL image to a base64-encoded PNG string.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.get("/health")
def health_check():
    """
    Health check endpoint for Health Universe monitoring and Navigator integration.
    Returns service status and model availability.
    """
    try:
        # Check if model is loaded
        if model is None:
            return {
                "status": "unhealthy",
                "detail": "Model not loaded",
                "service": "pneumonia-cxr",
                "model_loaded": False
            }
        
        # Model is available
        return {
            "status": "healthy",
            "service": "pneumonia-cxr",
            "model_loaded": True,
            "model_name": "ianpan/pneumonia-cxr"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "detail": f"Health check failed: {str(e)}",
            "service": "pneumonia-cxr",
            "model_loaded": False
        }

@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """
    Concise home page with clear usage instructions.
    """
    content = """
    <html>
      <head>
        <title>Pneumonia-CXR Detection API</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; max-width: 800px; }
          .header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
          .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
          .quick-start { background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; }
          .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }
          pre { background: #f4f4f4; padding: 10px; border-radius: 4px; }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>Pneumonia-CXR Detection API</h1>
          <p>AI-powered pneumonia detection and lung segmentation for chest X-rays</p>
        </div>

        <div class="warning">
          <strong>⚠️ Important:</strong> This is NOT a medical device. For research/educational use only. 
          Always consult qualified healthcare professionals for medical diagnosis.
        </div>

        <div class="quick-start">
          <h2>Quick Start Guide</h2>
          <ol>
            <li><strong>Upload</strong> a chest X-ray image (PNG/JPG, max 10MB)</li>
            <li><strong>Choose</strong> HTML display or JSON response</li>
            <li><strong>Review</strong> pneumonia probability and highlighted regions</li>
          </ol>
        </div>

        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
          <strong>GET /example</strong><br>
          View sample prediction with demo chest X-ray
        </div>

        <div class="endpoint">
          <strong>POST /predict</strong><br>
          Upload image → Get HTML page with results and overlay
        </div>

        <div class="endpoint">
          <strong>POST /predict_json</strong><br>
          Upload image → Get JSON response with base64-encoded overlay
        </div>

        <div class="endpoint">
          <strong>GET /health</strong><br>
          Check service status and model availability
        </div>
        <h2>What You Get</h2>
        <ul>
          <li><strong>Classification</strong>: PNEUMONIA or NORMAL label with probability (0.0-1.0)</li>
          <li><strong>Segmentation</strong>: Red overlay highlighting suspicious lung regions</li>
          <li><strong>Metadata</strong>: Processing time and request ID for tracking</li>
        </ul>

        <h2>File Requirements</h2>
        <ul>
          <li>Format: PNG or JPG</li>
          <li>Size: Maximum 10MB</li>
          <li>Content: Frontal chest X-ray images work best</li>
          <li>Quality: Clear, properly oriented images recommended</li>
        </ul>

        <h2>Model Information</h2>
        <p>
          <strong>Model</strong>: <a href="https://huggingface.co/ianpan/pneumonia-cxr" target="_blank">ianpan/pneumonia-cxr</a><br>
          <strong>Architecture</strong>: EfficientNetV2-S encoder with U-Net decoder<br>
          <strong>Training Data</strong>: RSNA Pneumonia Detection Challenge, SIIM-FISABIO-RSNA COVID-19 Detection
        </p>

        <div class="warning">
          <strong>Limitations:</strong> Model predictions are based on specific training datasets 
          and may not generalize to all patient populations or imaging conditions. 
          This tool cannot replace professional medical evaluation.
        </div>

        <hr>
        <p style="color: #666; font-size: 0.9em;">
          Pneumonia-CXR Detection API | Research and Educational Use Only
        </p>
      </body>
    </html>
    """
    return HTMLResponse(content=content, status_code=200)


@app.get("/example", response_class=HTMLResponse)
def example() -> HTMLResponse:
    """
    Example endpoint that uses a local chest X-ray image (e.g. content/chest_xray.png)
    to show how the model performs. Returns HTML with the pneumonia probability
    and the segmentation overlay image at the original resolution.
    """
    local_image_path = "content/chest_xray.png"
    if not os.path.exists(local_image_path):
        return HTMLResponse(content="<h2>Sample image not found on server.</h2>", status_code=404)

    # 1) Read the file bytes
    with open(local_image_path, "rb") as f:
        file_bytes = f.read()

    # 2) Store the original grayscale image (full size)
    orig_pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
    orig_width, orig_height = orig_pil_img.size

    # 3) Preprocess for the model (resized to 512x512)
    input_tensor, _ = preprocess_image(file_bytes)

    # 4) Inference
    with torch.no_grad():
        outputs = model(input_tensor)

    pneumonia_prob = outputs["cls"][0, 0].item()
    seg_mask = outputs["mask"][0, 0].cpu().numpy()

    # 5) Create overlay on the ORIGINAL image
    result_img = create_overlay_original(orig_pil_img, seg_mask)

    # 6) Convert overlay image to base64
    base64_img = pil_to_base64(result_img)

    # 7) Build an HTML page that shows the probability + the overlay
    label = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"
    html_content = f"""
    <html>
      <head>
        <title>Example - Pneumonia Segmentation</title>
      </head>
      <body style="font-family: Arial; margin: 20px;">
        <h1>Example Chest X-ray</h1>
        <p><b>Pneumonia Probability:</b> {pneumonia_prob:.4f}</p>
        <p><b>Label:</b> {label}</p>
        <p>
          Below is the original image with the suspicious area(s)
          highlighted in red (semi-transparent). 
        </p>
        <img src="data:image/png;base64,{base64_img}" alt="Segmentation Overlay" />
      </body>
    </html>
    """

    return HTMLResponse(content=html_content, status_code=200)


# Updated predict endpoint with same enhancements
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> HTMLResponse:
    """HTML prediction endpoint with structured logging and validation"""
    
    request_id = str(uuid.uuid4())[:8]
    start_time = datetime.utcnow()
    
    try:
        print(f"DEBUG: Starting predict for file: {file.filename}")
        if model is None:
            print("DEBUG: Model is None")
            log_error(request_id, "MODEL_UNAVAILABLE", "Model not loaded", "predict")
            raise HTTPException(status_code=503, detail="Model not available")
        
        print("DEBUG: Reading file bytes")
        file_bytes = await file.read()
        print(f"DEBUG: File size: {len(file_bytes)}")
        
        log_request_start(request_id, "predict", file.filename or "unknown", len(file_bytes))
        print(f"DEBUG: Logged request start")
        
        # Validation and processing (same as predict_json)
        try:
            print("DEBUG: Starting validation")
            validate_upload_file(file, file_bytes)  # This might be the issue
            print("DEBUG: Validation passed")

            print("DEBUG: Preprocessing image")
            input_tensor, resized_pil = preprocess_image(file_bytes)
            print("DEBUG: Preprocessing complete")
        except HTTPException as e:
            log_error(request_id, "VALIDATION_ERROR", e.detail, "predict")
            raise
        except Exception as e:
            log_error(request_id, "PREPROCESSING_ERROR", str(e), "predict")
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # Inference and response generation
        try:
            with torch.no_grad():
                outputs = model(input_tensor)
            
            pneumonia_prob = float(outputs["cls"][0, 0].item())
            seg_mask = outputs["mask"][0, 0].cpu().numpy()
            
            if not (0.0 <= pneumonia_prob <= 1.0):
                log_error(request_id, "MODEL_OUTPUT_ERROR", f"Invalid probability: {pneumonia_prob}", "predict")
                raise HTTPException(status_code=500, detail="Invalid model output")
            
            result_img = create_overlay(resized_pil, seg_mask)
            base64_img = pil_to_base64(result_img)
            label = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            log_prediction_result(request_id, label, pneumonia_prob, processing_time)
            
            html_content = f"""
            <html>
              <head><title>Predict - Pneumonia Segmentation</title></head>
              <body style="font-family: Arial; margin: 20px;">
                <h1>Uploaded Chest X-ray</h1>
                <p><b>Request ID:</b> {request_id}</p>
                <p><b>Pneumonia Probability:</b> {pneumonia_prob:.4f}</p>
                <p><b>Label:</b> {label}</p>
                <p><b>Processing Time:</b> {processing_time:.2f}s</p>
                <img src="data:image/png;base64,{base64_img}" alt="Segmentation Overlay"/>
              </body>
            </html>
            """
            return HTMLResponse(content=html_content, status_code=200)
            
        except HTTPException:
            raise
        except Exception as e:
            log_error(request_id, "PROCESSING_ERROR", str(e), "predict")
            raise HTTPException(status_code=500, detail="Prediction failed")
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(request_id, "UNEXPECTED_ERROR", str(e), "predict")
        raise HTTPException(status_code=500, detail="Internal server error")

# Updated predict_json endpoint with structured logging and validation
@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...)) -> dict:
    """JSON prediction endpoint with structured logging and comprehensive validation"""
    
    # Generate unique request ID for tracing
    request_id = str(uuid.uuid4())[:8]
    start_time = datetime.utcnow()
    
    try:
        # Check model availability
        if model is None:
            log_error(request_id, "MODEL_UNAVAILABLE", "Model not loaded", "predict_json")
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Read file first to get size
        file_bytes = await file.read()
        
        # Log request start
        log_request_start(request_id, "predict_json", file.filename or "unknown", len(file_bytes))
        
        # Comprehensive file validation
        try:
            validate_upload_file(file, file_bytes)
        except HTTPException as e:
            log_error(request_id, "VALIDATION_ERROR", e.detail, "predict_json")
            raise
        
        # Process image
        try:
            input_tensor, resized_pil = preprocess_image(file_bytes)
        except Exception as e:
            log_error(request_id, "PREPROCESSING_ERROR", str(e), "predict_json")
            raise HTTPException(status_code=400, detail="Failed to process image - ensure it's a valid chest X-ray")
        
        # Run inference
        try:
            with torch.no_grad():
                outputs = model(input_tensor)
            
            pneumonia_prob = float(outputs["cls"][0, 0].item())
            seg_mask = outputs["mask"][0, 0].cpu().numpy()
            
            # Validate model output
            if not (0.0 <= pneumonia_prob <= 1.0):
                log_error(request_id, "MODEL_OUTPUT_ERROR", f"Invalid probability: {pneumonia_prob}", "predict_json")
                raise HTTPException(status_code=500, detail="Invalid model output")
                
        except HTTPException:
            raise
        except Exception as e:
            log_error(request_id, "INFERENCE_ERROR", str(e), "predict_json")
            raise HTTPException(status_code=500, detail="Model prediction failed")
        
        # Generate response
        try:
            result_img = create_overlay(resized_pil, seg_mask)
            base64_img = pil_to_base64(result_img)
            label = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Log successful prediction
            log_prediction_result(request_id, label, pneumonia_prob, processing_time)
            
            return {
                "request_id": request_id,
                "label": label,
                "pneumonia_prob": round(pneumonia_prob, 4),
                "image": base64_img,
                "processing_time_seconds": round(processing_time, 3)
            }
            
        except Exception as e:
            log_error(request_id, "RESPONSE_GENERATION_ERROR", str(e), "predict_json")
            raise HTTPException(status_code=500, detail="Failed to generate response")
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(request_id, "UNEXPECTED_ERROR", str(e), "predict_json")
        raise HTTPException(status_code=500, detail="Internal server error")


# If you want to run via "python app.py" directly (instead of uvicorn app:app)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
