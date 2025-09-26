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
import magic  # pip install python-magic
import logging



app = FastAPI()
model = None  # Global model reference

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
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
    
    # Validate actual file content using python-magic
    try:
        mime_type = magic.from_buffer(file_bytes, mime=True)
        if mime_type not in ['image/png', 'image/jpeg']:
            raise HTTPException(status_code=400, detail="File content doesn't match extension")
    except Exception:
        # Fallback if python-magic not available
        logger.warning("python-magic not available for file validation")
def log_request_start(request_id: str, endpoint: str, filename: str, file_size: int) -> None:
    """Structured logging for request start"""
    logger.info(
        "REQUEST_START",
        extra={
            "request_id": request_id,
            "endpoint": endpoint,
            "filename": filename,
            "file_size_bytes": file_size,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

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
    Expanded Home page for the Pneumonia-CXR App.
    """
    content = f"""
    <html>
      <head>
        <title>Pneumonia-CXR App</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
          }}
          h1, h2, h3 {{
            margin-top: 1em;
          }}
          ul {{
            list-style-type: disc;
            margin-left: 1.5em;
          }}
          .note {{
            background: #fafaf0;
            border-left: 4px solid #ffd700;
            padding: 10px;
            margin-top: 1em;
          }}
        </style>
      </head>
      <body>
        <h1>Welcome to the Pneumonia-CXR App</h1>

        <p>
          This application uses a custom <strong>Hugging Face Transformers</strong> model 
          (<a href="https://huggingface.co/ianpan/pneumonia-cxr" target="_blank">ianpan/pneumonia-cxr</a>)
          to detect and segment potential pneumonia regions on frontal chest X-rays. 
          The model was trained on publicly available radiology datasets, including:
        </p>
        <ul>
          <li>
            <a href="https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge" target="_blank">
              RSNA Pneumonia Detection Challenge
            </a>
          </li>
          <li>
            <a href="https://www.kaggle.com/competitions/siim-covid19-detection" target="_blank">
              SIIM-FISABIO-RSNA COVID-19 Detection
            </a>
          </li>
        </ul>

        <p>
          The model performs two main tasks:
          <ul>
            <li><strong>Classification</strong>: Estimates the probability that an X-ray shows pneumonia.</li>
            <li><strong>Segmentation</strong>: Generates a mask highlighting suspicious lung areas.</li>
          </ul>
        </p>

        <div class="note">
          <strong>Important Disclaimer:</strong><br/>
          This tool is <strong>not</strong> a medical device and does <strong>not</strong> provide a definitive 
          medical diagnosis. It is intended for research or educational demonstration only. 
          Always consult a qualified healthcare professional for medical advice or diagnosis.
        </div>

        <h2>How to Use</h2>
        <p>
          You can interact with the following endpoints to see how the model performs on sample images 
          or your own uploads:
        </p>
        <ul>
          <li>
            <strong>GET <code>/example</code></strong> – Demonstrates a sample chest X-ray (source: https://pmc.ncbi.nlm.nih.gov/articles/PMC4845314/) 
            (from <code>content/chest_xray.png</code>) with an overlaid segmentation mask 
            at the <em>original</em> image size.
          </li>
          <li>
            <strong>POST <code>/predict</code></strong> – Upload an image (PNG/JPG). 
            Returns an HTML page containing the probability, label, and red overlay 
            (currently shown at 512x512).
          </li>
          <li>
            <strong>POST <code>/predict_json</code></strong> – Upload an image (PNG/JPG). 
            Returns a JSON response with:
            <ul>
              <li><code>label</code> (PNEUMONIA/NORMAL)</li>
              <li><code>pneumonia_prob</code> (float)</li>
              <li><code>image</code> (base64-encoded PNG of the overlay)</li>
            </ul>
          </li>
        </ul>

        <p>
          For quick tests, you can use <a href="/docs" target="_blank">Swagger UI</a> or 
          <a href="/redoc" target="_blank">ReDoc</a> to upload files interactively.
        </p>

        <h3>Installation & Running Locally</h3>
        <p>
          If you cloned this repository, install dependencies from 
          <code>requirements.txt</code>:
        </p>
        <pre>
pip install -r requirements.txt
        </pre>
        <p>Then run the app with:</p>
        <pre>
uvicorn app:app --reload --host 0.0.0.0 --port 8000
        </pre>
        <p>
          Open your browser at 
          <a href="http://127.0.0.1:8000" target="_blank">http://127.0.0.1:8000</a> 
          to see this homepage. 
        </p>

        <h3>Technical Details</h3>
        <p>
          Under the hood, this app:
          <ul>
            <li>Uses <strong>PyTorch</strong> + <strong>timm</strong> for the model.</li>
            <li>Leverages a <strong>U-Net</strong> style decoder for segmentation.</li>
            <li>Applies thresholding to produce a binary mask for the overlay.</li>
            <li>Embeds the resulting overlay image in HTML or returns it as base64 in JSON.</li>
          </ul>
        </p>

        <h3>Limitations & Notes</h3>
        <p>
          - The model’s predictions are based on patterns learned from specific datasets; it may not 
            generalize to all patient populations or image sources.<br/>
          - For actual medical concerns, please consult a licensed healthcare professional.
        </p>

        <hr/>
        <p style="font-size: 0.9em;">
          <em>
            &copy; 2023 Pneumonia-CXR. 
            This application is provided for demonstration purposes only.
          </em>
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
        if model is None:
            log_error(request_id, "MODEL_UNAVAILABLE", "Model not loaded", "predict")
            raise HTTPException(status_code=503, detail="Model not available")
        
        file_bytes = await file.read()
        log_request_start(request_id, "predict", file.filename or "unknown", len(file_bytes))
        
        # Validation and processing (same as predict_json)
        try:
            validate_upload_file(file, file_bytes)
            input_tensor, resized_pil = preprocess_image(file_bytes)
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
