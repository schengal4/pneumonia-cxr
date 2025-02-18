import io
import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse
import uvicorn
from transformers import AutoModel
import base64

app = FastAPI()
model = None  # Global model reference

@app.on_event("startup")
def load_model():
    global model
    print("Loading ianpan/pneumonia-cxr model from Hugging Face...")
    model = AutoModel.from_pretrained("ianpan/pneumonia-cxr", trust_remote_code=True)
    model.eval()
    print("Model loaded successfully!")

def preprocess_image(file_bytes: bytes):
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
    arr = np.array(pil_img)
    arr = A.Resize(512, 512)(image=arr)["image"]
    resized_pil = Image.fromarray(arr)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor, resized_pil

def create_overlay(base_gray: Image.Image, seg_mask: np.ndarray) -> Image.Image:
    bin_mask = (seg_mask >= 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(bin_mask, mode="L")

    base_rgb = base_gray.convert("RGB")
    overlay_rgba = Image.new("RGBA", base_rgb.size, (255, 0, 0, 128))
    base_rgba = base_rgb.convert("RGBA")

    result_rgba = Image.composite(overlay_rgba, base_rgba, mask_image)
    return result_rgba.convert("RGB")

def pil_to_base64(img: Image.Image) -> str:
    """
    Converts a PIL image to a base64-encoded PNG string.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """
    Home page for the Pneumonia-CXR App.
    """
    content = f"""
    <html>
      <head>
        <title>Pneumonia-CXR App</title>
      </head>
      <body style="font-family: Arial; margin: 20px;">
        <h1>Welcome to the Pneumonia-CXR App</h1>
        <p>This application uses a computer vision model (<b>ianpan/pneumonia-cxr</b>) to detect pneumonia on chest X-rays.</p>
        
        <h2>Available Endpoints</h2>
        <ul>
          <li><strong>GET <code>/example</code></strong> – Demonstrates a sample chest X-ray (<code>content/chest_xray.png</code>) with overlay.</li>
          <li><strong>POST <code>/predict</code></strong> – Upload an image (PNG/JPG). Returns an HTML page with the probability & overlay.</li>
          <li><strong>POST <code>/predict_json</code></strong> – Upload an image (PNG/JPG). Returns a JSON response with the probability & base64-encoded overlay.</li>
        </ul>

        <h3>Try the Interactive Docs</h3>
        <p>You can also visit the 
           <a href="/docs" target="_blank">API Docs</a> or 
           <a href="/redoc" target="_blank">ReDoc</a> to explore these endpoints interactively.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=content, status_code=200)

@app.get("/example", response_class=HTMLResponse)
def example() -> HTMLResponse:
    """
    Example endpoint that uses a local chest X-ray image (e.g. content/chest_xray.png)
    to show how the model performs. Returns HTML with the pneumonia probability
    and the segmentation overlay image inline.
    """
    local_image_path = "content/chest_xray.png"
    if not os.path.exists(local_image_path):
        return HTMLResponse(content="<h2>Sample image not found on server.</h2>", status_code=404)

    # Read the file
    with open(local_image_path, "rb") as f:
        file_bytes = f.read()

    # Preprocess
    input_tensor, resized_pil = preprocess_image(file_bytes)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)

    pneumonia_prob = outputs["cls"][0, 0].item()
    seg_mask = outputs["mask"][0, 0].cpu().numpy()

    # Overlay
    result_img = create_overlay(resized_pil, seg_mask)

    # Convert overlay image to base64
    base64_img = pil_to_base64(result_img)

    # Build an HTML page that shows the probability + the overlay
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
        <img src="data:image/png;base64,{base64_img}" alt="Segmentation Overlay"/>
      </body>
    </html>
    """

    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> HTMLResponse:
    """
    Endpoint that:
    1) Reads an uploaded chest X-ray image (PNG/JPG/etc.)
    2) Runs classification & segmentation
    3) Overlays the segmentation mask in red
    4) Returns an HTML page (PNG embedded in base64) with the result
    """
    file_bytes = await file.read()
    input_tensor, resized_pil = preprocess_image(file_bytes)

    with torch.no_grad():
        outputs = model(input_tensor)

    pneumonia_prob = outputs["cls"][0, 0].item()
    seg_mask = outputs["mask"][0, 0].cpu().numpy()

    # Overlay
    result_img = create_overlay(resized_pil, seg_mask)

    # Convert overlay image to base64
    base64_img = pil_to_base64(result_img)

    label = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"
    html_content = f"""
    <html>
      <head>
        <title>Predict - Pneumonia Segmentation</title>
      </head>
      <body style="font-family: Arial; margin: 20px;">
        <h1>Uploaded Chest X-ray</h1>
        <p><b>Pneumonia Probability:</b> {pneumonia_prob:.4f}</p>
        <p><b>Label:</b> {label}</p>
        <img src="data:image/png;base64,{base64_img}" alt="Segmentation Overlay"/>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...)) -> dict:
    """
    Endpoint that:
    1) Reads an uploaded chest X-ray image (PNG/JPG/etc.)
    2) Runs classification & segmentation
    3) Overlays the segmentation mask in red
    4) Returns a JSON object with the probability & base64-encoded overlay
    """
    file_bytes = await file.read()
    input_tensor, resized_pil = preprocess_image(file_bytes)

    with torch.no_grad():
        outputs = model(input_tensor)

    pneumonia_prob = outputs["cls"][0, 0].item()
    seg_mask = outputs["mask"][0, 0].cpu().numpy()

    # Overlay
    result_img = create_overlay(resized_pil, seg_mask)

    # Convert overlay image to base64
    base64_img = pil_to_base64(result_img)

    label = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"
    return {
        "label": label,
        "pneumonia_prob": pneumonia_prob,
        "image": base64_img  # base64-encoded PNG
    }

