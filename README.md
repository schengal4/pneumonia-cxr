

# Pneumonia-CXR: FastAPI Application

This repository contains a **FastAPI** application for detecting and segmenting pneumonia on chest X-ray images using a custom model from [Hugging Face](https://huggingface.co/ianpan/pneumonia-cxr). The model performs two tasks:

1. **Classification** – Predicts whether an X-ray is likely to show pneumonia (vs. normal).
2. **Segmentation** – Highlights regions in the lung fields that may correspond to pneumonia.

## Table of Contents

- [Overview](#overview)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Locally](#running-locally)
  - [Endpoints](#endpoints)
  - [Example Requests](#example-requests)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [License and Disclaimer](#license-and-disclaimer)

---

## Overview

The application is written in **Python** using **FastAPI** for the web server and **PyTorch** / **timm** for deep learning. When you start the app, it downloads and loads the `ianpan/pneumonia-cxr` model from Hugging Face. The model was trained on large public radiology datasets (including the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) and [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/competitions/siim-covid19-detection) datasets).

**Key features**:

- **Segmentation Overlay**: Generates a red overlay on suspicious lung regions.
- **Probability & Label**: Returns a pneumonia probability (0.0–1.0) and a binary label (`PNEUMONIA` vs. `NORMAL`).
- **Multiple Output Formats**: Returns either an HTML page (with embedded image) or a JSON response (with base64-encoded overlay).

---

## Model Details

- **Model Name**: [`ianpan/pneumonia-cxr`](https://huggingface.co/ianpan/pneumonia-cxr)
- **Architecture**:  
  - **Encoder**: [EfficientNetV2-S](https://github.com/rwightman/pytorch-image-models) (via `timm`).  
  - **Decoder**: U-Net–style decoder for segmentation.  
  - **Classification Head**: A linear layer on top of pooled features.
- **Input**: Grayscale images, typically 512×512.
- **Output**:  
  - `cls`: Probability of pneumonia (`0.0–1.0`).  
  - `mask`: A segmentation map highlighting suspected pneumonia lesions.

> **Note**: This model is intended for educational or research purposes and is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## Installation

1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/username/pneumonia-cxr-app.git
   cd pneumonia-cxr-app
   ```
2. **Install Dependencies** (it’s recommended to use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
   The key libraries installed are:
   - `fastapi`
   - `uvicorn`
   - `torch`
   - `transformers`
   - `albumentations`
   - `pillow`

3. (Optional) **GPU Usage**: If you have a CUDA-capable GPU, install a GPU-compatible version of PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
   ```
   (Adjust the CUDA version as needed.)

---

## Usage

### Running Locally

Run the FastAPI application with **uvicorn**:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- **`--reload`**: Automatically restarts the server on code changes.
- **`--host 0.0.0.0`**: Exposes the server to your local network (optional).
- **`--port 8000`**: Default port for FastAPI apps.

Once running, open your browser to [http://127.0.0.1:8000](http://127.0.0.1:8000). You’ll see a welcome page describing the endpoints. For an interactive API docs page, go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Endpoints

1. **`GET /`**  
   - A simple welcome page describing available endpoints.

2. **`GET /example`**  
   - Loads a sample chest X-ray from `content/chest_xray.png` (if present), runs inference, and returns an HTML page with:
     - Pneumonia probability.
     - Binary label (`PNEUMONIA` vs. `NORMAL`).
     - Red segmentation overlay on the lung fields.

3. **`POST /predict`**  
   - Accepts a file upload (PNG/JPG).  
   - Returns an HTML page with the same items as above.

4. **`POST /predict_json`**  
   - Accepts a file upload (PNG/JPG).  
   - Returns a **JSON** object with:
     - `"label"`: `"PNEUMONIA"` or `"NORMAL"`.  
     - `"pneumonia_prob"`: Probability of pneumonia (0.0–1.0).  
     - `"image"`: Base64-encoded PNG image (the red overlay).

### Example Requests

#### Using `curl` to `/predict`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/your_chest_xray.png"
```

**Response**: An HTML page showing the predicted probability and an embedded overlay image.

#### Using `curl` to `/predict_json`:

```bash
curl -X POST http://127.0.0.1:8000/predict_json \
  -F "file=@/path/to/your_chest_xray.png"
```

**Response** (JSON):
```json
{
  "label": "PNEUMONIA",
  "pneumonia_prob": 0.8765,
  "image": "iVBORw0KGgoAAAANSUhEUgAAAA..."
}
```

---

## Project Structure

```
pneumonia-cxr-app/
│
├─ app.py               # Main FastAPI app
├─ configuration.py     # Custom PneumoniaConfig for the HF model
├─ modeling.py          # Defines PneumoniaModel class
├─ unet.py              # UNet decoder components
├─ config.json          # Model config (used by HF to load model)
├─ requirements.txt     # Python dependencies
├─ README.md            # You're reading it!
└─ content/
    └─ chest_xray.png   # Sample X-ray (used by /example)
```

---

## Technical Details

1. **Preprocessing**:  
   - Converts the uploaded image to grayscale (`convert("L")`).  
   - Resizes to `512×512` using [albumentations](https://albumentations.ai).  
   - Normalizes pixel values from `[0, 255]` → `[-1, 1]`.

2. **Inference**:
   - `model(input_tensor)` returns:
     - `outputs["cls"]`: Probability of pneumonia (`sigmoid` output).
     - `outputs["mask"]`: Segmentation mask (`sigmoid` output).

3. **Postprocessing**:
   - Thresholds the segmentation mask at 0.5.
   - Overlays it in red over the original grayscale image.
   - Encodes the result in base64 for easy HTML/JSON embedding.

4. **Dependencies**:
   - **PyTorch** + **timm** for the model.
   - **Transformers** for easy integration with Hugging Face custom code.
   - **FastAPI** + **uvicorn** for the web server.

---

## License and Disclaimer

- **License**: This repository does not include an explicit license. If you intend to distribute or modify this code, please add or consult an appropriate open-source license.
- **Disclaimer**: This tool is **not** a medical device. It is intended for research or demonstration purposes only and **cannot** replace professional medical diagnosis or advice. Use at your own risk and consult a qualified medical professional for clinical decisions.

