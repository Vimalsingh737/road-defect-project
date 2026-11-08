import os
import gc
import uuid
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI(title="Road Defect AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

MODEL_PATH = "best.onnx" if os.path.exists("best.onnx") else "best.pt"
model = YOLO(MODEL_PATH, task="segment")


@app.get("/")
def home():
    return {"status": "AI Server Online", "model": MODEL_PATH}


def compute_metrics(results, width, height):
    if results.masks is not None:
        masks_tensor = results.masks.data.cpu().numpy()
        combined_mask = np.any(masks_tensor > 0.5, axis=0).astype(np.uint8)
        m_h, m_w = combined_mask.shape
        defect_ratio = float(np.sum(combined_mask)) / (m_h * m_w)

        area_ratio = round(defect_ratio * 100, 2)
        defect_pixels = int(defect_ratio * width * height)
        severity_score = min(100.0, round(area_ratio * 4.5, 1))
        num_defects = len(results.masks)
    else:
        defect_pixels = 0
        area_ratio = 0.0
        severity_score = 0.0
        num_defects = 0

    priority = (
        "Critical" if severity_score > 70
        else "Medium" if severity_score > 35
        else "Low" if severity_score > 0
        else "Clear"
    )

    return {
        "defects_detected": num_defects,
        "defect_pixels": defect_pixels,
        "damaged_area_percent": area_ratio,
        "severity_score": severity_score,
        "priority": priority,
    }


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), conf: float = 0.15):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    H, W, _ = img.shape
    if W > 640:
        scale = 640 / W
        img = cv2.resize(img, (640, int(H * scale)))
        H, W, _ = img.shape

    results = model.predict(source=img, conf=conf, verbose=False)[0]
    metrics = compute_metrics(results, W, H)

    annotated_img = results.plot()
    _, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    base64_img = base64.b64encode(buffer).decode("utf-8")
    gc.collect()

    return {
        "media_type": "image",
        **metrics,
        "annotated_image_base64": f"data:image/jpeg;base64,{base64_img}",
    }


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...), conf: float = 0.15):
    file_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(OUTPUT_DIR, f"in_{file_id}.mp4")
    output_filename = f"out_{file_id}.webm"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 480
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24

    # Optimized for Render Free Tier (0.1 CPU / 512MB RAM)
    width = 480 if orig_w > 480 else orig_w
    height = int(orig_h * (width / orig_w)) if orig_w > 480 else orig_h

    fourcc = cv2.VideoWriter_fourcc(*"vp80")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    max_frames = 80
    frame_stride = 8

    peak_metrics = {
        "defects_detected": 0,
        "damaged_area_percent": 0.0,
        "severity_score": 0.0,
        "priority": "Clear",
    }
    last_metrics = peak_metrics
    worst_frame_b64 = None
    last_annotated = None

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (width, height))

        if frame_idx % frame_stride == 1:
            results = model.predict(source=frame, conf=conf, verbose=False)[0]
            last_annotated = results.plot()
            last_metrics = compute_metrics(results, width, height)

            if last_metrics["severity_score"] >= peak_metrics["severity_score"]:
                peak_metrics = last_metrics
                _, buf = cv2.imencode(".jpg", last_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                worst_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        annotated = last_annotated.copy() if last_annotated is not None else frame
        color = (0, 0, 255) if last_metrics["severity_score"] > 70 else (0, 165, 255) if last_metrics["severity_score"] > 35 else (0, 255, 0)
        cv2.rectangle(annotated, (8, 8), (340, 78), (20, 20, 20), -1)
        cv2.rectangle(annotated, (8, 8), (340, 78), color, 2)
        cv2.putText(annotated, f"Defects: {last_metrics['defects_detected']} | Area: {last_metrics['damaged_area_percent']}%",
                    (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(annotated, f"Severity: {last_metrics['severity_score']}/100 ({last_metrics['priority']})",
                    (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        out.write(annotated)

    cap.release()
    out.release()
    if os.path.exists(input_path):
        os.remove(input_path)
    gc.collect()

    return {
        "media_type": "video",
        **peak_metrics,
        "video_url": f"https://road-defect-project.onrender.com/outputs/{output_filename}",
        "annotated_image_base64": worst_frame_b64,
    }