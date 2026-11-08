import os
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


def compute_metrics(results, width, height):
    total_pixels = width * height
    if results.masks is not None:
        masks_tensor = results.masks.data.cpu().numpy()
        combined_mask = np.any(masks_tensor > 0.5, axis=0).astype(np.uint8)
        combined_mask = cv2.resize(combined_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        defect_pixels = int(np.sum(combined_mask))
        area_ratio = round((defect_pixels / total_pixels) * 100, 2)
        severity_score = min(100.0, round(area_ratio * 4.5, 1))
        num_defects = len(results.masks)
    else:
        defect_pixels = 0
        area_ratio = 0.0
        severity_score = 0.0
        num_defects = 0

    if severity_score > 70:
        priority = "Critical"
    elif severity_score > 35:
        priority = "Medium"
    elif severity_score > 0:
        priority = "Low"
    else:
        priority = "Clear"

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
    results = model.predict(source=img, conf=conf, verbose=False)[0]
    metrics = compute_metrics(results, W, H)

    annotated_img = results.plot()
    _, buffer = cv2.imencode(".jpg", annotated_img)
    base64_img = base64.b64encode(buffer).decode("utf-8")

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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    fourcc = cv2.VideoWriter_fourcc(*"vp80")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    peak_metrics = {
        "defects_detected": 0,
        "damaged_area_percent": 0.0,
        "severity_score": 0.0,
        "priority": "Clear",
    }
    worst_frame_b64 = None

    while cap.isOpened() and frame_idx < 300:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(source=frame, conf=conf, verbose=False)[0]
        annotated = results.plot()
        metrics = compute_metrics(results, width, height)

        if metrics["severity_score"] >= peak_metrics["severity_score"]:
            peak_metrics = metrics
            _, buf = cv2.imencode(".jpg", annotated)
            worst_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        color = (0, 0, 255) if metrics["severity_score"] > 70 else (0, 165, 255) if metrics["severity_score"] > 35 else (0, 255, 0)
        cv2.rectangle(annotated, (15, 15), (460, 120), (20, 20, 20), -1)
        cv2.rectangle(annotated, (15, 15), (460, 120), color, 2)
        cv2.putText(annotated, f"Defects: {metrics['defects_detected']} | Area: {metrics['damaged_area_percent']}%",
                    (28, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Severity: {metrics['severity_score']}/100 ({metrics['priority']})",
                    (28, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        out.write(annotated)

    cap.release()
    out.release()
    if os.path.exists(input_path):
        os.remove(input_path)

    return {
        "media_type": "video",
        **peak_metrics,
        "video_url": f"http://127.0.0.1:8000/outputs/{output_filename}",
        "annotated_image_base64": worst_frame_b64,
    }