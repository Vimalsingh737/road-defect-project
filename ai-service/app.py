import os
import gc
import uuid
import base64
import subprocess
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Road Defect AI Service (Lightweight ONNX)")

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

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.enable_cpu_mem_arena = False

session = ort.InferenceSession("best.onnx", sess_options=sess_options, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


@app.get("/")
def home():
    return {"status": "online", "engine": "Pure ONNX Runtime"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def run_yolo_seg(frame, conf_thres=0.15, iou_thres=0.45):
    orig_h, orig_w = frame.shape[:2]
    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(img_rgb, (2, 0, 1))[None, ...]

    outputs = session.run(None, {input_name: tensor})
    preds = np.squeeze(outputs[0])
    protos = np.squeeze(outputs[1])

    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    num_mask = protos.shape[0]
    nc = preds.shape[1] - 4 - num_mask

    boxes_xywh = preds[:, :4]
    scores = np.max(preds[:, 4:4 + nc], axis=1)
    mask_coeffs = preds[:, 4 + nc:]

    keep = scores > conf_thres
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    mask_coeffs = mask_coeffs[keep]

    annotated = frame.copy()

    if len(scores) == 0:
        return annotated, {
            "defects_detected": 0,
            "defect_pixels": 0,
            "damaged_area_percent": 0.0,
            "severity_score": 0.0,
            "priority": "Clear",
        }

    boxes_nms = []
    for b in boxes_xywh:
        x1 = float(b[0] - b[2] / 2)
        y1 = float(b[1] - b[3] / 2)
        boxes_nms.append([x1, y1, float(b[2]), float(b[3])])

    indices = cv2.dnn.NMSBoxes(boxes_nms, scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return annotated, {
            "defects_detected": 0,
            "defect_pixels": 0,
            "damaged_area_percent": 0.0,
            "severity_score": 0.0,
            "priority": "Clear",
        }

    indices = np.array(indices).flatten()[:15]
    boxes_xywh = boxes_xywh[indices]
    scores = scores[indices]
    mask_coeffs = mask_coeffs[indices]

    protos_flat = protos.reshape(num_mask, -1)
    masks_raw = sigmoid(np.matmul(mask_coeffs, protos_flat)).reshape(-1, 160, 160)

    combined_mask_160 = np.zeros((160, 160), dtype=np.uint8)
    for i, box in enumerate(boxes_xywh):
        cx, cy, bw, bh = box
        mx1 = max(0, int((cx - bw / 2) / 4))
        my1 = max(0, int((cy - bh / 2) / 4))
        mx2 = min(160, int((cx + bw / 2) / 4))
        my2 = min(160, int((cy + bh / 2) / 4))

        m = np.zeros((160, 160), dtype=np.uint8)
        m[my1:my2, mx1:mx2] = (masks_raw[i, my1:my2, mx1:mx2] > 0.5).astype(np.uint8)
        combined_mask_160 = np.maximum(combined_mask_160, m)

    defect_ratio = float(np.sum(combined_mask_160)) / (160.0 * 160.0)
    area_ratio = round(defect_ratio * 100, 2)
    defect_pixels = int(defect_ratio * orig_w * orig_h)
    severity_score = min(100.0, round(area_ratio * 4.5, 1))
    num_defects = len(indices)

    priority = (
        "Critical" if severity_score > 70
        else "Medium" if severity_score > 35
        else "Low" if severity_score > 0
        else "Clear"
    )

    full_mask = cv2.resize(combined_mask_160, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    overlay = annotated.copy()
    overlay[full_mask == 1] = [0, 69, 255]
    annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

    for i, box in enumerate(boxes_xywh):
        cx, cy, bw, bh = box
        x1 = max(0, int((cx - bw / 2) * orig_w / 640))
        y1 = max(0, int((cy - bh / 2) * orig_h / 640))
        x2 = min(orig_w, int((cx + bw / 2) * orig_w / 640))
        y2 = min(orig_h, int((cy + bh / 2) * orig_h / 640))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            f"Defect {scores[i]:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    return annotated, {
        "defects_detected": int(num_defects),
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

    H, W = img.shape[:2]
    if W > 640:
        scale = 640 / W
        img = cv2.resize(img, (640, int(H * scale)))

    annotated_img, metrics = run_yolo_seg(img, conf_thres=conf)
    _, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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
    raw_path = os.path.join(OUTPUT_DIR, f"raw_{file_id}.mp4")
    output_filename = f"out_{file_id}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 480
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    # Even dimensions required by H.264 encoder
    width = 480 if orig_w > 480 else (orig_w // 2) * 2
    height = (int(orig_h * (width / orig_w)) // 2) * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    frame_idx = 0
    max_frames = 90
    frame_stride = 5

    peak_metrics = {
        "defects_detected": 0,
        "defect_pixels": 0,
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
            last_annotated, last_metrics = run_yolo_seg(frame, conf_thres=conf)
            if last_metrics["severity_score"] >= peak_metrics["severity_score"]:
                peak_metrics = last_metrics
                _, buf = cv2.imencode(".jpg", last_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                worst_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        annotated = last_annotated.copy() if last_annotated is not None else frame
        color = (0, 0, 255) if last_metrics["severity_score"] > 70 else (0, 165, 255) if last_metrics["severity_score"] > 35 else (0, 255, 0)
        cv2.rectangle(annotated, (8, 8), (340, 74), (20, 20, 20), -1)
        cv2.rectangle(annotated, (8, 8), (340, 74), color, 2)
        cv2.putText(annotated, f"Defects: {last_metrics['defects_detected']} | Area: {last_metrics['damaged_area_percent']}%",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(annotated, f"Severity: {last_metrics['severity_score']}/100 ({last_metrics['priority']})",
                    (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        out.write(annotated)

    cap.release()
    out.release()

    # Convert to browser-playable H.264 MP4 using ffmpeg
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", raw_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", "-crf", "28",
            "-movflags", "+faststart", output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    final_file = output_path if os.path.exists(output_path) else raw_path
    with open(final_file, "rb") as vf:
        video_b64 = base64.b64encode(vf.read()).decode("utf-8")

    for p in (input_path, raw_path, output_path):
        if os.path.exists(p):
            os.remove(p)
    gc.collect()

    return {
        "media_type": "video",
        **peak_metrics,
        "video_url": f"data:video/mp4;base64,{video_b64}",
        "annotated_image_base64": worst_frame_b64,
    }