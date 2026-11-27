
from __future__ import annotations

import base64
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from gradcam import GradCAMError, generate_gradcam
from guard import GERDGuard, InvalidImageError
from report_generator import generate_patient_report


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "model" / "gerd" /
    "model_gerd_efficientnetb5.keras"
)

GATE_MODEL_PATH = (
    BASE_DIR / "model" / "gate" /
    "image_type_classifier_efficientnetb4.keras"
)

GATE_CONFIG_PATH = (
    BASE_DIR / "model" / "gate" /
    "image_type_config.json"
)

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "image/bmp",
}

MODEL_NAME = "EfficientNetB5"
CLASSIFIER_THRESHOLD = 0.50

model: tf.keras.Model | None = None
guard: GERDGuard | None = None

lock = threading.Lock()


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, guard

    print("\n========== GERDGuard STARTUP ==========")

    for path in (
        MODEL_PATH,
        GATE_MODEL_PATH,
        GATE_CONFIG_PATH,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"File wajib tidak ditemukan: {path}"
            )

    try:
        print("[1/2] Loading EfficientNetB5...")

        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
        )

        if tuple(model.input_shape[1:]) != (456, 456, 3):
            raise RuntimeError(
                f"Input EfficientNetB5 tidak sesuai: "
                f"{model.input_shape}"
            )

        print("[2/2] Loading Image Type Gate...")

        guard = GERDGuard(
            GATE_MODEL_PATH,
            GATE_CONFIG_PATH,
        )

        print("\nGERDGuard API siap.")
        print("Model GERD      :", MODEL_PATH)
        print("Image-type gate :", GATE_MODEL_PATH)
        print("Gate threshold  :", guard.threshold)
        print("GERD threshold  :", CLASSIFIER_THRESHOLD)
        print("=======================================\n")

    except Exception:
        print("\n========== STARTUP ERROR ==========")
        traceback.print_exc()
        print("===================================\n")
        raise

    yield

    print("\nShutting down GERDGuard...")

    guard = None
    model = None

    tf.keras.backend.clear_session()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="GERDGuard API",
    description=(
        "GERD image classification with Image Type Gate, "
        "Grad-CAM explainability, and patient report."
    ),
    version="4.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# IMAGE ENCODING
# ============================================================

def _bgr_to_base64(
    image: np.ndarray,
    quality: int = 90,
) -> str:

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise RuntimeError(
            "Gagal mengubah gambar menjadi JPEG."
        )

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def _bytes_to_base64(
    data: bytes,
    mime_type: str,
) -> str:

    encoded = base64.b64encode(
        data
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ============================================================
# PREDICTION PIPELINE
# ============================================================

def run_prediction_pipeline(
    data: bytes,
    mime_type: str,
) -> dict:

    if model is None or guard is None:
        raise RuntimeError(
            "Model atau Image Type Gate belum siap."
        )

    print("\n========== PREDICTION START ==========")

    # --------------------------------------------------------
    # 1. IMAGE TYPE GATE
    # --------------------------------------------------------

    print("[1] Checking image type...")

    gate_result, pil_image = guard.check_bytes(data)

    gate_status = gate_result.get("status")
    gate_score = gate_result.get("skor_non_endoscopy")

    print("Gate status:", gate_status)
    print("Gate score :", gate_score)

    if (
        gate_status == "DITOLAK"
        and pil_image is None
    ):

        reason = gate_result.get("alasan", "")

        if (
            "rusak" in reason.lower()
            or "bukan gambar valid" in reason.lower()
            or "batas" in reason.lower()
            or "format gambar" in reason.lower()
        ):
            raise InvalidImageError(reason)

    # --------------------------------------------------------
    # 2. STOP IF IMAGE IS NOT ENDOSCOPY
    # --------------------------------------------------------

    if gate_status == "DITOLAK":

        print("[RESULT] Image rejected by Image Type Gate.")

        return {
            "status": "DITOLAK",
            "prediction": "DITOLAK",
            "predicted_class": None,
            "label": None,
            "probability_gerd": None,
            "probability_normal": None,
            "confidence": None,
            "gate_status": "DITOLAK",
            "gate_score": gate_score,
            "model_name": MODEL_NAME,
            "reason": gate_result.get(
                "alasan",
                "Gambar bukan endoskopi.",
            ),
            "original_image": None,
            "heatmap_image": None,
            "overlay_image": None,
            "report": None,
        }

    if pil_image is None:
        raise RuntimeError(
            "Image Gate meloloskan gambar tetapi "
            "objek gambar tidak tersedia."
        )

    # --------------------------------------------------------
    # 3. PREPROCESSING EFFICIENTNETB5
    # --------------------------------------------------------

    print("[2] Preparing EfficientNetB5 input...")

    arr_b5 = guard.prepare_classifier_image(
        pil_image
    )

    if arr_b5.shape != (456, 456, 3):
        raise RuntimeError(
            f"Input B5 tidak valid: {arr_b5.shape}"
        )

    batch = tf.convert_to_tensor(
        arr_b5[None, ...],
        dtype=tf.float32,
    )

    # --------------------------------------------------------
    # 4. CLASSIFICATION
    # --------------------------------------------------------

    print("[3] Running EfficientNetB5...")

    with lock:
        pred_tensor = model(
            batch,
            training=False,
        )

    prediction_values = np.asarray(
        pred_tensor
    ).reshape(-1)

    if prediction_values.size != 1:
        raise RuntimeError(
            "Output model GERD harus memiliki satu nilai sigmoid."
        )

    p_gerd = float(prediction_values[0])

    if not np.isfinite(p_gerd):
        raise RuntimeError(
            f"Output model tidak valid: {p_gerd}"
        )

    if not 0.0 <= p_gerd <= 1.0:
        raise RuntimeError(
        f"Probabilitas di luar rentang sigmoid: {p_gerd}"
    )

    p_normal = 1.0 - p_gerd

    predicted_class = (
        "GERD"
        if p_gerd >= CLASSIFIER_THRESHOLD
        else "GERD_Normal"
    )

    confidence = (
        p_gerd
        if predicted_class == "GERD"
        else p_normal
    )

    target_class_gerd = (
        predicted_class == "GERD"
    )

    print("P(GERD) :", p_gerd)
    print("P(Normal):", p_normal)
    print("Prediction:", predicted_class)

    # --------------------------------------------------------
    # 5. GRAD-CAM
    # --------------------------------------------------------

    print("[4] Generating Grad-CAM...")

    try:
        with lock:
            cam_result = generate_gradcam(
                model=model,
                img_array=arr_b5,
                original_image=pil_image,
                target_class_gerd=target_class_gerd,
            )

        if not isinstance(cam_result, dict):
            raise RuntimeError(
                "Output Grad-CAM harus berupa dictionary."
            )

        required_keys = {
            "heatmap",
            "overlay",
            "cam_raw"
        }

        missing_keys = required_keys.difference(
            cam_result.keys()
        )

        if missing_keys:
            raise RuntimeError(
                f"Output Grad-CAM tidak lengkap: {missing_keys}"
            )

        if (
        cam_result["heatmap"] is None
        or cam_result["overlay"] is None
        ):
            raise RuntimeError(
            "Grad-CAM menghasilkan gambar kosong."
        )

    # Validasi CAM raw
        cam_raw = np.asarray(
        cam_result["cam_raw"],
        dtype=np.float32
        )

        if cam_raw.ndim != 2 or cam_raw.size == 0:
            raise RuntimeError("Format cam_raw tidak valid.")

        if not np.isfinite(cam_raw).all():
            raise RuntimeError(
                "cam_raw mengandung NaN atau infinity."
        )

    except Exception as exc:
        print("\n========== GRAD-CAM ERROR ==========")
        traceback.print_exc()
        print("====================================\n")

        raise GradCAMError(
            f"Kegagalan komputasi Grad-CAM: {exc}"
        ) from exc

    # --------------------------------------------------------
    # 6. PATIENT REPORT GENERATOR
    # --------------------------------------------------------

    print("[5] Generating patient report...")

    try:
        patient_report = generate_patient_report(
            prediction=predicted_class,
            p_gerd=p_gerd,
            gate_status=gate_status,
            gradcam_result=cam_result,
        )

        print("\n========== PATIENT REPORT ==========")
        print(
            patient_report["explanation"]["summary"]
        )
        print("====================================\n")

    except Exception:
        print("\n========== REPORT GENERATOR ERROR ==========")
        traceback.print_exc()
        print("============================================\n")

        raise

    # --------------------------------------------------------
    # 7. ENCODE IMAGES
    # --------------------------------------------------------

    print("[6] Encoding result images...")

    original_b64 = _bytes_to_base64(
        data,
        mime_type,
    )

    heatmap_b64 = _bgr_to_base64(
        cam_result["heatmap"]
    )

    overlay_b64 = _bgr_to_base64(
        cam_result["overlay"]
    )

    # --------------------------------------------------------
    # 8. RESPONSE
    # --------------------------------------------------------

    result = {
        "status": predicted_class,
        "prediction": predicted_class,
        "predicted_class": predicted_class,
        "label": predicted_class,

        "probability_gerd": round(p_gerd, 6),
        "probability_normal": round(p_normal, 6),
        "confidence": round(confidence, 6),

        "gate_status": gate_status,
        "gate_score": gate_score,

        "model_name": MODEL_NAME,

        "original_image": original_b64,
        "heatmap_image": heatmap_b64,
        "overlay_image": overlay_b64,

        # Laporan individual dari Report Generator.
        "report": patient_report,
    }

    print("[SUCCESS] Prediction completed.")
    print("=====================================\n")

    return result


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():

    return {
        "message": "GERDGuard API aktif.",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():

    ready = (
        model is not None
        and guard is not None
    )

    return {
        "status": "ok" if ready else "not_ready",
        "model_loaded": model is not None,
        "image_type_gate_loaded": guard is not None,
        "model_name": MODEL_NAME,
        "model_path": str(
            MODEL_PATH.relative_to(BASE_DIR)
        ),
        "gate_model_path": str(
            GATE_MODEL_PATH.relative_to(BASE_DIR)
        ),
        "gate_threshold": (
            guard.threshold
            if guard is not None
            else None
        ),
        "classifier_threshold": CLASSIFIER_THRESHOLD,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    if model is None or guard is None:
        raise HTTPException(
            status_code=503,
            detail="Model atau filter belum siap.",
        )

    content_type = (
        file.content_type or ""
    ).lower()

    if content_type not in ALLOWED_MIME:
        await file.close()

        raise HTTPException(
            status_code=415,
            detail=(
                "Format gambar tidak didukung. "
                "Gunakan JPG, PNG, atau BMP."
            ),
        )

    try:
        data = await file.read(
            MAX_FILE_SIZE + 1
        )
    finally:
        await file.close()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="File gambar kosong.",
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Ukuran gambar maksimal 10 MB.",
        )

    try:
        result = await run_in_threadpool(
            run_prediction_pipeline,
            data,
            content_type,
        )

        return result

    except InvalidImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except GradCAMError as exc:
        print("\n========== GRAD-CAM ERROR ==========")
        traceback.print_exc()
        print("====================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM gagal: {exc}",
        ) from exc

    except Exception as exc:
        print("\n========== PREDICTION ERROR ==========")
        traceback.print_exc()
        print("======================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"Pemrosesan gambar gagal: {exc}",
        ) from exc
