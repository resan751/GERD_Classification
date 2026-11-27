
from __future__ import annotations

import numpy as np


REPORT_VERSION = "2.0"


REGION_NAMES = [
    ["kiri atas", "tengah atas", "kanan atas"],
    ["kiri tengah", "tengah citra", "kanan tengah"],
    ["kiri bawah", "tengah bawah", "kanan bawah"],
]


# ============================================================
# UTILITIES
# ============================================================

def _safe_probability(value):
    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return float(np.clip(value, 0.0, 1.0))

    except (TypeError, ValueError):
        return None


def _clean_cam(cam):
    if cam is None:
        return None

    cam = np.asarray(cam, dtype=np.float32)

    cam = np.squeeze(cam)

    if cam.ndim != 2 or cam.size == 0:
        raise ValueError("Format Grad-CAM tidak valid.")

    if not np.isfinite(cam).all():
        raise ValueError("Grad-CAM mengandung nilai tidak valid.")

    return np.maximum(cam, 0.0)


# ============================================================
# HEATMAP ANALYSIS
# ============================================================

def analyze_heatmap(cam):
    cam = _clean_cam(cam)

    if cam is None:
        return {
            "available": False,
            "dominant_region": None,
            "activation_pattern": "unavailable",
            "region_distribution": [],
            "top_regions": [],
            "coverage_percentage": None,
            "explanation": (
                "Visualisasi Grad-CAM tidak tersedia."
            ),
        }

    maximum = float(cam.max())

    if maximum <= 1e-8:
        return {
            "available": False,
            "dominant_region": None,
            "activation_pattern": "zero_activation",
            "region_distribution": [],
            "top_regions": [],
            "coverage_percentage": 0.0,
            "explanation": (
                "Grad-CAM tidak menghasilkan aktivasi "
                "yang dapat dianalisis."
            ),
        }

    # Normalisasi CAM.
    cam = cam / maximum

    height, width = cam.shape

    region_scores = []

    # Membagi heatmap menjadi sembilan area.
    for row in range(3):
        for col in range(3):

            y1 = round(row * height / 3)
            y2 = round((row + 1) * height / 3)

            x1 = round(col * width / 3)
            x2 = round((col + 1) * width / 3)

            region = cam[y1:y2, x1:x2]

            if region.size == 0:
                score = 0.0
            else:
                score = float(np.mean(region))

            region_scores.append({
                "region": REGION_NAMES[row][col],
                "score": score,
            })

    total_score = sum(
        item["score"]
        for item in region_scores
    )

    if total_score <= 1e-8:
        return {
            "available": False,
            "dominant_region": None,
            "activation_pattern": "zero_activation",
            "region_distribution": [],
            "top_regions": [],
            "coverage_percentage": 0.0,
            "explanation": (
                "Distribusi aktivasi tidak dapat dihitung."
            ),
        }

    # Persentase distribusi setiap area.
    for item in region_scores:
        item["percentage"] = round(
            item["score"] / total_score * 100,
            2,
        )

    region_scores.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    top_regions = region_scores[:3]

    dominant = top_regions[0]

    dominant_percentage = dominant["percentage"]

    top_three_percentage = sum(
        item["percentage"]
        for item in top_regions
    )

    # Pola distribusi aktivasi.
    if dominant_percentage >= 40:
        pattern = "localized"

        pattern_text = (
            "Aktivasi relatif terkonsentrasi pada satu area."
        )

    elif top_three_percentage >= 65:
        pattern = "multifocal"

        pattern_text = (
            "Aktivasi terkonsentrasi pada beberapa area utama."
        )

    else:
        pattern = "distributed"

        pattern_text = (
            "Aktivasi tersebar pada beberapa bagian citra."
        )

    # Luas area dengan aktivasi relatif kuat.
    active_threshold = 0.5

    coverage = float(
        np.mean(cam >= active_threshold) * 100
    )

    # Deskripsi area tertinggi.
    top_region_text = ", ".join(
        f"{item['region']} ({item['percentage']:.2f}%)"
        for item in top_regions
    )

    explanation = (
        f"Aktivasi tertinggi berada di bagian "
        f"{dominant['region']} dengan distribusi "
        f"{dominant_percentage:.2f}%. "
        f"Area dominan berikutnya adalah {top_region_text}. "
        f"{pattern_text}"
    )

    return {
        "available": True,
        "dominant_region": dominant["region"],
        "dominant_activation_percentage": dominant_percentage,
        "activation_pattern": pattern,
        "activation_pattern_description": pattern_text,
        "coverage_percentage": round(coverage, 2),
        "region_distribution": region_scores,
        "top_regions": top_regions,
        "explanation": explanation,
    }


# ============================================================
# CLASSIFICATION INTERPRETATION
# ============================================================

def analyze_classification(prediction, p_gerd):

    probability = _safe_probability(p_gerd)

    if probability is None:
        raise ValueError(
            "Probabilitas GERD tidak valid."
        )

    if prediction not in ("GERD", "GERD_Normal"):
        raise ValueError(
            f"Kelas prediksi tidak dikenal: {prediction}"
        )

    p_normal = 1.0 - probability

    confidence = (
        probability
        if prediction == "GERD"
        else p_normal
    )

    if confidence >= 0.90:
        confidence_level = "Tinggi"
        confidence_description = (
            "Model menunjukkan nilai probabilitas "
            "yang tinggi terhadap kelas prediksi."
        )

    elif confidence >= 0.70:
        confidence_level = "Sedang"
        confidence_description = (
            "Model menunjukkan kecenderungan terhadap "
            "kelas prediksi, tetapi hasil perlu ditinjau."
        )

    else:
        confidence_level = "Rendah"
        confidence_description = (
            "Model belum menunjukkan kecenderungan "
            "prediksi yang kuat."
        )

    if prediction == "GERD":

        explanation = (
            "EfficientNetB5 mengklasifikasikan citra "
            "sebagai GERD."
        )

        detail = (
            f"Probabilitas GERD sebesar {probability * 100:.2f}%, "
            f"sedangkan probabilitas GERD_Normal sebesar "
            f"{p_normal * 100:.2f}%."
        )

    else:

        explanation = (
            "EfficientNetB5 mengklasifikasikan citra "
            "sebagai GERD_Normal."
        )

        detail = (
            f"Probabilitas GERD_Normal sebesar "
            f"{p_normal * 100:.2f}%, sedangkan "
            f"probabilitas GERD sebesar "
            f"{probability * 100:.2f}%."
        )

    return {
        "predicted_class": prediction,
        "p_gerd": round(probability, 6),
        "p_normal": round(p_normal, 6),
        "confidence": round(confidence, 6),
        "confidence_percentage": round(
            confidence * 100, 2
        ),
        "confidence_level": confidence_level,
        "confidence_description": confidence_description,
        "explanation": explanation,
        "detail": detail,
    }


# ============================================================
# INDIVIDUAL REPORT GENERATOR
# ============================================================

def generate_patient_report(
    prediction,
    p_gerd,
    gate_status="LOLOS_GATE",
    gradcam_result=None,
):

    # --------------------------------------------------------
    # 1. CLASSIFICATION
    # --------------------------------------------------------

    classification = analyze_classification(
        prediction,
        p_gerd,
    )

    # --------------------------------------------------------
    # 2. IMAGE GATE
    # --------------------------------------------------------

    if gate_status == "LOLOS_GATE":

        gate_explanation = (
            "Citra berhasil melewati pemeriksaan "
            "Image Type Gate."
        )

    else:

        gate_explanation = (
            f"Status pemeriksaan citra adalah {gate_status}."
        )

    # --------------------------------------------------------
    # 3. GRAD-CAM
    # --------------------------------------------------------

    cam = None

    if isinstance(gradcam_result, dict):
        cam = gradcam_result.get("cam_raw")

    visualization = analyze_heatmap(cam)

    # --------------------------------------------------------
    # 4. DYNAMIC INTERPRETATION
    # --------------------------------------------------------

    prediction_text = classification["explanation"]

    probability_text = classification["detail"]

    if visualization["available"]:

        heatmap_text = visualization["explanation"]

        summary = (
            f"{prediction_text} "
            f"{probability_text} "
            f"{heatmap_text}"
        )

        visual_interpretation = (
            f"Model menunjukkan aktivasi tertinggi "
            f"pada area {visualization['dominant_region']}. "
            f"Distribusi aktivasi area tersebut sebesar "
            f"{visualization['dominant_activation_percentage']:.2f}%. "
            f"Pola aktivasi termasuk "
            f"{visualization['activation_pattern']}."
        )

    else:

        heatmap_text = (
            "Interpretasi visual Grad-CAM belum tersedia "
            "karena aktivasi tidak dapat dianalisis."
        )

        summary = (
            f"{prediction_text} {probability_text} "
            f"{heatmap_text}"
        )

        visual_interpretation = heatmap_text

    # --------------------------------------------------------
    # 5. CLINICAL FINDINGS
    # --------------------------------------------------------

    clinical_note = (
        "Hasil ini merupakan keluaran model klasifikasi "
        "berbasis citra. Grad-CAM menunjukkan area visual "
        "yang berkontribusi terhadap prediksi model. "
        "Area aktivasi tidak dapat dianggap sebagai bukti "
        "langsung adanya erosi, ulserasi, inflamasi, atau "
        "kelainan anatomi. Penentuan temuan klinis dan "
        "diagnosis memerlukan evaluasi tenaga medis."
    )

    findings = {
        "lesion_detection": "not_available",
        "erosion_detection": "not_available",
        "inflammation_detection": "not_available",
        "ulcer_detection": "not_available",
        "la_grade": "not_available",
        "explanation": (
            "Model klasifikasi saat ini belum dilatih "
            "untuk mendeteksi erosi, inflamasi, ulserasi, "
            "atau menentukan LA Grade."
        ),
    }

    # --------------------------------------------------------
    # 6. FINAL REPORT
    # --------------------------------------------------------

    return {
        "report_version": REPORT_VERSION,

        "classification": classification,

        "image_gate": {
            "status": gate_status,
            "explanation": gate_explanation,
        },

        "visualization": visualization,

        "findings": findings,

        "explanation": {
            "summary": summary,
            "prediction_interpretation": prediction_text,
            "probability_interpretation": probability_text,
            "visual_interpretation": visual_interpretation,
            "clinical_note": clinical_note,
        },
    }
