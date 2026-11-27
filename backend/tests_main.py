"""Pengujian integrasi endpoint FastAPI GERDGuard (main.py).

Mencakup pengujian wajib:
1. Endpoint /health.
2. Gambar endoskopi yang diterima Gate (LOLOS_GATE).
3. Gambar non-endoskopi yang ditolak Gate (DITOLAK, B4 & Grad-CAM tidak dijalankan).
4. Prediksi GERD (lbaidcw.jpg).
5. Prediksi GERD_Normal (Gerd Normal_3579.jpg).
6. Penanganan gambar rusak (corrupted bytes).
7. Format file tidak didukung (MIME type selain JPEG/PNG/BMP).
8. Penanganan kegagalan Grad-CAM (GradCAMError).

Jalankan dengan:
    cd backend
    .venv\\Scripts\\python.exe tests_main.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from gradcam import GradCAMError

# ---- Test runner tracking ----
_results: list[tuple[str, bool, str]] = []


def _test(name: str):
    def decorator(fn):
        def wrapper():
            print(f"\n{'='*65}")
            print(f"TEST: {name}")
            print(f"{'='*65}")
            try:
                fn()
                _results.append((name, True, ""))
                print(f"  ✓ PASS")
            except Exception as e:
                tb = traceback.format_exc()
                _results.append((name, False, str(e)))
                print(f"  ✗ FAIL: {e}")
                print(tb)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Setup Test Client & Lifespan
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "testing_guard"
GERD_IMG_PATH = TEST_DIR / "valid_endoscopy" / "lbaidcw.jpg"
NORMAL_IMG_PATH = TEST_DIR / "valid_endoscopy" / "Gerd Normal_3579.jpg"
FOREIGN_IMG_PATH = TEST_DIR / "foreign_images" / "images.jpg"


print("Menginisialisasi TestClient & memuat model melalui lifespan...")
with TestClient(main.app) as client:

    # -----------------------------------------------------------------------
    # TEST 1: Endpoint /health
    # -----------------------------------------------------------------------
    @_test("Endpoint GET /health mengembalikan status 'ok' dan metadata model")
    def test_health():
        res = client.get("/health")
        assert res.status_code == 200, f"HTTP status: {res.status_code}"
        data = res.json()
        assert data.get("status") == "ok", f"Status: {data.get('status')}"
        assert data.get("model_loaded") is True
        assert data.get("image_type_gate_loaded") is True
        assert data.get("model_name") == "EfficientNetB5_GERD_Model"
        assert data.get("classifier_threshold") == 0.50
        assert data.get("gate_threshold") is not None
        print(f"  Response: {data}")

    test_health()

    # -----------------------------------------------------------------------
    # TEST 2: Gambar endoskopi yang diterima Gate
    # -----------------------------------------------------------------------
    @_test("Gambar endoskopi diterima oleh Image Type Gate")
    def test_gate_acceptance():
        assert GERD_IMG_PATH.is_file(), f"File tidak ditemukan: {GERD_IMG_PATH}"
        with open(GERD_IMG_PATH, "rb") as f:
            res = client.post(
                "/predict",
                files={"file": (GERD_IMG_PATH.name, f, "image/jpeg")},
            )
        assert res.status_code == 200, f"HTTP {res.status_code}: {res.text}"
        data = res.json()
        assert data.get("gate_status") == "LOLOS_GATE", f"gate_status: {data.get('gate_status')}"
        assert data.get("gate_score") is not None
        assert data.get("gate_score") < 0.50, f"Score non_endoscopy terlalu tinggi: {data.get('gate_score')}"
        # Gambar visualisasi harus tersedia
        assert data.get("original_image", "").startswith("data:image/")
        assert data.get("heatmap_image", "").startswith("data:image/jpeg;base64,")
        assert data.get("overlay_image", "").startswith("data:image/jpeg;base64,")
        print(f"  Gate status : {data.get('gate_status')}")
        print(f"  Gate score  : {data.get('gate_score')}")
        print(f"  Predicted   : {data.get('predicted_class')}")

    test_gate_acceptance()

    # -----------------------------------------------------------------------
    # TEST 3: Gambar non-endoskopi yang ditolak Gate
    # -----------------------------------------------------------------------
    @_test("Gambar non-endoskopi ditolak oleh Gate (B4 & Grad-CAM tidak dijalankan)")
    def test_gate_rejection():
        assert FOREIGN_IMG_PATH.is_file(), f"File tidak ditemukan: {FOREIGN_IMG_PATH}"
        with open(FOREIGN_IMG_PATH, "rb") as f:
            res = client.post(
                "/predict",
                files={"file": (FOREIGN_IMG_PATH.name, f, "image/jpeg")},
            )
        assert res.status_code == 200, f"HTTP {res.status_code}: {res.text}"
        data = res.json()
        assert data.get("status") == "DITOLAK", f"Status: {data.get('status')}"
        assert data.get("gate_status") == "DITOLAK"
        assert data.get("gate_score") >= 0.50, f"Gate score: {data.get('gate_score')}"
        # Pastikan EfficientNetB4 dan Grad-CAM TIDAK dijalankan
        assert data.get("predicted_class") is None, "predicted_class harus None saat ditolak"
        assert data.get("label") is None, "label harus None saat ditolak"
        assert data.get("probability_gerd") is None, "probability_gerd harus None saat ditolak"
        assert data.get("probability_normal") is None, "probability_normal harus None saat ditolak"
        assert data.get("confidence") is None, "confidence harus None saat ditolak"
        assert data.get("heatmap_image") is None, "heatmap_image harus None saat ditolak"
        assert data.get("overlay_image") is None, "overlay_image harus None saat ditolak"
        assert data.get("reason"), "Harus ada alasan penolakan"
        print(f"  Status        : {data.get('status')}")
        print(f"  Gate score    : {data.get('gate_score')}")
        print(f"  Alasan        : {data.get('reason')}")
        print(f"  B4/CAM bypassed: OK (semua field bernilai None)")

    test_gate_rejection()

    # -----------------------------------------------------------------------
    # TEST 4: Prediksi GERD (kelas 1)
    # -----------------------------------------------------------------------
    @_test("Prediksi kelas GERD (output sigmoid >= 0.50) & Grad-CAM kelas GERD")
    def test_prediction_gerd():
        with open(GERD_IMG_PATH, "rb") as f:
            res = client.post(
                "/predict",
                files={"file": (GERD_IMG_PATH.name, f, "image/jpeg")},
            )
        assert res.status_code == 200
        data = res.json()
        assert data.get("predicted_class") == "GERD", f"Expected GERD, got {data.get('predicted_class')}"
        assert data.get("label") == "GERD"
        assert data.get("probability_gerd") >= 0.50
        assert data.get("confidence") == data.get("probability_gerd")
        assert abs(data.get("probability_gerd") + data.get("probability_normal") - 1.0) < 1e-4
        print(f"  Predicted class   : {data.get('predicted_class')}")
        print(f"  P(GERD)           : {data.get('probability_gerd')}")
        print(f"  P(GERD_Normal)    : {data.get('probability_normal')}")
        print(f"  Confidence        : {data.get('confidence')}")
        print(f"  Heatmap base64 len: {len(data.get('heatmap_image'))}")
        print(f"  Overlay base64 len: {len(data.get('overlay_image'))}")

    test_prediction_gerd()

    # -----------------------------------------------------------------------
    # TEST 5: Prediksi GERD_Normal (kelas 0)
    # -----------------------------------------------------------------------
    @_test("Prediksi kelas GERD_Normal (output sigmoid < 0.50) & Grad-CAM kelas Normal")
    def test_prediction_normal():
        with open(NORMAL_IMG_PATH, "rb") as f:
            res = client.post(
                "/predict",
                files={"file": (NORMAL_IMG_PATH.name, f, "image/jpeg")},
            )
        assert res.status_code == 200
        data = res.json()
        assert data.get("predicted_class") == "GERD_Normal", f"Expected GERD_Normal, got {data.get('predicted_class')}"
        assert data.get("label") == "GERD_Normal"
        assert data.get("probability_gerd") < 0.50
        assert data.get("probability_normal") >= 0.50
        assert data.get("confidence") == data.get("probability_normal")
        print(f"  Predicted class   : {data.get('predicted_class')}")
        print(f"  P(GERD)           : {data.get('probability_gerd')}")
        print(f"  P(GERD_Normal)    : {data.get('probability_normal')}")
        print(f"  Confidence        : {data.get('confidence')}")

    test_prediction_normal()

    # -----------------------------------------------------------------------
    # TEST 6: Penanganan gambar rusak
    # -----------------------------------------------------------------------
    @_test("Penanganan gambar rusak menghasilkan HTTP 400 dengan pesan jelas")
    def test_corrupted_image():
        corrupted_bytes = b"CORRUPTED_HEADER_NOT_A_VALID_JPEG_IMAGE_DATA_12345"
        res = client.post(
            "/predict",
            files={"file": ("corrupted.jpg", corrupted_bytes, "image/jpeg")},
        )
        assert res.status_code == 400, f"Expected 400, got {res.status_code}"
        data = res.json()
        assert "detail" in data
        print(f"  HTTP status : {res.status_code}")
        print(f"  Error detail: {data.get('detail')}")

    test_corrupted_image()

    # -----------------------------------------------------------------------
    # TEST 7: Penanganan format file tidak didukung
    # -----------------------------------------------------------------------
    @_test("Format file tidak didukung menghasilkan HTTP 415")
    def test_unsupported_format():
        dummy_pdf = b"%PDF-1.4 dummy pdf bytes"
        res = client.post(
            "/predict",
            files={"file": ("report.pdf", dummy_pdf, "application/pdf")},
        )
        assert res.status_code == 415, f"Expected 415, got {res.status_code}"
        data = res.json()
        assert "format gambar tidak didukung" in data.get("detail", "").lower()
        print(f"  HTTP status : {res.status_code}")
        print(f"  Error detail: {data.get('detail')}")

    test_unsupported_format()

    # -----------------------------------------------------------------------
    # TEST 8: Penanganan kegagalan Grad-CAM
    # -----------------------------------------------------------------------
    @_test("Kegagalan Grad-CAM ditangani dengan HTTP 500 dan error yang jelas")
    def test_gradcam_failure():
        # Mock generate_gradcam agar me-raise GradCAMError
        with patch("main.generate_gradcam", side_effect=GradCAMError("Simulasi error komputasi gradien")):
            with open(GERD_IMG_PATH, "rb") as f:
                res = client.post(
                    "/predict",
                    files={"file": (GERD_IMG_PATH.name, f, "image/jpeg")},
                )
            assert res.status_code == 500, f"Expected 500, got {res.status_code}"
            data = res.json()
            assert "Grad-CAM" in data.get("detail", "")
            print(f"  HTTP status : {res.status_code}")
            print(f"  Error detail: {data.get('detail')}")

    test_gradcam_failure()


# ---------------------------------------------------------------------------
# Ringkasan Pengujian
# ---------------------------------------------------------------------------

print(f"\n{'='*65}")
print("RINGKASAN PENGUJIAN INTEGRASI MAIN.PY (TAHAP 4)")
print(f"{'='*65}")
passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
for name, ok, err in _results:
    status = "✓ PASS" if ok else f"✗ FAIL: {err}"
    print(f"  {status:12s}  {name}")
print(f"\nTotal: {passed} passed, {failed} failed, {len(_results)} total")
if failed > 0:
    sys.exit(1)
else:
    print("\n🎉 Semua 8 skenario pengujian integrasi BERHASIL!")
    sys.exit(0)
