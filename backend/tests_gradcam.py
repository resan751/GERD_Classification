"""Pengujian Grad-CAM untuk model GERD EfficientNetB5.

Jalankan:
    cd backend
    .venv\\Scripts\\python.exe tests_gradcam.py

Memerlukan model asli di:
    model/combined/gerdguard_b5_best.keras
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

# ---- Hasil test tracking ----
_results: list[tuple[str, bool, str]] = []


def _test(name: str):
    """Decorator sederhana untuk test runner."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print(f"{'='*60}")
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
# Muat model dan buat gambar test (sekali saja)
# ---------------------------------------------------------------------------

print("Memuat model...")
t0 = time.time()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "combined" / "gerdguard_b5_best.keras"

import tensorflow as tf
model = tf.keras.models.load_model(
    str(MODEL_PATH), compile=False
)
print(f"Model dimuat dalam {time.time() - t0:.1f}s")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

from gradcam import generate_gradcam, GradCAMError, _find_target_layer
from PIL import Image

# Buat gambar sintetis dengan pola yang punya gradien
# (bukan noise murni, agar heatmap tidak sepenuhnya uniform)
np.random.seed(42)
_synthetic_rgb = np.random.randint(50, 200, (600, 800, 3), dtype=np.uint8)
# Tambah area terang di tengah untuk memberi sinyal spatial
_synthetic_rgb[200:400, 300:500, :] = 240
_pil_image = Image.fromarray(_synthetic_rgb, "RGB")

# Preprocessing 456×456 untuk model (sama seperti guard.prepare_classifier_image)
_img_456 = np.asarray(
    _pil_image.resize((456, 456), Image.Resampling.BILINEAR),
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# TEST 1: Verifikasi target layer ditemukan dan valid
# ---------------------------------------------------------------------------

@_test("Target layer ditemukan dan merupakan Conv2D")
def test_target_layer():
    layer = _find_target_layer(model)
    assert layer.name == "top_conv", f"Expected 'top_conv', got '{layer.name}'"
    assert "Conv" in layer.__class__.__name__, (
        f"Bukan Conv layer: {layer.__class__.__name__}"
    )
    print(f"  Layer: {layer.name} ({layer.__class__.__name__})")

test_target_layer()


# ---------------------------------------------------------------------------
# TEST 2: Validasi feature extractor sub-model
# ---------------------------------------------------------------------------

@_test("Grad-CAM feature extractor menghasilkan shape aktivasi yang benar")
def test_gradcam_model_build():
    effnet = model.get_layer("efficientnetb5")
    top_conv = effnet.get_layer("top_conv")
    feat_model = tf.keras.Model(inputs=effnet.input, outputs=top_conv.output)
    conv_out = feat_model(tf.zeros((1, 456, 456, 3)), training=False)
    print(f"  Conv activations shape: {conv_out.shape}")
    assert conv_out.shape[1:3] == (15, 15), f"Spatial dims wrong: {conv_out.shape}"
    assert conv_out.shape[3] == 2048, f"Channels wrong: {conv_out.shape}"

test_gradcam_model_build()


# ---------------------------------------------------------------------------
# TEST 3: Grad-CAM untuk prediksi GERD (kelas 1)
# ---------------------------------------------------------------------------

@_test("Grad-CAM untuk kelas GERD")
def test_gradcam_gerd():
    result = generate_gradcam(
        model=model,
        img_array=_img_456,
        original_image=_pil_image,
        target_class_gerd=True,
    )

    # Periksa semua kunci ada
    required_keys = {"heatmap", "overlay", "cam_raw", "score", "p_gerd", "target", "prediction"}
    missing = required_keys - set(result.keys())
    assert not missing, f"Kunci hilang: {missing}"

    # Periksa target
    assert result["target"] == "GERD", f"Target salah: {result['target']}"

    # Periksa cam_raw
    cam = result["cam_raw"]
    assert cam.dtype == np.float32, f"cam_raw dtype: {cam.dtype}"
    assert cam.shape == (15, 15), f"cam_raw shape: {cam.shape}"
    assert 0.0 <= cam.min() <= cam.max() <= 1.0, (
        f"cam_raw range: [{cam.min()}, {cam.max()}]"
    )
    print(f"  cam_raw: shape={cam.shape}, min={cam.min():.4f}, max={cam.max():.4f}")

    # Periksa heatmap
    hm = result["heatmap"]
    assert hm.dtype == np.uint8, f"heatmap dtype: {hm.dtype}"
    assert hm.shape == (600, 800, 3), f"heatmap shape: {hm.shape} (expected (600, 800, 3))"
    assert hm.size > 0, "heatmap kosong"
    print(f"  heatmap: shape={hm.shape}, dtype={hm.dtype}")

    # Periksa overlay
    ov = result["overlay"]
    assert ov.dtype == np.uint8, f"overlay dtype: {ov.dtype}"
    assert ov.shape == (600, 800, 3), f"overlay shape: {ov.shape} (expected (600, 800, 3))"
    assert ov.size > 0, "overlay kosong"
    print(f"  overlay: shape={ov.shape}, dtype={ov.dtype}")

    # Periksa score
    score = result["score"]
    assert 0.0 <= score <= 1.0, f"score di luar range: {score}"
    print(f"  score (GERD): {score:.6f}")
    print(f"  p_gerd: {result['p_gerd']:.6f}")
    print(f"  prediction: {result['prediction']}")

test_gradcam_gerd()


# ---------------------------------------------------------------------------
# TEST 4: Grad-CAM untuk prediksi GERD_Normal (kelas 0)
# ---------------------------------------------------------------------------

@_test("Grad-CAM untuk kelas GERD_Normal")
def test_gradcam_normal():
    result = generate_gradcam(
        model=model,
        img_array=_img_456,
        original_image=_pil_image,
        target_class_gerd=False,
    )

    assert result["target"] == "GERD_Normal"

    cam = result["cam_raw"]
    assert cam.shape == (15, 15)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0

    # Score untuk GERD_Normal harus = 1 - p_gerd
    expected_score = 1.0 - result["p_gerd"]
    assert abs(result["score"] - expected_score) < 1e-4, (
        f"Score mismatch: {result['score']} != 1 - {result['p_gerd']} = {expected_score}"
    )
    print(f"  score (GERD_Normal): {result['score']:.6f}")
    print(f"  p_gerd: {result['p_gerd']:.6f}")
    print(f"  1 - p_gerd: {expected_score:.6f}")

    # Heatmap dan overlay ukuran asli
    assert result["heatmap"].shape == (600, 800, 3)
    assert result["overlay"].shape == (600, 800, 3)
    print(f"  heatmap shape: {result['heatmap'].shape}")
    print(f"  overlay shape: {result['overlay'].shape}")

test_gradcam_normal()


# ---------------------------------------------------------------------------
# TEST 5: Bentuk tensor aktivasi dan gradien
# ---------------------------------------------------------------------------

@_test("Bentuk tensor aktivasi dan gradien benar")
def test_activation_gradient_shapes():
    effnet = model.get_layer("efficientnetb5")
    top_conv = effnet.get_layer("top_conv")
    top_bn = effnet.get_layer("top_bn")
    top_act = effnet.get_layer("top_activation")
    try:
        gap = model.get_layer("global_average_pooling2d")
    except ValueError:
        gap = effnet.get_layer("avg_pool")
    dropout = model.get_layer("dropout")
    dense = model.get_layer("gerd_probability")
    aug_layers = [
        l for l in model.layers
        if any(k in l.name for k in ["flip", "rotation", "zoom", "contrast", "augmentation"])
    ]

    feat_model = tf.keras.Model(inputs=effnet.input, outputs=top_conv.output)
    img_tensor = tf.convert_to_tensor(_img_456[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        h = img_tensor
        for aug in aug_layers:
            h = aug(h, training=False)
        conv_out = feat_model(h, training=False)
        tape.watch(conv_out)
        h2 = top_bn(conv_out, training=False)
        h2 = top_act(h2)
        h2 = gap(h2)
        h2 = dropout(h2, training=False)
        pred = dense(h2, training=False)
        target = pred[0, 0]

    grads = tape.gradient(target, conv_out)

    assert conv_out.shape == (1, 15, 15, 2048), f"conv_out shape: {conv_out.shape}"
    assert grads is not None, "Gradien None!"
    assert grads.shape == (1, 15, 15, 2048), f"grads shape: {grads.shape}"

    grads_np = grads.numpy()
    assert not np.allclose(grads_np, 0), "Semua gradien nol — graph tidak terhubung"

    print(f"  conv_out shape: {conv_out.shape}")
    print(f"  grads shape: {grads.shape}")
    print(f"  grads min: {grads_np.min():.6e}")
    print(f"  grads max: {grads_np.max():.6e}")
    print(f"  grads mean: {grads_np.mean():.6e}")

test_activation_gradient_shapes()


# ---------------------------------------------------------------------------
# TEST 6: Rentang nilai heatmap
# ---------------------------------------------------------------------------

@_test("Rentang nilai heatmap [0, 255] uint8")
def test_heatmap_range():
    result = generate_gradcam(model, _img_456, _pil_image, target_class_gerd=True)
    hm = result["heatmap"]
    assert hm.dtype == np.uint8
    assert hm.min() >= 0
    assert hm.max() <= 255
    # Heatmap harus memiliki variasi (bukan solid satu warna)
    unique = len(np.unique(hm))
    print(f"  Heatmap range: [{hm.min()}, {hm.max()}]")
    print(f"  Unique values: {unique}")
    assert unique > 1, "Heatmap solid satu warna — kemungkinan ada masalah"

test_heatmap_range()


# ---------------------------------------------------------------------------
# TEST 7: Overlay valid
# ---------------------------------------------------------------------------

@_test("Overlay menggabungkan gambar asli dan heatmap")
def test_overlay_valid():
    result = generate_gradcam(model, _img_456, _pil_image, target_class_gerd=True)
    ov = result["overlay"]
    hm = result["heatmap"]

    import cv2

    # Overlay harus berbeda dari heatmap murni dan gambar asli murni
    orig_bgr = cv2.cvtColor(np.array(_pil_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    orig_resized = cv2.resize(orig_bgr, (ov.shape[1], ov.shape[0]))

    diff_from_heatmap = np.mean(np.abs(ov.astype(float) - hm.astype(float)))
    diff_from_original = np.mean(np.abs(ov.astype(float) - orig_resized.astype(float)))

    print(f"  Mean diff overlay vs heatmap:  {diff_from_heatmap:.2f}")
    print(f"  Mean diff overlay vs original: {diff_from_original:.2f}")

    assert diff_from_heatmap > 1.0, "Overlay identik dengan heatmap"
    assert diff_from_original > 1.0, "Overlay identik dengan gambar asli"

test_overlay_valid()


# ---------------------------------------------------------------------------
# TEST 8: Grad-CAM tidak mengubah probabilitas prediksi
# ---------------------------------------------------------------------------

@_test("Grad-CAM tidak mengubah probabilitas prediksi model")
def test_prediction_unchanged():
    # Prediksi sebelum Grad-CAM
    pred_before = model(_img_456[None, ...], training=False)
    p_before = float(np.asarray(pred_before).reshape(-1)[0])

    # Jalankan Grad-CAM
    result = generate_gradcam(model, _img_456, _pil_image, target_class_gerd=True)

    # Prediksi setelah Grad-CAM
    pred_after = model(_img_456[None, ...], training=False)
    p_after = float(np.asarray(pred_after).reshape(-1)[0])

    diff = abs(p_before - p_after)
    print(f"  p_gerd sebelum Grad-CAM: {p_before:.8f}")
    print(f"  p_gerd setelah Grad-CAM: {p_after:.8f}")
    print(f"  p_gerd dari result:      {result['p_gerd']:.8f}")
    print(f"  Selisih before/after:    {diff:.2e}")

    assert diff < 1e-5, (
        f"Prediksi berubah setelah Grad-CAM: {p_before:.8f} → {p_after:.8f}"
    )

test_prediction_unchanged()


# ---------------------------------------------------------------------------
# TEST 9: Input numpy array sebagai original_image
# ---------------------------------------------------------------------------

@_test("original_image berupa numpy array uint8 juga berfungsi")
def test_numpy_original():
    orig_np = np.array(_pil_image, dtype=np.uint8)
    result = generate_gradcam(model, _img_456, orig_np, target_class_gerd=True)

    assert result["heatmap"].shape == (600, 800, 3)
    assert result["overlay"].shape == (600, 800, 3)
    print(f"  OK — shapes match original image dimensions")

test_numpy_original()


# ---------------------------------------------------------------------------
# TEST 10: Error handling — input tidak valid
# ---------------------------------------------------------------------------

@_test("Error handling untuk input tidak valid")
def test_error_handling():
    # img_array None
    try:
        generate_gradcam(model, None, _pil_image)
        assert False, "Seharusnya raise ValueError"
    except ValueError as e:
        print(f"  img_array=None → ValueError: {e}")

    # img_array shape salah
    try:
        bad_arr = np.zeros((380, 380, 3), dtype=np.float32)
        generate_gradcam(model, bad_arr, _pil_image)
        assert False, "Seharusnya raise ValueError"
    except ValueError as e:
        print(f"  Shape (380,380,3) → ValueError: {e}")

    # img_array dtype salah
    try:
        bad_dtype = np.zeros((456, 456, 3), dtype=np.uint8)
        generate_gradcam(model, bad_dtype, _pil_image)
        assert False, "Seharusnya raise ValueError"
    except ValueError as e:
        print(f"  dtype uint8 → ValueError: {e}")

    # original_image bukan gambar
    try:
        generate_gradcam(model, _img_456, "bukan_gambar")
        assert False, "Seharusnya raise ValueError"
    except ValueError as e:
        print(f"  original_image=str → ValueError: {e}")

test_error_handling()


# ---------------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("RINGKASAN PENGUJIAN")
print(f"{'='*60}")
passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
for name, ok, err in _results:
    status = "✓ PASS" if ok else f"✗ FAIL: {err}"
    print(f"  {status:12s}  {name}")
print(f"\nTotal: {passed} passed, {failed} failed, {len(_results)} total")
if failed > 0:
    sys.exit(1)
else:
    print("\n🎉 Semua pengujian BERHASIL!")
    sys.exit(0)
