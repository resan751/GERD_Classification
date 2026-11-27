
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


TARGET_BACKBONE = "efficientnetb5"
TARGET_LAYER = "top_conv"
INPUT_SIZE = (456, 456)
OVERLAY_ALPHA = 0.45


class GradCAMError(RuntimeError):
    pass


class GradCAMExplainer:

    def __init__(self, model):

        self.model = model

        try:
            self.backbone = model.get_layer(TARGET_BACKBONE)
            self.top_conv = self.backbone.get_layer(TARGET_LAYER)
            self.top_bn = self.backbone.get_layer("top_bn")
            self.top_activation = self.backbone.get_layer(
                "top_activation"
            )

            self.gap = model.get_layer(
                "global_average_pooling2d"
            )
            self.dropout = model.get_layer("dropout")
            self.classifier = model.get_layer("gerd_probability")

        except Exception as exc:
            raise GradCAMError(
                f"Gagal menemukan layer model: {exc}"
            ) from exc

        try:
            self.feature_model = tf.keras.Model(
                inputs=self.backbone.inputs,
                outputs=self.top_conv.output,
                name="gradcam_feature_extractor",
            )

        except Exception as exc:
            raise GradCAMError(
                f"Gagal membangun feature extractor: {exc}"
            ) from exc

        if not hasattr(self.classifier, "kernel"):
            raise GradCAMError(
                "Layer gerd_probability tidak memiliki kernel Dense."
            )

        @tf.function(reduce_retracing=True)
        def gradcam_step(image_tensor, target_is_gerd):

            with tf.GradientTape() as tape:

                features = self.feature_model(
                    image_tensor,
                    training=False,
                )

                tape.watch(features)

                x = self.top_bn(
                    features,
                    training=False,
                )

                x = self.top_activation(x)
                x = self.gap(x)
                x = self.dropout(x, training=False)

                prediction = self.classifier(
                    x,
                    training=False,
                )

                probability = tf.cast(
                    prediction[0, 0],
                    tf.float32,
                )

                # Samakan dtype sebelum operasi MatMul.
                x32 = tf.cast(x, tf.float32)

                kernel32 = tf.cast(
                    self.classifier.kernel,
                    tf.float32,
                )

                logits = tf.linalg.matmul(
                    x32,
                    kernel32,
                )

                if self.classifier.use_bias:
                    bias32 = tf.cast(
                        self.classifier.bias,
                        tf.float32,
                    )
                    logits = logits + bias32

                logit = logits[0, 0]

                target_score = tf.where(
                    target_is_gerd,
                    logit,
                    -logit,
                )

            gradients = tape.gradient(
                target_score,
                features,
            )

            if gradients is None:
                raise GradCAMError(
                    "Gradient tidak terhubung dengan top_conv."
                )

            # Konversi untuk stabilitas perhitungan CAM.
            gradients = tf.cast(
                gradients,
                tf.float32,
            )

            features32 = tf.cast(
                features,
                tf.float32,
            )

            weights = tf.reduce_mean(
                gradients,
                axis=(1, 2),
                keepdims=True,
            )

            cam = tf.reduce_sum(
                features32 * weights,
                axis=-1,
            )[0]

            cam = tf.nn.relu(cam)

            maximum = tf.reduce_max(cam)

            cam = tf.cond(
                maximum > 1e-8,
                lambda: cam / maximum,
                lambda: tf.zeros_like(cam),
            )

            return probability, cam, target_score

        self._gradcam_step = gradcam_step

    def warmup(self):

        dummy = tf.zeros(
            (1, 456, 456, 3),
            dtype=tf.float32,
        )

        self._gradcam_step(
            dummy,
            tf.constant(True),
        )

        self._gradcam_step(
            dummy,
            tf.constant(False),
        )

        print("[GradCAM] Warmup berhasil.")


_EXPLAINER_CACHE = {}


def get_or_create_explainer(model):

    key = id(model)

    cached = _EXPLAINER_CACHE.get(key)

    if cached is None or cached.model is not model:

        print("[GradCAM] Membuat explainer baru...")

        try:
            cached = GradCAMExplainer(model)
            cached.warmup()

            _EXPLAINER_CACHE[key] = cached

        except Exception as exc:

            if isinstance(exc, GradCAMError):
                raise

            raise GradCAMError(
                f"Gagal menginisialisasi Grad-CAM: {exc}"
            ) from exc

    return cached


def _to_rgb_array(image):

    if isinstance(image, Image.Image):

        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

    elif isinstance(image, np.ndarray):

        rgb = image

        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                f"Gambar harus RGB HWC. Shape: {rgb.shape}"
            )

        if rgb.dtype != np.uint8:
            rgb = np.clip(
                rgb,
                0,
                255,
            ).astype(np.uint8)

    else:
        raise ValueError(
            "Gambar harus PIL Image atau numpy array."
        )

    if rgb.size == 0:
        raise ValueError("Gambar kosong.")

    return np.ascontiguousarray(rgb)


def generate_gradcam(
    model,
    img_array,
    original_image,
    target_class_gerd=True,
    explainer=None,
):

    if not isinstance(img_array, np.ndarray):
        raise ValueError("img_array harus numpy array.")

    if img_array.shape != (456, 456, 3):
        raise ValueError(
            f"Input harus (456,456,3), didapat {img_array.shape}"
        )

    if not np.isfinite(img_array).all():
        raise ValueError("Input mengandung NaN atau infinity.")

    img_array = np.ascontiguousarray(
        img_array,
        dtype=np.float32,
    )

    original_rgb = _to_rgb_array(original_image)

    height, width = original_rgb.shape[:2]

    exp = (
        explainer
        if explainer is not None
        else get_or_create_explainer(model)
    )

    if exp.model is not model:
        raise GradCAMError(
            "Explainer tidak cocok dengan model."
        )

    image_tensor = tf.convert_to_tensor(
        img_array[None, ...],
        dtype=tf.float32,
    )

    try:

        probability_tensor, cam_tensor, score_tensor = (
            exp._gradcam_step(
                image_tensor,
                tf.constant(bool(target_class_gerd)),
            )
        )

    except Exception as exc:

        if isinstance(exc, GradCAMError):
            raise

        raise GradCAMError(
            f"Kegagalan komputasi gradient: {exc}"
        ) from exc

    probability = float(
        probability_tensor.numpy()
    )

    cam = cam_tensor.numpy().astype(np.float32)

    score = float(score_tensor.numpy())

    print("\n========== GRAD-CAM DEBUG ==========")
    print(f"Model       : {model.name}")
    print(f"Target      : {'GERD' if target_class_gerd else 'NORMAL'}")
    print(f"P(GERD)     : {probability:.8f}")
    print(f"Target score: {score:.8f}")
    print(f"CAM shape   : {cam.shape}")
    print(f"CAM min     : {cam.min():.8f}")
    print(f"CAM max     : {cam.max():.8f}")
    print(f"CAM mean    : {cam.mean():.8f}")
    print(f"CAM std     : {cam.std():.8f}")
    print("====================================\n")

    if cam.ndim != 2:
        raise GradCAMError(
            f"CAM harus 2D, didapat {cam.shape}"
        )

    if not np.isfinite(cam).all():
        raise GradCAMError(
            "CAM mengandung NaN atau infinity."
        )

    if (cam.max()) <= 1e-8:
        raise GradCAMError(
            "CAM bernilai nol. Gradient tidak menghasilkan aktivasi."
        )
    # Simpan aktivasi asli sebelum proses visualisasi.
    cam_raw = cam.copy()    

    
    # 1. Resize CAM ke resolusi gambar asli.
    cam = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    # 2. Normalisasi berdasarkan persentil.
    # Mengurangi pengaruh aktivasi ekstrem.
    lower = np.percentile(cam, 5)
    upper = np.percentile(cam, 99)

    if upper - lower < 1e-8:
        raise GradCAMError(
            "Rentang aktivasi CAM terlalu kecil."
        )

    cam = np.clip(
        (cam - lower) / (upper - lower),
        0.0,
        1.0,
    )

    # 3. Tingkatkan visibilitas aktivasi rendah.
    gamma = 0.65
    cam = np.power(cam, gamma)

    # 4. Haluskan noise tanpa menghilangkan detail.
    cam = cv2.GaussianBlur(
        cam,
        (3, 3),
        sigmaX=0.6,
    )

    cam = np.clip(cam, 0.0, 1.0)

    # 5. Buat heatmap dengan JET.
    heatmap = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET,
    )

    # 6. Buat overlay.
    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR,
    ).astype(np.float32)

    heatmap_float = heatmap.astype(np.float32)

    # Aktivasi lebih tinggi mendapat warna lebih dominan.
    alpha = (
        0.50
        * np.clip(cam * 1.25, 0.0, 1.0)
    )[..., np.newaxis]

    overlay = (
        original_bgr * (1.0 - alpha)
        + heatmap_float * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)


    return {
    "heatmap": heatmap,
    "overlay": overlay,
    "cam_raw": cam_raw.astype(np.float32),
    "score": score,
    "p_gerd": probability,
    "target": (
        "GERD" if target_class_gerd
        else "GERD_Normal"
    ),
    "prediction": (
        "GERD" if probability >= 0.5
        else "GERD_Normal"
    ),
}
