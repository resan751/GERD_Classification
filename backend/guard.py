
"""GERDGuard Image Type Gate and preprocessing utilities."""

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import (
    Image,
    UnidentifiedImageError,
)


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_FORMATS = (
    "JPEG",
    "PNG",
    "BMP",
)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_PIXELS = 50_000_000
MIN_SIDE = 32

GATE_SIZE = (380, 380)
CLASSIFIER_SIZE = (456, 456)


# ============================================================
# EXCEPTIONS
# ============================================================

class InvalidImageError(ValueError):
    """Gambar tidak valid atau tidak memenuhi persyaratan."""


# ============================================================
# IMAGE VALIDATION
# ============================================================

def _load_rgb(
    image_bytes: bytes,
) -> Image.Image:

    if not image_bytes:
        raise InvalidImageError(
            "File gambar kosong."
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise InvalidImageError(
            "Ukuran gambar maksimal 10 MB."
        )

    try:
        with Image.open(
            io.BytesIO(image_bytes)
        ) as image:

            image_format = image.format

            if image_format not in ALLOWED_FORMATS:
                raise InvalidImageError(
                    "Format gambar tidak didukung. "
                    "Gunakan JPG, PNG, atau BMP."
                )

            width, height = image.size

            if (
                width * height > MAX_PIXELS
                or min(width, height) < MIN_SIDE
            ):
                raise InvalidImageError(
                    f"Resolusi {width}x{height} tidak memenuhi "
                    f"batas. Minimal sisi {MIN_SIDE}px dan "
                    f"maksimal {MAX_PIXELS:,} piksel."
                )

            image.load()

            return image.convert("RGB")

    except InvalidImageError:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:

        raise InvalidImageError(
            "File bukan gambar valid atau rusak."
        ) from exc


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def _resize_to_array(
    image: Image.Image,
    size: tuple[int, int],
) -> np.ndarray:
    """Resize menjadi RGB float32 dengan rentang 0-255."""

    resized = image.resize(
        (size[1], size[0]),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        resized,
        dtype=np.float32,
    )

    if array.shape != (
        size[0],
        size[1],
        3,
    ):
        raise RuntimeError(
            f"Preprocessing menghasilkan shape tidak valid: "
            f"{array.shape}"
        )

    if not np.isfinite(array).all():
        raise RuntimeError(
            "Preprocessing menghasilkan nilai NaN atau Inf."
        )

    return array


# ============================================================
# GERD GUARD
# ============================================================

class GERDGuard:
    """Image Type Gate EfficientNetB4 dan preprocessing B5."""

    def __init__(
        self,
        gate_model_path: str | Path,
        config_path: str | Path,
    ):

        gate_model_path = Path(
            gate_model_path
        )

        config_path = Path(
            config_path
        )

        if not gate_model_path.is_file():
            raise FileNotFoundError(
                f"Gate model tidak ditemukan: {gate_model_path}"
            )

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Gate config tidak ditemukan: {config_path}"
            )

        # ----------------------------------------------------
        # LOAD CONFIGURATION
        # ----------------------------------------------------

        try:
            with open(
                config_path,
                "r",
                encoding="utf-8",
            ) as file:

                self.cfg = json.load(file)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Format JSON gate config tidak valid: {exc}"
            ) from exc

        size = self.cfg.get(
            "input_size",
            [380, 380],
        )

        if (
            not isinstance(size, list)
            or len(size) != 2
        ):
            raise RuntimeError(
                "input_size pada konfigurasi gate harus "
                "berisi [height, width]."
            )

        self.img_size = (
            int(size[0]),
            int(size[1]),
        )

        if self.img_size != GATE_SIZE:
            raise RuntimeError(
                f"Ukuran gate harus {GATE_SIZE}, "
                f"konfigurasi saat ini {self.img_size}."
            )

        self.threshold = float(
            self.cfg.get(
                "non_endoscopy_threshold",
                0.01,
            )
        )

        if not 0.0 <= self.threshold <= 1.0:
            raise RuntimeError(
                f"Gate threshold tidak valid: {self.threshold}"
            )

        # ----------------------------------------------------
        # LOAD GATE MODEL
        # ----------------------------------------------------

        print("[GERDGuard] Loading Image Type Gate...")

        try:
            self.gate = tf.keras.models.load_model(
                str(gate_model_path),
                compile=False,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Gagal memuat Image Type Gate: {exc}"
            ) from exc

        if tuple(
            self.gate.input_shape[1:]
        ) != (*self.img_size, 3):

            raise RuntimeError(
                f"Input Gate tidak sesuai. "
                f"Model: {self.gate.input_shape}, "
                f"Expected: (None, {self.img_size[0]}, "
                f"{self.img_size[1]}, 3)."
            )

        if self.gate.output_shape[-1] != 1:
            raise RuntimeError(
                "Output Image Type Gate harus satu neuron sigmoid."
            )

        # ----------------------------------------------------
        # FAST INFERENCE
        # ----------------------------------------------------

        @tf.function(reduce_retracing=True)
        def _fast_gate_call(
            x: tf.Tensor,
        ) -> tf.Tensor:

            return self.gate(
                x,
                training=False,
            )

        self._fast_gate_call = _fast_gate_call

        print("[GERDGuard] Image Type Gate loaded.")
        print("[GERDGuard] Input size:", self.img_size)
        print("[GERDGuard] Threshold:", self.threshold)

    # ========================================================
    # WARMUP
    # ========================================================

    def warmup(self):

        dummy = tf.zeros(
            (1, *self.img_size, 3),
            dtype=tf.float32,
        )

        self._fast_gate_call(dummy)

    # ========================================================
    # IMAGE TYPE GATE
    # ========================================================

    def check_bytes(
        self,
        image_bytes: bytes,
    ) -> tuple[dict, Image.Image | None]:

        # Decode dan validasi gambar.
        pil_image = _load_rgb(
            image_bytes
        )

        # Preprocessing khusus EfficientNetB4.
        gate_array = _resize_to_array(
            pil_image,
            self.img_size,
        )

        gate_tensor = tf.convert_to_tensor(
            gate_array[None, ...],
            dtype=tf.float32,
        )

        try:
            prediction = self._fast_gate_call(
                gate_tensor
            )

            score = float(
                np.asarray(prediction).reshape(-1)[0]
            )

        except Exception as exc:
            raise RuntimeError(
                f"Image Type Gate gagal melakukan inference: {exc}"
            ) from exc

        if not np.isfinite(score):
            raise RuntimeError(
                f"Output Image Type Gate tidak valid: {score}"
            )

        score = float(
            np.clip(score, 0.0, 1.0)
        )

        # Kelas 0 = endoscopy.
        # Kelas 1 = non-endoscopy.
        if score >= self.threshold:

            return {
                "status": "DITOLAK",
                "alasan": (
                    "Gambar terdeteksi sebagai non-endoskopi "
                    "oleh Image Type Gate."
                ),
                "skor_non_endoscopy": round(score, 6),
                "ambang_non_endoscopy": self.threshold,
            }, None

        return {
            "status": "LOLOS_GATE",
            "skor_non_endoscopy": round(score, 6),
            "ambang_non_endoscopy": self.threshold,
        }, pil_image

    # ========================================================
    # EFFICIENTNETB5 PREPROCESSING
    # ========================================================

    def prepare_classifier_image(
        self,
        source: bytes | Image.Image,
    ) -> np.ndarray:

        if isinstance(source, bytes):

            pil_image = _load_rgb(
                source
            )

        elif isinstance(source, Image.Image):

            pil_image = (
                source
                if source.mode == "RGB"
                else source.convert("RGB")
            )

        else:

            raise TypeError(
                "source harus berupa bytes atau PIL.Image.Image. "
                f"Input diterima: {type(source).__name__}"
            )

        return _resize_to_array(
            pil_image,
            CLASSIFIER_SIZE,
        )
