"""
Script Pembantu untuk Ekspor Prediksi Citra ke Format CSV RapidMiner
====================================================================
Cara Kerja:
1. Membaca gambar dari folder uji (misal: folder berisi subfolder GERD/ dan GERD_Normal/)
2. Menjalankan inferensi menggunakan model_gerd_combined.keras (atau via API FastAPI)
3. Menyimpan hasil ke 'hasil_prediksi_rapidminer.csv' yang siap diimpor ke RapidMiner Studio
   untuk dianalisis menggunakan operator 'Performance (Binominal Classification)'.

Penggunaan:
    python export_for_rapidminer.py --folder "path/ke/folder_test" --out "hasil_rapidminer.csv"
"""

import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "model" / "gerd" / "gerdguard_b5_best.keras"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def process_image(img_path: Path, target_size=(456, 456)):
    """Memuat dan mengubah ukuran gambar ke format input model (456x456x3)."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
    return arr


def run_batch_evaluation(data_dir: Path, model_path: Path, output_csv: Path, threshold=0.50):
    import tensorflow as tf

    print(f"[1/3] Memuat model dari: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    print(f"[2/3] Memindai gambar di: {data_dir}")
    image_paths = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if not image_paths:
        print(f"Peringatan: Tidak ditemukan gambar ({', '.join(SUPPORTED_EXTS)}) di {data_dir}")
        return

    print(f"Ditemukan {len(image_paths)} citra. Menjalankan inferensi...")

    rows = []
    for idx, path in enumerate(image_paths, 1):
        try:
            arr = process_image(path)
            batch = np.expand_dims(arr, axis=0)
            score = float(np.asarray(model(batch, training=False)).reshape(-1)[0])
            score = float(np.clip(score, 0.0, 1.0))

            pred = "GERD" if score >= threshold else "GERD_Normal"
            p_gerd = round(score, 6)
            p_normal = round(1.0 - score, 6)

            # Deteksi label aktual dari nama folder induk (jika terstruktur folder class)
            parent_name = path.parent.name.lower()
            if "gerd" in parent_name and "normal" in parent_name:
                actual = "GERD_Normal"
            elif "gerd" in parent_name:
                actual = "GERD"
            elif "normal" in parent_name:
                actual = "GERD_Normal"
            else:
                actual = "UNKNOWN"

            rows.append({
                "id": idx,
                "filename": path.name,
                "relative_path": str(path.relative_to(data_dir)),
                "actual_label": actual,
                "prediction": pred,
                "confidence_GERD": p_gerd,
                "confidence_Normal": p_normal,
                "confidence": p_gerd if pred == "GERD" else p_normal,
                "threshold": threshold,
            })

            if idx % 50 == 0 or idx == len(image_paths):
                print(f"  Diproses: {idx}/{len(image_paths)} citra")

        except Exception as e:
            print(f"  Gagal memproses {path.name}: {e}")

    # Simpan ke CSV
    print(f"[3/3] Menyimpan hasil ke: {output_csv}")
    fieldnames = [
        "id", "filename", "relative_path", "actual_label",
        "prediction", "confidence_GERD", "confidence_Normal",
        "confidence", "threshold"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Selesai! Berkas CSV siap digunakan di RapidMiner Studio.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekspor Prediksi GERD untuk RapidMiner")
    parser.add_argument("--folder", type=str, default=str(BASE_DIR / "testing_guard"),
                        help="Folder dataset uji (dapat berisi subfolder kelas)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Path ke file model_gerd_efficientnetb5.keras")
    parser.add_argument("--out", type=str, default=str(BASE_DIR / "hasil_prediksi_rapidminer.csv"),
                        help="Path output file CSV")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Ambang batas probabilitas (default: 0.50)")

    args = parser.parse_args()
    run_batch_evaluation(Path(args.folder), Path(args.model), Path(args.out), args.threshold)
